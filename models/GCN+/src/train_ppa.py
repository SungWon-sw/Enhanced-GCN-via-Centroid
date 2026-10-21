import math
import sys
import random

import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel, update_bn
from ogb.graphproppred import Evaluator

from model_baseline import GCNGraphClassifier
from dataset_ppa import build_loaders


def set_seed(seed):
    """RNG 시드 고정. original 브랜치(935a5eb)와 동일한 정책: None이면 건너뜀."""
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# 안전한 모델 로드를 위한 패치
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_load(*args, **kwargs)
torch.load = _patched_load

# Ampere+ 에서 공짜 가속 (main 브랜치 stash: WIP on main 8d35b75, VN과 무관한 부분)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


def load_config(config_path="configs/config_ppa.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def make_scheduler(optimizer, warmup_epochs, total_epochs):
    """warmup_epochs 선형 warmup 후 cosine으로 0까지 감소 (epoch 단위)."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, criterion, device,
                     amp_dtype=None, grad_clip=None, ema=None):
    model.train()
    total_loss = torch.zeros((), device=device)
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)

        label = data.y.view(-1).to(torch.long)
        if amp_dtype is not None:
            with torch.autocast('cuda', dtype=amp_dtype):
                loss = criterion(model(data), label)
        else:
            loss = criterion(model(data), label)

        loss.backward()  # autocast 밖; bf16은 GradScaler 불필요
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update_parameters(model)

        total_loss += loss.detach()  # 배치마다 .item()으로 동기화하지 않고 GPU에 누적
    return (total_loss / len(loader)).item()  # 동기화는 epoch 끝에 한 번만


@torch.no_grad()
def evaluate(model, loader, evaluator, device, amp_dtype=None):
    model.eval()
    y_true_batches, y_pred_batches = [], []
    for data in loader:
        data = data.to(device)
        if amp_dtype is not None:
            with torch.autocast('cuda', dtype=amp_dtype):
                logits = model(data)
        else:
            logits = model(data)
        pred = logits.float().argmax(dim=-1, keepdim=True)  # GPU에 둔 채로 모음
        y_pred_batches.append(pred)
        y_true_batches.append(data.y.view(-1, 1))

    y_true = torch.cat(y_true_batches, dim=0).cpu()  # 동기화(.cpu())는 전체 배치 끝난 뒤 한 번만
    y_pred = torch.cat(y_pred_batches, dim=0).cpu()
    return evaluator.eval({'y_true': y_true, 'y_pred': y_pred})['acc']


def main():
    cfg = load_config(sys.argv[1]) if len(sys.argv) > 1 else load_config()
    seed = cfg['train'].get('seed', 0)
    set_seed(seed)
    cuda_num = cfg['cuda']['cuda_number']
    device = torch.device(f'cuda:{cuda_num}' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    tcfg = cfg['train']
    epochs        = int(tcfg.get('epochs', 100))
    warmup_epochs = int(tcfg.get('warmup_epochs', 5))
    grad_clip     = tcfg.get('grad_clip', 1.0)
    ema_decay     = tcfg.get('ema_decay', 0.999)
    ema_decay     = None if ema_decay is None else float(ema_decay)
    use_amp       = bool(tcfg.get('amp', True)) and device.type == 'cuda'
    amp_dtype     = torch.bfloat16 if use_amp else None

    # VN/centroid 전처리 없이 원본 ogbg-ppa 그래프 그대로 로드
    train_loader, val_loader, test_loader, num_tasks, edge_dim = build_loaders(cfg)
    print(f'num_tasks(classes): {num_tasks}, edge_dim: {edge_dim}')

    model = GCNGraphClassifier(cfg=cfg, num_tasks=num_tasks, edge_dim=edge_dim).to(device)

    # adamw: GCN+ 와 동일한 decoupled weight decay. 기본값은 기존 동작(adam).
    opt_cls = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}[str(tcfg.get('optimizer', 'adam')).lower()]
    optimizer = opt_cls(model.parameters(), lr=tcfg['lr'], weight_decay=tcfg['weight_decay'])
    scheduler = make_scheduler(optimizer, warmup_epochs, epochs)
    criterion = nn.CrossEntropyLoss()
    evaluator = Evaluator(cfg['data']['dataset_name'])

    # 가중치 EMA: 파라미터만 평균, BN 통계는 학습 후 update_bn으로 재계산.
    # ema_decay: null이면 EMA 전체를 건너뜀 (raw 체크포인트만 사용).
    ema = None
    if ema_decay is not None:
        ema = AveragedModel(
            model,
            avg_fn=lambda avg, cur, n: ema_decay * avg + (1.0 - ema_decay) * cur,
        )

    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'epochs={epochs} warmup={warmup_epochs} '
          f'amp={"bf16" if use_amp else "off"} grad_clip={grad_clip} ema_decay={ema_decay}')

    best_acc = 0.0
    model_name = cfg['model']['model_name']
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            amp_dtype=amp_dtype, grad_clip=grad_clip,
            ema=(ema if ema is not None and epoch > warmup_epochs else None),
        )
        val_acc = evaluate(model, val_loader, evaluator, device, amp_dtype=amp_dtype)
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_name)

        lr_now = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | '
              f'Best: {best_acc:.4f} | lr: {lr_now:.2e}')

    # --- raw best (val로 고른 체크포인트) ---
    model.load_state_dict(torch.load(model_name))
    val_raw  = evaluate(model, val_loader, evaluator, device, amp_dtype=amp_dtype)
    test_raw = evaluate(model, test_loader, evaluator, device, amp_dtype=amp_dtype)

    if ema is None:
        print(f'raw : val {val_raw:.4f} | test {test_raw:.4f}')
        print(f'==> Final Test Acc: {test_raw:.4f} (raw)')
        return

    # --- EMA (BN running stats를 train 데이터로 재계산한 뒤 평가) ---
    update_bn(train_loader, ema, device=device)
    val_ema  = evaluate(ema, val_loader, evaluator, device, amp_dtype=amp_dtype)
    test_ema = evaluate(ema, test_loader, evaluator, device, amp_dtype=amp_dtype)
    torch.save(ema.module.state_dict(), 'best_ema_' + model_name)

    use_ema = val_ema >= val_raw
    print(f'raw : val {val_raw:.4f} | test {test_raw:.4f}')
    print(f'ema : val {val_ema:.4f} | test {test_ema:.4f}')
    print(f'==> Final Test Acc: {(test_ema if use_ema else test_raw):.4f} '
          f'({"ema" if use_ema else "raw"}, selected by val)')


if __name__ == "__main__":
    main()
