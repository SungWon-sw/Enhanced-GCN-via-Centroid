"""
models/ 안의 모든 변형을 한 프로세스에서 순차 학습하는 통합 러너.

제외: GCN+SD (.gitkeep 뿐이고 요청에 따라 제외)

포함 모델
---------
    ogbg-ppa   : TCD, GCN+CD, SD, GCN+, GCN_ppa
    ogbg-code2 : CD, GCN_code2

모델 폴더마다 model.py / dataset.py / utils.py 가 같은 플랫 이름으로 있어서
한 프로세스에서 같이 import 하면 서로를 덮어씁니다. 그래서 모델 하나를 돌리는
동안에만 그 폴더를 sys.path 에 올리고, 끝나면 그 폴더에서 온 모듈을
sys.modules 에서 걷어냅니다 (isolated_sys_path).

주의: DataLoader worker 가 돌려주는 배치를 언피클할 때 utils_vn_connect.VNData
같은 클래스를 다시 import 해야 하므로, 격리 구간은 "학습이 끝날 때까지" 유지해야
합니다. 빌드만 감싸면 안 됩니다.

실행 (반드시 레포 루트에서)
---------------------------
    python src/train_all.py                          # 전부 순차 실행
    python src/train_all.py --list                   # 모델 목록만 보기
    python src/train_all.py --models TCD,SD          # 일부만
    python src/train_all.py --smoke                  # 배선 확인 (1 epoch, 5 batch)
    python src/train_all.py --epochs 100             # 모든 모델 epoch 통일 (공정 비교용)
    python src/train_all.py --cuda 3 --seed 1        # GPU / 시드 지정

결과: outputs/<모델명>/ 에 체크포인트와 로그, outputs/results.json 에 요약.
"""

import argparse
import contextlib
import copy
import json
import math
import os
import random
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn, update_bn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / 'models'
OUT_DIR = ROOT / 'outputs'
DEFAULT_CONFIG = Path(__file__).resolve().parent / 'configs' / 'config.yaml'

# DataLoader(num_workers>0) 가 워커로 텐서를 넘길 때 torch 내부에서 뜨는 경고. 무해.
warnings.filterwarnings('ignore', message='TypedStorage is deprecated')

# 구버전 체크포인트 로드 호환 (train.py 원본과 동일한 패치)
_original_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_load(*args, **kwargs)


torch.load = _patched_load

# Ampere+ 에서 공짜 가속
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


# ======================================================================
# 모듈 격리
# ======================================================================
@contextlib.contextmanager
def isolated_sys_path(*paths):
    """paths 를 sys.path 앞에 얹고, 빠져나올 때 그 경로에서 온 모듈을 제거."""
    resolved = [str(Path(p).resolve()) for p in paths]
    before = set(sys.modules)
    for p in reversed(resolved):
        sys.path.insert(0, p)
    try:
        yield
    finally:
        for p in resolved:
            while p in sys.path:
                sys.path.remove(p)
        for name in list(sys.modules):
            if name in before:
                continue
            mod = sys.modules.get(name)
            path = getattr(mod, '__file__', None) or ''
            if any(path.startswith(p + os.sep) for p in resolved):
                del sys.modules[name]


class Tee:
    """stdout 을 화면과 로그 파일에 동시에 씀."""

    def __init__(self, path):
        self.path = path
        self.handle = None
        self.stdout = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.path, 'a', encoding='utf-8', buffering=1)
        self.stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, *exc):
        sys.stdout = self.stdout
        self.handle.close()
        return False

    def write(self, text):
        self.stdout.write(text)
        self.handle.write(text)

    def flush(self):
        self.stdout.flush()
        self.handle.flush()


# ======================================================================
# config
# ======================================================================
def deep_merge(base, over):
    out = copy.deepcopy(base)
    for key, value in (over or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_configs(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}
    common = raw.get('common', {})
    overlays = raw.get('models', {})
    return common, overlays


def config_for(name, common, overlays):
    cfg = deep_merge(common, overlays.get(name, {}))
    cfg.setdefault('data', {}).setdefault('dir', 'dataset')
    return cfg


# ======================================================================
# 공통 유틸 (원본 train.py 에서 가져옴)
# ======================================================================
def set_seed(seed):
    """multi-seed 평균±std 용도. bitwise 결정성까지는 안 감 (cudnn.benchmark 유지)."""
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(cfg):
    cuda_num = int(cfg.get('cuda', {}).get('cuda_number', 0))
    if cuda_num == -1:
        return torch.device('cpu')
    if not torch.cuda.is_available():
        raise RuntimeError(
            f'CUDA를 사용할 수 없습니다 (torch={torch.__version__}, '
            f'CUDA build={torch.version.cuda}). CPU 실행은 cuda_number: -1 로 명시하세요.'
        )
    if not 0 <= cuda_num < torch.cuda.device_count():
        raise ValueError(
            f'cuda_number={cuda_num}: 사용 가능한 GPU 번호는 '
            f'0~{torch.cuda.device_count() - 1} 입니다.'
        )
    device = torch.device(f'cuda:{cuda_num}')
    torch.empty(1, device=device)   # 데이터 로드 전에 실제 CUDA 초기화 확인
    return device


def make_scheduler(optimizer, warmup_epochs, total_epochs):
    """warmup_epochs 선형 warmup 후 cosine 으로 0 까지 감소 (epoch 단위)."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def to_device(data, device):
    """배치를 GPU 로 비동기 전송. VN 총 개수는 CPU 텐서일 때 python int 로 미리 뽑아
    forward 안의 GPU->CPU 동기화(.max())를 제거."""
    if getattr(data, 'num_vn', None) is not None:
        data.num_vn_total = int(data.num_vn.sum())
    return data.to(device, non_blocking=True)


@contextlib.contextmanager
def maybe_autocast(amp_dtype):
    if amp_dtype is None:
        yield
    else:
        with torch.autocast('cuda', dtype=amp_dtype):
            yield


def make_optimizer(cfg, model, device):
    tcfg = cfg['train']
    name = str(tcfg.get('optimizer', 'adam')).lower()
    opt_cls = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}[name]
    kw = dict(lr=tcfg['lr'], weight_decay=tcfg.get('weight_decay', 0.0))
    if device.type == 'cuda':
        kw['fused'] = True
    return opt_cls(model.parameters(), **kw)


def make_loaders(dataset, split_idx, cfg):
    """split 인덱스로 train/val/test DataLoader 생성."""
    tcfg = cfg['train']
    num_workers = int(tcfg.get('num_workers', 0))
    kw = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available())
    if num_workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=4)
    batch_size = int(tcfg['batch_size'])

    def loader(key, shuffle):
        return DataLoader(Subset(dataset, split_idx[key]), batch_size=batch_size,
                          shuffle=shuffle, **kw)

    return loader('train', True), loader('valid', False), loader('test', False)


def inner_data(dataset):
    """InMemoryDataset 의 내부 Data 객체 (PyG 버전에 따라 .data / ._data)."""
    for attr in ('data', '_data'):
        obj = getattr(dataset, attr, None)
        if obj is not None and hasattr(obj, 'keys'):
            return obj
    return None


def train_labels(dataset, idx):
    """code2 vocab 생성용 라벨 리스트. 가능하면 내부 data.y 직접 접근 (훨씬 빠름)."""
    data = inner_data(dataset)
    if data is not None and getattr(data, 'y', None) is not None:
        try:
            ys = data.y
            return [ys[int(i)] for i in idx]
        except Exception:
            pass
    return [dataset[int(i)].y for i in idx]


# picklable transform (num_workers>0 + spawn 환경에서도 안전)
def add_ppa_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data


class SDTransform:
    def __init__(self, prepare_graph, cfg):
        self.prepare_graph = prepare_graph
        self.cfg = cfg

    def __call__(self, data):
        return self.prepare_graph(data, self.cfg)


class Code2Transform:
    """augment_edge -> encode_y_to_arr 를 한 번에."""

    def __init__(self, augment_edge, encode_y_to_arr, vocab2idx, max_seq_len):
        self.augment_edge = augment_edge
        self.encode_y_to_arr = encode_y_to_arr
        self.vocab2idx = vocab2idx
        self.max_seq_len = max_seq_len

    def __call__(self, data):
        return self.encode_y_to_arr(self.augment_edge(data), self.vocab2idx,
                                    self.max_seq_len)


class ArrToSeq:
    def __init__(self, decode_arr_to_seq, idx2vocab):
        self.decode_arr_to_seq = decode_arr_to_seq
        self.idx2vocab = idx2vocab

    def __call__(self, arr):
        return self.decode_arr_to_seq(arr, self.idx2vocab)


# ======================================================================
# 학습 / 평가 — ogbg-ppa (단일 라벨 37-way, metric=acc)
# ======================================================================
def train_epoch_ppa(model, loader, optimizer, criterion, device,
                    amp_dtype=None, grad_clip=None, ema=None, limit=None):
    model.train()
    total_loss = torch.zeros((), device=device)
    total_graphs = 0

    for step, data in enumerate(loader):
        if limit is not None and step >= limit:
            break
        data = to_device(data, device)
        optimizer.zero_grad(set_to_none=True)

        with maybe_autocast(amp_dtype):
            logits = model(data)
            labels = data.y.view(-1).long()
            loss = criterion(logits, labels)

        loss.backward()                      # autocast 밖; bf16 은 GradScaler 불필요
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update_parameters(model)

        batch_size = labels.size(0)
        total_loss += loss.detach().float() * batch_size
        total_graphs += batch_size

    return total_loss.item() / max(total_graphs, 1)


@torch.no_grad()
def evaluate_ppa(model, loader, evaluator, device, amp_dtype=None, limit=None, **_):
    model.eval()
    y_true, y_pred = [], []

    for step, data in enumerate(loader):
        if limit is not None and step >= limit:
            break
        data = to_device(data, device)
        with maybe_autocast(amp_dtype):
            logits = model(data)
        y_true.append(data.y.view(-1, 1))
        y_pred.append(logits.argmax(dim=1, keepdim=True))

    if not y_true:
        return float('nan')
    y_true = torch.cat(y_true, dim=0).cpu()
    y_pred = torch.cat(y_pred, dim=0).cpu()
    return evaluator.eval({'y_true': y_true, 'y_pred': y_pred})['acc']


# ======================================================================
# 학습 / 평가 — ogbg-code2 (max_seq_len 개 헤드, metric=F1)
# ======================================================================
def train_epoch_code2(model, loader, optimizer, criterion, device,
                      amp_dtype=None, grad_clip=None, ema=None, limit=None):
    model.train()
    total_loss = torch.zeros((), device=device)
    num_batches = 0

    for step, data in enumerate(loader):
        if limit is not None and step >= limit:
            break
        data = to_device(data, device)
        # 노드가 1개거나 그래프가 1개인 배치는 BatchNorm 이 터지므로 건너뜀 (OGB 참조 구현과 동일)
        if data.x.size(0) == 1 or int(data.batch[-1]) == 0:
            continue

        optimizer.zero_grad(set_to_none=True)
        with maybe_autocast(amp_dtype):
            pred_list = model(data)
            loss = sum(criterion(pred.float(), data.y_arr[:, i])
                       for i, pred in enumerate(pred_list)) / len(pred_list)

        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update_parameters(model)

        total_loss += loss.detach().float()
        num_batches += 1

    return total_loss.item() / max(num_batches, 1)


@torch.no_grad()
def evaluate_code2(model, loader, evaluator, device, amp_dtype=None, limit=None,
                   arr_to_seq=None):
    model.eval()
    seq_ref, seq_pred = [], []

    for step, data in enumerate(loader):
        if limit is not None and step >= limit:
            break
        data = to_device(data, device)
        if data.x.size(0) == 1:
            continue
        with maybe_autocast(amp_dtype):
            pred_list = model(data)
        mat = torch.cat([pred.argmax(dim=1).view(-1, 1) for pred in pred_list],
                        dim=1).cpu()
        seq_pred.extend(arr_to_seq(arr) for arr in mat)
        seq_ref.extend(data.y[i] for i in range(len(data.y)))

    if not seq_ref:
        return float('nan')
    return evaluator.eval({'seq_ref': seq_ref, 'seq_pred': seq_pred})['F1']


TASK_FNS = {
    'ppa':   (train_epoch_ppa,   evaluate_ppa,   'acc'),
    'code2': (train_epoch_code2, evaluate_code2, 'F1'),
}


# ======================================================================
# 모델별 어댑터
#   각 함수는 isolated_sys_path 안에서 호출되며
#   (loaders, model, task, extra) 를 돌려준다.
# ======================================================================
def build_cfg_vn(cfg):
    """TCD, GCN+CD — 자체 dataset.build_loaders + GCN(cfg, PPANodeEncoder, num_classes)."""
    import dataset as ds_mod
    import model as model_mod
    import utils as utils_mod

    train_loader, val_loader, test_loader, _num_tasks, num_classes = ds_mod.build_loaders(cfg)
    net = model_mod.GCN(
        cfg=cfg,
        node_encoder=utils_mod.PPANodeEncoder(cfg['train']['emb_dim']),
        num_classes=int(num_classes),
    )
    return (train_loader, val_loader, test_loader), net, 'ppa', {}


def build_gcn_plus(cfg):
    """GCN+ — VN 없는 순수 GCN+ baseline (dataset_ppa + model_baseline)."""
    import dataset_ppa
    import model_baseline

    train_loader, val_loader, test_loader, num_tasks, edge_dim = dataset_ppa.build_loaders(cfg)
    net = model_baseline.GCNGraphClassifier(
        cfg=cfg,
        num_tasks=int(num_tasks),
        edge_dim=int(edge_dim),
        node_encoder=None,          # ppa 는 노드 피처가 없어 더미 임베딩 사용
    )
    return (train_loader, val_loader, test_loader), net, 'ppa', {}


def build_sd(cfg):
    """SD — 패키지 형태(prepare_graph transform + build_model). dataset.py 가 없어 여기서 로더 구성."""
    import SD

    dataset = PygGraphPropPredDataset(
        name=cfg['data']['dataset_name'],
        root=cfg['data']['dir'],
        transform=SDTransform(SD.prepare_graph, cfg),   # x=0 채우기까지 내부에서 처리
    )
    split_idx = dataset.get_idx_split()
    num_classes = int(dataset.num_classes)
    loaders = make_loaders(dataset, split_idx, cfg)
    net = SD.build_model(cfg, num_classes=num_classes)
    return loaders, net, 'ppa', {}


def build_gcn_ppa_official(cfg):
    """GCN_ppa — OGB 공식 baseline (argparse 대신 config 값을 넣어 gnn.GNN 직접 생성)."""
    import gnn

    dataset = PygGraphPropPredDataset(
        name=cfg['data']['dataset_name'],
        root=cfg['data']['dir'],
        transform=add_ppa_zeros,
    )
    split_idx = dataset.get_idx_split()
    num_classes = int(dataset.num_classes)
    loaders = make_loaders(dataset, split_idx, cfg)

    mcfg = cfg.get('model', {})
    net = gnn.GNN(
        num_class=num_classes,
        num_layer=int(mcfg.get('num_layers', 5)),
        emb_dim=int(cfg['train']['emb_dim']),
        gnn_type=str(mcfg.get('gnn_type', 'gcn')),
        virtual_node=bool(mcfg.get('virtual_node', False)),
        drop_ratio=float(cfg['train'].get('drop_ratio', 0.5)),
    )
    return loaders, net, 'ppa', {}


def build_cd(cfg):
    """CD — ogbg-code2 + centroid tree 에지.

    CD/src/dataset.py 의 build_loaders 는 쓰지 않는다. 그 안에
        dataset, depth_data = convert_into_easy(dataset)
    가 있는데 convert_into_easy 는 값을 하나만 돌려줘서 언패킹에서 깨진다
    (utils.py 149~165 참고). 같은 파이프라인을 여기서 올바르게 구성한다.
    upstream 을 고치면 ds_mod.build_loaders(cfg) 호출로 바꿔도 된다.
    """
    import model as model_mod
    import utils as utils_mod

    tcfg = cfg['train']
    dataset = PygGraphPropPredDataset(
        name=cfg['data']['dataset_name'],
        root=cfg['data']['dir'],
    )
    split_idx = dataset.get_idx_split()

    vocab2idx, idx2vocab = utils_mod.get_vocab_mapping(
        train_labels(dataset, split_idx['train']), int(tcfg['num_vocab'])
    )
    dataset.transform = Code2Transform(
        utils_mod.augment_edge, utils_mod.encode_y_to_arr,
        vocab2idx, int(tcfg['max_seq_len']),
    )

    print('centroid tree 전처리 중 (code2 전체를 메모리에 올립니다 — 시간/메모리 주의)...')
    easy = utils_mod.convert_into_easy(dataset)
    loaders = make_loaders(easy, split_idx, cfg)

    data = inner_data(easy)
    num_nodetypes = int(data.x[:, 0].max().item()) + 1
    num_nodeattributes = int(data.x[:, 1].max().item()) + 1

    encoder = utils_mod.ASTNodeEncoder(
        int(tcfg['emb_dim']),
        num_nodetypes=num_nodetypes,
        num_nodeattributes=num_nodeattributes,
        max_depth=int(tcfg.get('max_depth', 20)),
    )
    net = model_mod.GCN(cfg=cfg, node_encoder=encoder, num_tasks=len(vocab2idx))
    return loaders, net, 'code2', {
        'arr_to_seq': ArrToSeq(utils_mod.decode_arr_to_seq, idx2vocab)
    }


def build_gcn_code2_official(cfg):
    """GCN_code2 — OGB 공식 code2 baseline."""
    import pandas as pd

    import gnn
    import utils as utils_mod

    tcfg = cfg['train']
    dataset = PygGraphPropPredDataset(
        name=cfg['data']['dataset_name'],
        root=cfg['data']['dir'],
    )
    split_idx = dataset.get_idx_split()

    vocab2idx, idx2vocab = utils_mod.get_vocab_mapping(
        train_labels(dataset, split_idx['train']), int(tcfg['num_vocab'])
    )
    dataset.transform = Code2Transform(
        utils_mod.augment_edge, utils_mod.encode_y_to_arr,
        vocab2idx, int(tcfg['max_seq_len']),
    )
    loaders = make_loaders(dataset, split_idx, cfg)

    # 공식 구현과 동일하게 mapping 파일에서 타입/속성 개수를 읽음
    nodetypes = pd.read_csv(os.path.join(dataset.root, 'mapping', 'typeidx2type.csv.gz'))
    nodeattrs = pd.read_csv(os.path.join(dataset.root, 'mapping', 'attridx2attr.csv.gz'))
    encoder = utils_mod.ASTNodeEncoder(
        int(tcfg['emb_dim']),
        num_nodetypes=len(nodetypes['type']),
        num_nodeattributes=len(nodeattrs['attr']),
        max_depth=int(tcfg.get('max_depth', 20)),
    )

    mcfg = cfg.get('model', {})
    net = gnn.GNN(
        num_vocab=len(vocab2idx),
        max_seq_len=int(tcfg['max_seq_len']),
        node_encoder=encoder,
        num_layer=int(mcfg.get('num_layers', 5)),
        gnn_type=str(mcfg.get('gnn_type', 'gcn')),
        virtual_node=bool(mcfg.get('virtual_node', False)),
        emb_dim=int(tcfg['emb_dim']),
        drop_ratio=float(tcfg.get('drop_ratio', 0.0)),
    )
    return loaders, net, 'code2', {
        'arr_to_seq': ArrToSeq(utils_mod.decode_arr_to_seq, idx2vocab)
    }


# name -> (sys.path 에 올릴 경로들, 빌더)
REGISTRY = {
    'TCD':       (lambda: [MODELS_DIR / 'TCD' / 'src'],    build_cfg_vn),
    'GCN+CD':    (lambda: [MODELS_DIR / 'GCN+CD' / 'src'], build_cfg_vn),
    'SD':        (lambda: [MODELS_DIR],                    build_sd),
    'GCN+':      (lambda: [MODELS_DIR / 'GCN+' / 'src'],   build_gcn_plus),
    'GCN_ppa':   (lambda: [MODELS_DIR / 'GCN_ppa'],        build_gcn_ppa_official),
    'CD':        (lambda: [MODELS_DIR / 'CD' / 'src'],     build_cd),
    'GCN_code2': (lambda: [MODELS_DIR / 'GCN_code2'],      build_gcn_code2_official),
}
DEFAULT_ORDER = ['TCD', 'GCN+CD', 'SD', 'GCN+', 'GCN_ppa', 'CD', 'GCN_code2']


# ======================================================================
# 한 모델 학습
# ======================================================================
def run_one(name, cfg, args):
    paths, builder = REGISTRY[name]
    out_dir = OUT_DIR / name.replace('+', 'plus')
    out_dir.mkdir(parents=True, exist_ok=True)

    tcfg = cfg['train']
    seed = args.seed if args.seed is not None else tcfg.get('seed', 0)
    set_seed(seed)
    device = select_device(cfg)

    epochs = int(args.epochs if args.epochs is not None else tcfg.get('epochs', 50))
    warmup = min(int(tcfg.get('warmup_epochs', 0)), epochs)
    grad_clip = tcfg.get('grad_clip', None)
    ema_decay = tcfg.get('ema_decay', None)
    ema_decay = None if ema_decay is None else float(ema_decay)
    use_amp = bool(tcfg.get('amp', False)) and device.type == 'cuda'
    amp_dtype = torch.bfloat16 if use_amp else None

    print(f'[{name}] device={device} seed={seed} dataset={cfg["data"]["dataset_name"]}')

    with isolated_sys_path(*paths()):
        loaders, model, task, extra = builder(cfg)
        train_loader, val_loader, test_loader = loaders
        train_fn, eval_fn, metric = TASK_FNS[task]

        model = model.to(device)
        optimizer = make_optimizer(cfg, model, device)
        criterion = nn.CrossEntropyLoss()
        evaluator = Evaluator(cfg['data']['dataset_name'])
        scheduler = make_scheduler(optimizer, warmup, epochs)

        ema = None
        if ema_decay is not None:
            ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))

        print(f'[{name}] params={sum(p.numel() for p in model.parameters()):,} '
              f'epochs={epochs} warmup={warmup} amp={"bf16" if use_amp else "off"} '
              f'grad_clip={grad_clip} ema_decay={ema_decay} metric={metric}')

        ckpt = out_dir / str(cfg.get('model', {}).get('model_name', 'best_model.pt'))
        best_val = float('-inf')
        started = time.time()

        for epoch in range(1, epochs + 1):
            loss = train_fn(
                model, train_loader, optimizer, criterion, device,
                amp_dtype=amp_dtype, grad_clip=grad_clip,
                ema=(ema if ema is not None and epoch > warmup else None),
                limit=args.limit_batches,
            )
            val = eval_fn(model, val_loader, evaluator, device,
                          amp_dtype=amp_dtype, limit=args.limit_batches, **extra)
            scheduler.step()

            if val > best_val:
                best_val = val
                torch.save(model.state_dict(), ckpt)

            print(f'[{name}] epoch {epoch:03d} | loss {loss:.4f} | '
                  f'val {metric} {val:.4f} | best {best_val:.4f} | '
                  f'lr {optimizer.param_groups[0]["lr"]:.2e}')

        # --- raw best (val 로 고른 체크포인트) ---
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        val_raw = eval_fn(model, val_loader, evaluator, device,
                          amp_dtype=amp_dtype, limit=args.limit_batches, **extra)
        test_raw = eval_fn(model, test_loader, evaluator, device,
                           amp_dtype=amp_dtype, limit=args.limit_batches, **extra)
        result = {
            'model': name, 'task': task, 'metric': metric, 'seed': seed,
            'epochs': epochs, 'val': val_raw, 'test': test_raw,
            'selected': 'raw', 'minutes': round((time.time() - started) / 60, 2),
            'checkpoint': str(ckpt), 'status': 'ok',
        }
        print(f'[{name}] raw : val {val_raw:.4f} | test {test_raw:.4f}')

        # --- EMA (BN running stats 를 train 데이터로 재계산한 뒤 평가) ---
        if ema is not None:
            update_bn(train_loader, ema, device=device)
            val_ema = eval_fn(ema, val_loader, evaluator, device,
                              amp_dtype=amp_dtype, limit=args.limit_batches, **extra)
            test_ema = eval_fn(ema, test_loader, evaluator, device,
                               amp_dtype=amp_dtype, limit=args.limit_batches, **extra)
            torch.save(ema.module.state_dict(), out_dir / 'best_ema.pt')
            print(f'[{name}] ema : val {val_ema:.4f} | test {test_ema:.4f}')
            if val_ema >= val_raw:
                result.update(val=val_ema, test=test_ema, selected='ema')

        print(f'[{name}] ==> final test {result["metric"]} {result["test"]:.4f} '
              f'({result["selected"]}, selected by val)')

        # 모델/옵티마이저를 먼저 놓아줘야 다음 모델이 GPU 메모리를 온전히 쓸 수 있다
        del model, optimizer, ema, loaders, train_loader, val_loader, test_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


# ======================================================================
# main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description='models/ 안의 모든 모델을 순차 학습 (GCN+SD 제외)')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--models', default=None,
                        help='쉼표로 구분 (기본: ' + ','.join(DEFAULT_ORDER) + ')')
    parser.add_argument('--skip', default=None, help='제외할 모델 (쉼표 구분)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='모든 모델의 epoch 수를 통일 (공정 비교용)')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--cuda', type=int, default=None, help='cuda_number 덮어쓰기')
    parser.add_argument('--limit-batches', type=int, default=None,
                        help='epoch 당 배치 수 제한 (디버깅용)')
    parser.add_argument('--smoke', action='store_true',
                        help='--epochs 1 --limit-batches 5 와 동일 (배선 확인)')
    parser.add_argument('--list', action='store_true', help='모델 목록만 출력')
    args = parser.parse_args()

    if args.smoke:
        args.epochs = args.epochs or 1
        args.limit_batches = args.limit_batches or 5

    common, overlays = load_configs(args.config)

    names = [n.strip() for n in args.models.split(',')] if args.models else list(DEFAULT_ORDER)
    if args.skip:
        skip = {n.strip() for n in args.skip.split(',')}
        names = [n for n in names if n not in skip]
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        parser.error(f'모르는 모델: {unknown}. 가능한 값: {list(REGISTRY)}')

    if args.list:
        for name in DEFAULT_ORDER:
            cfg = config_for(name, common, overlays)
            print(f'{name:10s} {cfg["data"]["dataset_name"]:12s} '
                  f'epochs={cfg["train"].get("epochs")} '
                  f'emb_dim={cfg["train"].get("emb_dim")} '
                  f'layers={cfg.get("model", {}).get("num_layers")}')
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'실행 대상: {names}')

    results = []
    for name in names:
        cfg = config_for(name, common, overlays)
        if args.cuda is not None:
            cfg.setdefault('cuda', {})['cuda_number'] = args.cuda

        log_path = OUT_DIR / name.replace('+', 'plus') / 'train.log'
        print(f'\n{"=" * 70}\n{name}\n{"=" * 70}')
        try:
            with Tee(log_path):
                results.append(run_one(name, cfg, args))
        except Exception as exc:                      # 한 모델이 죽어도 나머지는 계속
            traceback.print_exc()
            print(f'[{name}] 실패: {type(exc).__name__}: {exc}')
            results.append({'model': name, 'status': 'failed',
                            'error': f'{type(exc).__name__}: {exc}'})
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        with open(OUT_DIR / 'results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f'\n{"=" * 70}\n요약\n{"=" * 70}')
    print(f'{"model":11s} {"task":6s} {"metric":7s} {"val":>8s} {"test":>8s} '
          f'{"min":>7s}  status')
    for r in results:
        if r.get('status') == 'ok':
            print(f'{r["model"]:11s} {r["task"]:6s} {r["metric"]:7s} '
                  f'{r["val"]:8.4f} {r["test"]:8.4f} {r["minutes"]:7.1f}  ok '
                  f'({r["selected"]})')
        else:
            print(f'{r["model"]:11s} {"-":6s} {"-":7s} {"-":>8s} {"-":>8s} '
                  f'{"-":>7s}  {r.get("error", "failed")}')
    print(f'\n상세: {OUT_DIR / "results.json"}')


if __name__ == '__main__':
    main()
