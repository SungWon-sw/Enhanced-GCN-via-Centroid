import warnings

import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset
from vn_cache import CachedVNDataset
from utils_pe import PEDataset

# DataLoader(num_workers>0)가 워커 프로세스로 텐서를 넘길 때 torch 내부(torch/_utils.py)
# 가 deprecated된 Tensor.storage()를 자체적으로 호출해서 뜨는 경고. 우리 코드가 아니라
# torch 내부 구현에서 발생하는 것이라 무해하며, 메시지만 걸러서 숨김.
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")


def add_ppa_node_features(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

def apply_pe(dataset, cfg):
    """위치 인코딩(RWSE): model.pos_encoding.enabled 일 때만 data.pe 를 붙임 (VN 캐시와 무관)."""
    pe_cfg = cfg.get('model', {}).get('pos_encoding', {})
    if pe_cfg.get('enabled', False):
        return PEDataset(dataset, int(pe_cfg.get('dim', 16)))
    return dataset


def prepare_dataset(cfg):
    """
    무거운 부분(원본 로드 + VN 캐시 래핑)만 수행하고 DataLoader는 만들지 않습니다.
    batch_size 같은 하이퍼파라미터를 바꿔도 이 함수는 다시 호출할 필요가 없도록
    분리 (Optuna 등 반복 튜닝에서 재사용).

    ogbg-ppa 전용: 단백질 연관 네트워크에 대한 다중 클래스(37-way) 분류
    데이터셋이라, code2의 vocab/시퀀스 전처리나 molhiv의 이진 라벨 전제가
    필요 없음.
    """
    dataset = PygGraphPropPredDataset(
        name=cfg['data']['dataset_name'],
        root=cfg['data']['dir'],
    )
    split_idx = dataset.get_idx_split()
    cached_dataset = CachedVNDataset(dataset)
    print(f'VN cache: {cached_dataset.cache_dir} (compute once on first access)')

    cached_dataset = apply_pe(cached_dataset, cfg)

    num_tasks = dataset.num_tasks                             # ppa는 1 (단일 다중클래스 태스크)
    num_classes = dataset.num_classes

    return cached_dataset, split_idx, num_tasks, num_classes


def make_loaders(cached_dataset, split_idx, batch_size, num_workers):
    """전처리(VN 캐시 래핑)가 끝난 dataset으로부터 (가벼운) DataLoader만 새로 만듭니다."""
    # pin_memory: non_blocking H2D 전송용. persistent_workers: 에폭/평가마다 워커 재생성 방지.
    kw = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available())
    if num_workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(
        Subset(cached_dataset, split_idx['train']),
        batch_size=batch_size,
        shuffle=True,
        **kw,
    )
    val_loader = DataLoader(
        Subset(cached_dataset, split_idx['valid']),
        batch_size=batch_size,
        shuffle=False,
        **kw,
    )
    test_loader = DataLoader(
        Subset(cached_dataset, split_idx['test']),
        batch_size=batch_size,
        shuffle=False,
        **kw,
    )
    return train_loader, val_loader, test_loader


def build_loaders(cfg):
    """
    config 설정을 받아 train, valid, test 데이터 로더와
    모델 빌드에 필요한 메타데이터(태스크 수, 클래스 수)를 반환합니다.
    """
    cached_dataset, split_idx, num_tasks, num_classes = prepare_dataset(cfg)

    train_loader, val_loader, test_loader = make_loaders(
        cached_dataset, split_idx,
        batch_size=cfg['train']['batch_size'],
        num_workers=cfg['train']['num_workers']
    )

    return train_loader, val_loader, test_loader, num_tasks, num_classes
