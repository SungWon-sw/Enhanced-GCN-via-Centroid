import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset
from vn_cache import CachedVNDataset


def add_ppa_node_features(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

def build_loaders(cfg):
    """
    config 설정을 받아 train, valid, test 데이터 로더와
    모델 빌드에 필요한 메타데이터(태스크 수, 클래스 수)를 반환합니다.

    ogbg-ppa 전용: 단백질 연관 네트워크에 대한 다중 클래스(37-way) 분류
    데이터셋이라, code2의 vocab/시퀀스 전처리나 molhiv의 이진 라벨 전제가
    필요 없음.
    """
    # 1. 원본 데이터셋 로드 및 스플릿 인덱스 확보
    dataset = PygGraphPropPredDataset(
        name=cfg['data']['dataset_name'],
        root=cfg['data']['dir'],
    )
    split_idx = dataset.get_idx_split()
    cached_dataset = CachedVNDataset(dataset)
    print(f'VN cache: {cached_dataset.cache_dir} (compute once on first access)')

    # 2. DataLoader 생성
    train_loader = DataLoader(
        Subset(cached_dataset, split_idx['train']),
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=cfg['train']['num_workers']
    )
    val_loader = DataLoader(
        Subset(cached_dataset, split_idx['valid']),
        batch_size=cfg['train']['batch_size'],
        shuffle=False,
        num_workers=cfg['train']['num_workers']
    )
    test_loader = DataLoader(
        Subset(cached_dataset, split_idx['test']),
        batch_size=cfg['train']['batch_size'],
        shuffle=False,
        num_workers=cfg['train']['num_workers']
    )

    # 3. 모델 입력에 필요한 메타데이터
    num_tasks = dataset.num_tasks                             # ppa는 1 (단일 다중클래스 태스크)
    num_classes = dataset.num_classes

    return train_loader, val_loader, test_loader, num_tasks, num_classes
