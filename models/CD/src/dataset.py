import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset

# 유틸리티 함수 임포트
from utils import get_vocab_mapping, encode_y_to_arr, augment_edge, convert_into_easy

def build_loaders(cfg):
    """
    config 설정을 받아 train, valid, test 데이터 로더와 
    모델 빌드에 필요한 메타데이터(idx2vocab, 노드 타입 수 등)를 반환합니다.
    """
    # 1. 원본 데이터셋 로드 및 스플릿 인덱스 확보
    dataset = PygGraphPropPredDataset(name=cfg['data']['dataset_name'])
    split_idx = dataset.get_idx_split()

    # 2. 훈련 데이터를 바탕으로 Vocab 매핑 생성
    vocab2idx, idx2vocab = get_vocab_mapping(
        [dataset[i].y for i in split_idx['train']], 
        cfg['train']['num_vocab']
    )

    # 3. GNN 전처리 transform 함수 적용
    def transform_fn(data):
        return encode_y_to_arr(augment_edge(data), vocab2idx, cfg['train']['max_seq_len'])
    
    dataset.transform = transform_fn
    dataset, depth_data = convert_into_easy(dataset)

    # 4. DataLoader 생성
    train_loader = DataLoader(
        Subset(dataset, split_idx['train']), 
        batch_size=cfg['train']['batch_size'], 
        shuffle=True, 
        num_workers=cfg['train']['num_workers']
    )
    val_loader = DataLoader(
        Subset(dataset, split_idx['valid']), 
        batch_size=cfg['train']['batch_size'], 
        shuffle=False, 
        num_workers=cfg['train']['num_workers']
    )
    test_loader = DataLoader(
        Subset(dataset, split_idx['test']),  
        batch_size=cfg['train']['batch_size'], 
        shuffle=False, 
        num_workers=cfg['train']['num_workers']
    )

    # 5. 모델 입력에 필요한 메타데이터 계산
    num_nodetypes = int(dataset.data.x[:, 0].max().item()) + 1
    num_nodeattributes = int(dataset.data.x[:, 1].max().item()) + 1

    return train_loader, val_loader, test_loader, idx2vocab, num_nodetypes, num_nodeattributes, depth_data