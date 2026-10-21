import builtins
import contextlib

import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset

from utils_pe import PEDataset


def _add_dummy_node_feature(data):
    # ogbg-ppa는 노드 입력 피처가 없다(x=None). 모든 노드를 동일한 더미 인덱스(0)로
    # 채워서 모델의 dummy_node_emb(nn.Embedding(1, emb_dim))가 처리하게 한다.
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data


@contextlib.contextmanager
def _auto_confirm_dataset_prompts():
    """ogb는 (1) 로컬 캐시 버전이 안 맞으면 업데이트할지, (2) 실제 다운로드(ppa는 2.79GB)를
    진행할지를 input()으로 물어본다. nohup 등 stdin이 없는 환경에서는 input() 자체가
    OSError(Bad file descriptor)로 죽고, 'N'으로 자동응답하면 ogb가 download()에서
    decide_download()==False로 판단해 'Stop downloading.' 후 exit(-1)로 프로세스를
    통째로 종료시켜버린다(캐시가 없으면 다운로드가 필수이므로). 그래서 비대화형
    환경에서는 두 프롬프트 모두 'y'로 자동 응답해 정상적으로 받아지게 한다."""
    original_input = builtins.input

    def _auto_yes(prompt=''):
        print(prompt, end='')
        print('y  (자동 응답 — 비대화형 환경이라 필요한 다운로드/갱신을 그대로 진행)')
        return 'y'

    builtins.input = _auto_yes
    try:
        yield
    finally:
        builtins.input = original_input


def build_loaders(cfg):
    """VN/centroid 전처리를 전혀 거치지 않고 원본 edge_index/edge_attr을 그대로 사용한다."""
    with _auto_confirm_dataset_prompts():
        dataset = PygGraphPropPredDataset(
            name=cfg['data']['dataset_name'],
            transform=_add_dummy_node_feature,
        )
    split_idx = dataset.get_idx_split()

    num_tasks = int(dataset.num_classes)   # PEDataset 래핑 전에 읽어둠

    pe_cfg = cfg.get('model', {}).get('pos_encoding', {})
    if pe_cfg.get('enabled', False):
        dataset = PEDataset(dataset, int(pe_cfg.get('dim', 16)))

    train_loader = DataLoader(
        Subset(dataset, split_idx['train']),
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=cfg['train']['num_workers'],
    )
    val_loader = DataLoader(
        Subset(dataset, split_idx['valid']),
        batch_size=cfg['train']['batch_size'],
        shuffle=False,
        num_workers=cfg['train']['num_workers'],
    )
    test_loader = DataLoader(
        Subset(dataset, split_idx['test']),
        batch_size=cfg['train']['batch_size'],
        shuffle=False,
        num_workers=cfg['train']['num_workers'],
    )

    edge_dim = dataset[0].edge_attr.size(-1)

    return train_loader, val_loader, test_loader, num_tasks, edge_dim
