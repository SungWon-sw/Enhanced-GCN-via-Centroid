import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import degree


class GCNConv(MessagePassing):
    """VN/centroid 트리 엣지 없이 원본 데이터셋의 edge_index만 사용하는 순수 GCN 레이어.
    edge_dim을 인자로 받아 ogbg-code2(augment_edge의 2차원)와 ogbg-ppa(STRING 채널 7차원)
    양쪽에 대응한다."""

    def __init__(self, emb_dim, edge_dim, use_edge_attr=True):
        super().__init__(aggr='add')
        self.linear       = nn.Linear(emb_dim, emb_dim)
        self.root_emb     = nn.Embedding(1, emb_dim)
        # use_edge_attr=false: 엣지 피처 무시 (메시지는 x_j만 사용)
        self.edge_encoder = nn.Linear(edge_dim, emb_dim) if use_edge_attr else None

    def forward(self, x, edge_index, edge_attr):
        x        = self.linear(x)
        edge_emb = self.edge_encoder(edge_attr.float()) if self.edge_encoder is not None else None
        row, col     = edge_index
        deg          = degree(row, x.size(0), dtype=torch.float32) + 1   # bf16 은 256 초과 정수 부정확
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt = deg_inv_sqrt.masked_fill(torch.isinf(deg_inv_sqrt), 0)   # isolated node -> 0, not 1e9
        norm = (deg_inv_sqrt[row] * deg_inv_sqrt[col]).to(x.dtype)
        agg  = self.propagate(edge_index, x=x, edge_attr=edge_emb, norm=norm)
        return agg + F.relu(x + self.root_emb.weight) / deg.to(x.dtype).view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        if edge_attr is not None:
            x_j = x_j + edge_attr
        return norm.view(-1, 1) * F.relu(x_j)


def make_norm(kind, dim):
    """Backbone normalization: 'batch' | 'layer' | 'none'."""
    if kind == 'batch':
        return nn.BatchNorm1d(dim)
    if kind == 'layer':
        return nn.LayerNorm(dim)
    if kind == 'none':
        return nn.Identity()
    raise ValueError(f"model.norm must be 'batch', 'layer' or 'none', got {kind!r}")


class GCNGraphClassifier(nn.Module):
    """VN/centroid 기작이 전혀 없는 baseline GCN. 그래프 하나당 단일 라벨(num_tasks 클래스)을
    예측한다. ogbg-ppa처럼 노드 입력 피처가 없는 데이터셋은 node_encoder=None으로 두면
    모든 노드에 동일한 학습 가능한 더미 임베딩을 부여해 처리한다.

    backbone_residual: main 브랜치 stash(WIP on main: 8d35b75)의 개선 중 VN과 무관한 부분만
    이식 — 각 GCN 레이어(conv+bn+dropout) 주변에 skip connection을 둔다. 같은 stash의
    vn_residual(centroid tree로 전파된 값을 되먹이는 것)은 VN 전용 기작이라 여긴 없음."""

    def __init__(self, cfg, num_tasks, edge_dim, node_encoder=None):
        super().__init__()

        train_cfg = cfg.get('train', {})
        model_cfg = cfg.get('model', {})

        emb_dim    = train_cfg.get('emb_dim', 300)
        num_layers = model_cfg.get('num_layers', 5)
        drop_ratio = train_cfg.get('drop_ratio', 0.5)

        self.node_encoder = node_encoder
        if self.node_encoder is None:
            self.dummy_node_emb = nn.Embedding(1, emb_dim)

        # GCN+ 옵션 (기본값 = 기존 baseline 동작과 동일)
        norm_kind     = model_cfg.get('norm', 'batch')            # batch | layer | none
        use_edge_attr = bool(model_cfg.get('edge_attr', True))
        self.use_ffn  = bool(model_cfg.get('ffn', False))
        ffn_ratio     = int(model_cfg.get('ffn_ratio', 2))
        pe_cfg        = model_cfg.get('pos_encoding', {})
        self.use_pe   = bool(pe_cfg.get('enabled', False))

        # RWSE (data.pe, [N, pe_dim]) -> emb_dim, 노드 임베딩에 더함
        if self.use_pe:
            self.pe_encoder = nn.Linear(int(pe_cfg.get('dim', 16)), emb_dim)

        self.convs = nn.ModuleList([GCNConv(emb_dim, edge_dim, use_edge_attr) for _ in range(num_layers)])
        self.bns   = nn.ModuleList([make_norm(norm_kind, emb_dim) for _ in range(num_layers)])
        self.drop_ratio = drop_ratio
        self.num_layer  = num_layers
        self.backbone_residual = bool(model_cfg.get('backbone_residual', True))

        # 레이어마다 자체 norm + skip을 가진 FFN: h = h + FFN(norm(h))
        if self.use_ffn:
            self.ffn_norms = nn.ModuleList([make_norm(norm_kind, emb_dim) for _ in range(num_layers)])
            self.ffns = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(emb_dim, ffn_ratio * emb_dim), nn.ReLU(),
                    nn.Dropout(drop_ratio),
                    nn.Linear(ffn_ratio * emb_dim, emb_dim),
                ) for _ in range(num_layers)
            ])

        # post-MP 헤드 (GCN+ ppa: layers_post_mp=2). 1 이면 기존 선형 헤드와 동일.
        post_mp = max(int(model_cfg.get('post_mp_layers', 1)), 1)
        head = []
        for _ in range(post_mp - 1):
            head += [nn.Linear(emb_dim, emb_dim), nn.ReLU()]
        head.append(nn.Linear(emb_dim, num_tasks))
        self.pred_head = nn.Sequential(*head)

    def forward(self, data):
        if self.node_encoder is not None:
            h = self.node_encoder(data.x, data.node_depth.view(-1))
        else:
            h = self.dummy_node_emb(data.x)
        if self.use_pe:
            h = h + self.pe_encoder(data.pe.to(h.dtype))

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h_in = h
            h = bn(conv(h, data.edge_index, data.edge_attr))
            h = F.dropout(F.relu(h) if i < self.num_layer - 1 else h,
                          p=self.drop_ratio, training=self.training)
            if self.backbone_residual:
                h = h + h_in
            if self.use_ffn:
                h = h + F.dropout(self.ffns[i](self.ffn_norms[i](h)),
                                  p=self.drop_ratio, training=self.training)

        h_graph = global_mean_pool(h, data.batch)
        return self.pred_head(h_graph)
