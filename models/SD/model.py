"""Standalone ogbg-ppa baseline: GCN with separator-tree virtual nodes.

Call prepare_graph on each raw graph, then collate with a PyG DataLoader.
The batched input contains x [N] (all-zero node IDs), edge_index [2, E],
edge_attr [E, 7], batch [N], vn_incidence [2, M], vn_edge_index [2, Evn],
num_vn [B], and vn_kind [K] (including empty connector roots).

Extracted from src/model.py and src/utils.py. Parameter names and forward
computation are preserved for baseline state_dict compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import degree
from torch_geometric.utils import scatter


class PPANodeEncoder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(1, emb_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x):
        return self.embedding(x.view(-1).long())


def graph_normalization(edge_index, num_nodes, dtype, add_self, cache=None):
    """Cache only within one forward, preserving the convolution's AMP dtype."""
    if cache is not None and dtype in cache:
        return cache[dtype]
    row, col = edge_index
    deg = degree(row, num_nodes, dtype=dtype)
    if add_self:
        deg = deg + 1
    inv = deg.pow(-0.5)
    inv = inv.masked_fill(torch.isinf(inv), 0)
    result = (deg, inv[row] * inv[col])
    if cache is not None:
        cache[dtype] = result
    return result


class GCNConv(MessagePassing):
    """OGB-style GCN layer for the real protein-association graph (uses edge features)."""

    def __init__(self, emb_dim):
        super().__init__(aggr='add')
        self.linear       = nn.Linear(emb_dim, emb_dim)
        self.root_emb     = nn.Embedding(1, emb_dim)
        self.edge_encoder = nn.Linear(7, emb_dim)

    def forward(self, x, edge_index, edge_attr, normalization=None):
        x        = self.linear(x)
        edge_emb = self.edge_encoder(edge_attr.float())

        deg, norm = graph_normalization(edge_index, x.size(0), x.dtype,
                                       add_self=True, cache=normalization)

        agg = self.propagate(edge_index, x=x, edge_attr=edge_emb, norm=norm)
        root = F.relu(x + self.root_emb.weight) / deg.view(-1, 1)

        return agg + root

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)


class GCNConvVN(MessagePassing):
    """GCN layer for the separator tree of virtual nodes (no edge features).

    edge_index is passed in `forward` (NOT stored on the module) so it can
    change every batch and follow `model.to(device)` for free.
    """

    def __init__(self, emb_dim):
        super().__init__(aggr='add')
        self.linear   = nn.Linear(emb_dim, emb_dim)
        self.root_emb = nn.Embedding(1, emb_dim)

    def forward(self, x, edge_index, normalization=None):
        x = self.linear(x)

        deg, norm = graph_normalization(edge_index, x.size(0), x.dtype,
                                       add_self=False, cache=normalization)

        agg = self.propagate(edge_index, x=x, norm=norm)
        return agg + F.relu(x + self.root_emb.weight) / deg.clamp(min=1).view(-1, 1)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * F.relu(x_j)


class GCN(nn.Module):
    def __init__(self, cfg, node_encoder, num_classes, num_tasks=None):
        super().__init__()

        train_cfg = cfg.get('train', {})
        model_cfg = cfg.get('model', {})

        emb_dim       = train_cfg.get('emb_dim', 300)
        num_layers    = model_cfg.get('num_layers', 3)
        num_sublayers = model_cfg.get('num_sublayers', 3)
        drop_ratio    = train_cfg.get('drop_ratio', 0.5)
        max_seq_len   = train_cfg.get('max_seq_len', 5)
        if num_tasks is None:
            num_tasks = train_cfg.get('num_vocab', 5000)

        self.num_layers    = num_layers
        self.num_sublayers = num_sublayers
        self.drop_ratio    = drop_ratio
        self.max_seq_len   = max_seq_len
        # residual around each separator-tree block; alpha scales the propagated
        # branch. vn_residual=false recovers the plain "replace vn" behaviour.
        self.vn_residual       = bool(model_cfg.get('vn_residual', True))
        self.vn_residual_alpha = float(model_cfg.get('vn_residual_alpha', 1.0))
        # skip connection around each backbone GCN layer (inject+conv+bn+dropout),
        # wired to the previous layer's output as in the OGB reference GNN.
        # backbone_residual=false recovers the plain feed-forward stack.
        self.backbone_residual = bool(model_cfg.get('backbone_residual', True))

        self.node_encoder = node_encoder

        self.convs = nn.ModuleList([GCNConv(emb_dim) for _ in range(num_layers)])
        self.bns   = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(num_layers)])

        # --- virtual-node / separator-tree machinery --------------------------
        self.virtualnode_embedding = nn.Embedding(1, emb_dim)
        nn.init.zeros_(self.virtualnode_embedding.weight)          # actually 0-init now

        # the VN state is only updated *between* GCN layers -> num_layers - 1 blocks
        n_vn_blocks = max(num_layers - 1, 0)

        self.mlp_virtualnode = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim, 2 * emb_dim), nn.BatchNorm1d(2 * emb_dim), nn.ReLU(),
                nn.Linear(2 * emb_dim, emb_dim), nn.BatchNorm1d(emb_dim),     nn.ReLU(),
            ) for _ in range(n_vn_blocks)
        ])
        self.subconvs = nn.ModuleList(
            [GCNConvVN(emb_dim) for _ in range(n_vn_blocks * num_sublayers)])
        self.subbns = nn.ModuleList(
            [nn.BatchNorm1d(emb_dim) for _ in range(n_vn_blocks * num_sublayers)])

        self.graph_pred_linear = nn.Linear(emb_dim, num_classes)

    # ----------------------------------------------------------------------
    def _propagate_vn(self, vn, vn_edge_index, block, normalization=None):
        """One block of `num_sublayers` GCNConvVN layers over the separator tree."""
        h     = vn
        start = block * self.num_sublayers
        for k in range(self.num_sublayers):
            idx  = start + k
            h    = self.subbns[idx](self.subconvs[idx](h, vn_edge_index, normalization))
            last = (k == self.num_sublayers - 1)
            h    = F.dropout(h if last else F.relu(h),
                             p=self.drop_ratio, training=self.training)
        return h

    def _update_nodes(self, h, h_in, data, layer, normalization=None):
        """Real-node block; variants can extend it before the VN update."""
        h = self.convs[layer](h, data.edge_index, data.edge_attr, normalization)
        h = self.bns[layer](h)
        h = F.dropout(F.relu(h) if layer < self.num_layers - 1 else h,
                      self.drop_ratio, training=self.training)
        if self.backbone_residual:
            h = h + h_in
        return h

    # ----------------------------------------------------------------------
    def forward(self, data):
        vn_edge_index = data.vn_edge_index
        inc_node, inc_vn = data.vn_incidence

        # 연결용 빈 루트까지 포함한 배치 전체 VN 개수
        num_vn = data.vn_kind.size(0)
        N = data.x.size(0)
        real_normalization, vn_normalization = {}, {}

        h = self.node_encoder(data.x)               # [N, D]
        memb = scatter(torch.ones(inc_node.numel(), device=h.device),
                   inc_node, dim=0, dim_size=N, reduce='sum').clamp(min=1)
        
        vn = self.virtualnode_embedding(
            torch.zeros(num_vn, dtype=torch.long, device=data.x.device))      # [num_vn, D]


        for i in range(self.num_layers):
            h_in = h                         # previous layer output (OGB-ref skip target)
            h = h + scatter(vn[inc_vn], inc_node, dim=0, dim_size=N, reduce='mean')
                                             # inject hub state
            h = self._update_nodes(h, h_in, data, i, real_normalization)

            if i < self.num_layers - 1:
                # pool real nodes into their hub, refine, diffuse over separator tree
                msg    = h[inc_node] / memb[inc_node].view(-1, 1)

                pooled = scatter(msg, inc_vn, dim=0, dim_size=num_vn, reduce='sum')
                update = self.mlp_virtualnode[i](pooled + vn)
                update = F.dropout(update, self.drop_ratio, training=self.training)
                vn     = vn + update

                propagated = self._propagate_vn(vn, vn_edge_index, block=i,
                                                normalization=vn_normalization)
                if self.vn_residual:
                    vn = vn + self.vn_residual_alpha * propagated
                else:
                    vn = propagated

        h_graph = global_mean_pool(h, data.batch, size=data.num_graphs)
        return self.graph_pred_linear(h_graph)                # [B, num_classes]


def build_model(cfg, num_classes=37):
    """Build the baseline with its PPA node encoder and original weight keys."""
    name = cfg.get('model', {}).get('name', 'baseline')
    if name != 'baseline':
        raise ValueError(f'This package only supports baseline, got {name!r}')
    emb_dim = cfg.get('train', {}).get('emb_dim', 300)
    return GCN(cfg, PPANodeEncoder(emb_dim), num_classes=num_classes)
