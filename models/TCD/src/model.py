"""
GCN + Centroid-Tree virtual nodes for ogbg-code2  (bug-fixed version)

Assumptions about the batched `data` object (in addition to the usual
x, edge_index, edge_attr, node_depth, batch):

    data.vn_edge_index : LongTensor [2, Evn]  edges of the centroid tree
                                              (indices in [0, num_vn))
    data.vn_batch      : LongTensor [num_vn]  graph id of every centroid-tree node
    data.node2vn       : LongTensor [N]       for every real node, the index of
                                              the centroid-tree node it is attached to

If you only have one virtual node per graph (classic OGB VN), pass
    data.node2vn      = data.batch
    data.vn_batch     = torch.arange(num_graphs)
    data.vn_edge_index = torch.empty(2, 0, dtype=torch.long)
and this module degenerates to the standard GCN-with-virtual-node.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import degree
from torch_geometric.utils import scatter

class GCNConv(MessagePassing):
    """OGB-style GCN layer for the real AST graph (uses edge features)."""

    def __init__(self, emb_dim):
        super().__init__(aggr='add')
        self.linear       = nn.Linear(emb_dim, emb_dim)
        self.root_emb     = nn.Embedding(1, emb_dim)
        self.edge_encoder = nn.Linear(7, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x        = self.linear(x)
        edge_emb = self.edge_encoder(edge_attr.float())

        row, col       = edge_index
        deg          = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0          # isolated node -> 0, not 1e9
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        agg = self.propagate(edge_index, x=x, edge_attr=edge_emb, norm=norm)
        root = F.relu(x + self.root_emb.weight) / deg.view(-1, 1)

        return agg + root

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)


class GCNConvVN(MessagePassing):
    """GCN layer for the centroid tree of virtual nodes (no edge features).

    edge_index is passed in `forward` (NOT stored on the module) so it can
    change every batch and follow `model.to(device)` for free.
    """

    def __init__(self, emb_dim):
        super().__init__(aggr='add')
        self.linear   = nn.Linear(emb_dim, emb_dim)
        self.root_emb = nn.Embedding(1, emb_dim)

    def forward(self, x, edge_index):
        x = self.linear(x)

        row, _       = edge_index
        deg          = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]

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

        self.node_encoder = node_encoder

        self.convs = nn.ModuleList([GCNConv(emb_dim) for _ in range(num_layers)])
        self.bns   = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(num_layers)])

        # --- virtual-node / centroid-tree machinery --------------------------
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
    def _propagate_vn(self, vn, vn_edge_index, block):
        """One block of `num_sublayers` GCNConvVN layers over the centroid tree."""
        h     = vn
        start = block * self.num_sublayers
        for k in range(self.num_sublayers):
            idx  = start + k
            h    = self.subbns[idx](self.subconvs[idx](h, vn_edge_index))
            last = (k == self.num_sublayers - 1)
            h    = F.dropout(h if last else F.relu(h),
                             p=self.drop_ratio, training=self.training)
        return h

    # ----------------------------------------------------------------------
    def forward(self, data):
        vn_edge_index = data.vn_edge_index
        inc_node, inc_vn       = data.vn_incidence
        vn_batch = scatter(data.batch[inc_node], inc_vn, dim=0,
                   dim_size=int(inc_vn.max()) + 1, reduce='max')
        num_vn        = vn_batch.size(0)
        N = data.x.size(0)

        h = self.node_encoder(data.x)               # [N, D]
        memb = scatter(torch.ones(inc_node.numel(), device=h.device),
                   inc_node, dim=0, dim_size=N, reduce='sum').clamp(min=1)
        
        vn = self.virtualnode_embedding(
            torch.zeros(num_vn, dtype=torch.long, device=data.x.device))      # [num_vn, D]


        for i in range(self.num_layers):
            h = h + scatter(vn[inc_vn], inc_node, dim=0, dim_size=N, reduce='mean')
                                             # inject hub state
            h = self.convs[i](h, data.edge_index, data.edge_attr)
            h = self.bns[i](h)
            h = F.dropout(F.relu(h) if i < self.num_layers - 1 else h,
                          self.drop_ratio, training=self.training)

            if i < self.num_layers - 1:
                # pool real nodes into their hub, refine, diffuse over centroid tree
                msg    = h[inc_node] / memb[inc_node].view(-1, 1)

                pooled = scatter(msg, inc_vn, dim=0, dim_size=num_vn, reduce='sum')
                update = self.mlp_virtualnode[i](pooled + vn)
                update = F.dropout(update, self.drop_ratio, training=self.training)
                vn     = vn + update
                vn     = self._propagate_vn(vn, vn_edge_index, block=i)

        h_graph = global_mean_pool(h, data.batch)
        return self.graph_pred_linear(h_graph)                # list of [B, num_tasks]
