"""
GCN + Centroid-Tree virtual nodes for ogbg-ppa.

Fields expected on the batched `data` object (produced by
utils_vn_connect.add_ppa_virtual_nodes, then cached / collated):

    data.x             : LongTensor [N]       zeros (ppa has no node features)
    data.edge_index    : LongTensor [2, E]    real protein-association graph
    data.edge_attr     : Tensor     [E, 7]    real edge features
    data.batch         : LongTensor [N]       graph id per real node
    data.vn_incidence  : LongTensor [2, M]    (real node, virtual node) memberships
    data.vn_edge_index : LongTensor [2, Evn]  centroid-tree edges over virtual nodes
    data.num_vn        : LongTensor scalar    #virtual nodes for the graph

Each virtual node is a bag of the (reduced) tree decomposition; the virtual
nodes are wired into the centroid tree of that bag tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import degree
from torch_geometric.utils import scatter

class GCNConv(MessagePassing):
    """OGB-style GCN layer for the real AST graph (uses edge features)."""

    def __init__(self, emb_dim, use_edge_attr=True):
        super().__init__(aggr='add')
        self.linear       = nn.Linear(emb_dim, emb_dim)
        self.root_emb     = nn.Embedding(1, emb_dim)
        # use_edge_attr=false: edge features are ignored (messages use x_j only)
        self.edge_encoder = nn.Linear(7, emb_dim) if use_edge_attr else None

    def forward(self, x, edge_index, edge_attr):
        x        = self.linear(x)
        edge_emb = self.edge_encoder(edge_attr.float()) if self.edge_encoder is not None else None

        row, col       = edge_index
        deg          = degree(row, x.size(0), dtype=torch.float32) + 1   # bf16 은 256 초과 정수 부정확
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt = deg_inv_sqrt.masked_fill(torch.isinf(deg_inv_sqrt), 0)   # isolated node -> 0, not 1e9
        norm = (deg_inv_sqrt[row] * deg_inv_sqrt[col]).to(x.dtype)

        agg = self.propagate(edge_index, x=x, edge_attr=edge_emb, norm=norm)
        root = F.relu(x + self.root_emb.weight) / deg.to(x.dtype).view(-1, 1)

        return agg + root

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
        deg          = degree(row, x.size(0), dtype=torch.float32)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt = deg_inv_sqrt.masked_fill(torch.isinf(deg_inv_sqrt), 0)
        norm = (deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]).to(x.dtype)

        agg = self.propagate(edge_index, x=x, norm=norm)
        return agg + F.relu(x + self.root_emb.weight) / deg.clamp(min=1).to(x.dtype).view(-1, 1)

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
        # residual around each centroid-tree block; alpha scales the propagated
        # branch. vn_residual=false recovers the plain "replace vn" behaviour.
        self.vn_residual       = bool(model_cfg.get('vn_residual', True))
        self.vn_residual_alpha = float(model_cfg.get('vn_residual_alpha', 1.0))
        # skip connection around each backbone GCN layer (inject+conv+bn+dropout),
        # wired to the previous layer's output as in the OGB reference GNN.
        # backbone_residual=false recovers the plain feed-forward stack.
        self.backbone_residual = bool(model_cfg.get('backbone_residual', True))

        # GCN+ style options (all switchable from config `model:`)
        norm_kind     = model_cfg.get('norm', 'batch')            # batch | layer | none
        use_edge_attr = bool(model_cfg.get('edge_attr', True))
        self.use_ffn  = bool(model_cfg.get('ffn', False))
        ffn_ratio     = int(model_cfg.get('ffn_ratio', 2))
        pe_cfg        = model_cfg.get('pos_encoding', {})
        self.use_pe   = bool(pe_cfg.get('enabled', False))

        self.node_encoder = node_encoder

        # RWSE (data.pe, [N, pe_dim]) -> emb_dim, added to the node embedding
        if self.use_pe:
            self.pe_encoder = nn.Linear(int(pe_cfg.get('dim', 16)), emb_dim)

        self.convs = nn.ModuleList([GCNConv(emb_dim, use_edge_attr) for _ in range(num_layers)])
        self.bns   = nn.ModuleList([make_norm(norm_kind, emb_dim) for _ in range(num_layers)])

        # per-layer FFN with its own norm and skip connection: h = h + FFN(norm(h))
        if self.use_ffn:
            self.ffn_norms = nn.ModuleList([make_norm(norm_kind, emb_dim) for _ in range(num_layers)])
            self.ffns = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(emb_dim, ffn_ratio * emb_dim), nn.ReLU(),
                    nn.Dropout(drop_ratio),
                    nn.Linear(ffn_ratio * emb_dim, emb_dim),
                ) for _ in range(num_layers)
            ])

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

        # post-MP 헤드 (GCN+ ppa: layers_post_mp=2). 1 이면 기존 선형 헤드와 동일.
        post_mp = max(int(model_cfg.get('post_mp_layers', 1)), 1)
        head = []
        for _ in range(post_mp - 1):
            head += [nn.Linear(emb_dim, emb_dim), nn.ReLU()]
        head.append(nn.Linear(emb_dim, num_classes))
        self.graph_pred_linear = nn.Sequential(*head)

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
        # train.to_device 가 미리 계산해 둔 python int 사용 (없으면 .max() 로 폴백 = GPU 동기화)
        num_vn = getattr(data, 'num_vn_total', None)
        if num_vn is None:
            num_vn = int(inc_vn.max()) + 1
        N = data.x.size(0)

        h = self.node_encoder(data.x)               # [N, D]
        if self.use_pe:
            h = h + self.pe_encoder(data.pe.to(h.dtype))
        memb = scatter(torch.ones(inc_node.numel(), device=h.device),
                   inc_node, dim=0, dim_size=N, reduce='sum').clamp(min=1)
        
        vn = self.virtualnode_embedding(
            torch.zeros(num_vn, dtype=torch.long, device=data.x.device))      # [num_vn, D]


        for i in range(self.num_layers):
            h_in = h                         # previous layer output (OGB-ref skip target)
            h = h + scatter(vn[inc_vn], inc_node, dim=0, dim_size=N, reduce='mean')
                                             # inject hub state
            h = self.convs[i](h, data.edge_index, data.edge_attr)
            h = self.bns[i](h)
            h = F.dropout(F.relu(h) if i < self.num_layers - 1 else h,
                          self.drop_ratio, training=self.training)
            if self.backbone_residual:
                h = h + h_in                 # skip around inject+conv+bn+dropout
            if self.use_ffn:
                h = h + F.dropout(self.ffns[i](self.ffn_norms[i](h)),
                                  self.drop_ratio, training=self.training)

            if i < self.num_layers - 1:
                # pool real nodes into their hub, refine, diffuse over centroid tree
                msg    = h[inc_node] / memb[inc_node].view(-1, 1)

                pooled = scatter(msg, inc_vn, dim=0, dim_size=num_vn, reduce='sum')
                update = self.mlp_virtualnode[i](pooled + vn)
                update = F.dropout(update, self.drop_ratio, training=self.training)
                vn     = vn + update

                propagated = self._propagate_vn(vn, vn_edge_index, block=i)
                if self.vn_residual:
                    vn = vn + self.vn_residual_alpha * propagated
                else:
                    vn = propagated

        h_graph = global_mean_pool(h, data.batch)
        return self.graph_pred_linear(h_graph)                # [B, num_classes]
