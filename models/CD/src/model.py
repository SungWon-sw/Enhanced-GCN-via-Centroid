import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import degree

class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super().__init__(aggr='add')
        self.linear       = nn.Linear(emb_dim, emb_dim)
        self.root_emb     = nn.Embedding(1, emb_dim)
        self.edge_encoder = nn.Linear(2, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x        = self.linear(x)
        edge_emb = self.edge_encoder(edge_attr.float())
        row, _   = edge_index
        deg          = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5).clamp(max=1e9)
        norm = deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]
        agg  = self.propagate(edge_index, x=x, edge_attr=edge_emb, norm=norm)
        return agg + F.relu(x + self.root_emb.weight) / deg.view(-1,1).clamp(min=1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1,1) * F.relu(x_j + edge_attr)
    

class GCN(nn.Module):
    def __init__(self, cfg, node_encoder, num_tasks=None):
        """
        Args:
            cfg (dict): config.yaml이 로드된 딕셔너리 객체
            node_encoder (nn.Module): 외부에서 주입받는 ASTNodeEncoder 객체
            num_tasks (int, optional): 분류할 클래스(단어장) 개수. 
                                       지정하지 않으면 cfg['train']['num_vocab'] 사용.
        """
        super().__init__()
        
        # 1. 딕셔너리 안전하게 unpacking 후 기본값(Fallback) 지정
        train_cfg = cfg.get('train', {})
        model_cfg = cfg.get('model', {})
        
        # 이전 config에서 모델/학습에 흩어져 있던 하이퍼파라미터 연동
        emb_dim     = train_cfg.get('emb_dim', 300)
        num_layers  = model_cfg.get('num_layers', 3)       # config의 num_layers 사용
        drop_ratio  = train_cfg.get('drop_ratio', 0.5)
        max_seq_len = train_cfg.get('max_seq_len', 5)
        
        # num_tasks가 주어지지 않으면 cfg의 num_vocab을 기본 타겟 크기로 설정
        if num_tasks is None:
            num_tasks = train_cfg.get('num_vocab', 5000)

        # 2. 레이어 정의
        self.convs        = nn.ModuleList([GCNConv(emb_dim) for _ in range(num_layers)])
        self.bns          = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(num_layers)])
        self.node_encoder = node_encoder
        self.drop_ratio   = drop_ratio
        self.num_layer    = num_layers
        self.max_seq_len  = max_seq_len

        # 각 시퀀스 위치별 분류를 위한 Prediction Heads (num_tasks 크기로 생성)
        self.pred_heads   = nn.ModuleList([
            nn.Linear(emb_dim, num_tasks) for _ in range(max_seq_len)
        ])

    def forward(self, data):
        # ⚠️ 부모 클래스(GCNConv)에서 x의 device를 따르므로, node_depth도 동일하게 처리되도록 .view(-1) 유지
        h = self.node_encoder(data.x, data.node_depth.view(-1))
        
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h = bn(conv(h, data.edge_index, data.edge_attr))
            h = F.dropout(F.relu(h) if i < self.num_layer - 1 else h,
                          p=self.drop_ratio, training=self.training)
            
        h_graph = global_mean_pool(h, data.batch)
        return [head(h_graph) for head in self.pred_heads]