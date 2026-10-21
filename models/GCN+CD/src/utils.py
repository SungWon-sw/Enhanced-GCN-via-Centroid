import torch.nn as nn

class PPANodeEncoder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(1, emb_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x):
        return self.embedding(x.view(-1).long())