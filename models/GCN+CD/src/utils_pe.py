"""Random-walk structural positional encoding (RWSE) for ogbg-ppa.

pe[i, k-1] = (D^-1 A)^k [i, i]  -- return probability of a k-step random walk.
Unlike Laplacian eigenvectors there is no sign/basis ambiguity, so it can be
computed once per graph and fed to the model as-is.
"""

import torch
from torch.utils.data import Dataset


def compute_rwse(edge_index, num_nodes, k):
    """Return [num_nodes, k] return probabilities for walk lengths 1..k."""
    A = torch.zeros(num_nodes, num_nodes)
    A[edge_index[0], edge_index[1]] = 1.0
    deg = A.sum(dim=1).clamp(min=1)
    P = A / deg.view(-1, 1)

    out = torch.empty(num_nodes, k)
    Pk = P
    for i in range(k):
        out[:, i] = Pk.diagonal()
        if i + 1 < k:
            Pk = Pk @ P
    return out


class PEDataset(Dataset):
    """Adds `data.pe` ([N, pe_dim]) to every graph of the wrapped dataset."""

    def __init__(self, dataset, pe_dim):
        self.dataset = dataset
        self.pe_dim = pe_dim

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        data.pe = compute_rwse(data.edge_index, data.num_nodes, self.pe_dim)
        return data
