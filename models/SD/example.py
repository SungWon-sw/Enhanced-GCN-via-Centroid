"""CPU forward/backward example: python -m baseline_model.example."""

from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from . import build_model, prepare_graph


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    cfg = yaml.safe_load(Path(__file__).with_name('config.yaml').read_text())
    graphs = []
    # A path and a disconnected graph; neither requires dataset downloads.
    for label, (num_nodes, pairs) in enumerate([
        (5, [(0, 1), (1, 2), (2, 3), (3, 4)]),
        (4, [(0, 1), (2, 3)]),
    ]):
        edges = pairs + [(v, u) for u, v in pairs]
        raw = Data(
            num_nodes=num_nodes,
            edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
            edge_attr=torch.randn(len(edges), 7),
            y=torch.tensor([label]),
        )
        graphs.append(prepare_graph(raw, cfg))

    loader = DataLoader(graphs, batch_size=2)
    model = build_model(cfg, num_classes=37)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    model.train()
    batch = next(iter(loader))
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch)
    loss = F.cross_entropy(logits, batch.y.view(-1))
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite example loss')
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(batch).argmax(dim=1)
    print(f'logits: {tuple(logits.shape)}')
    print(f'loss: {loss.item():.6f}')
    print(f'predictions: {predictions.tolist()}')
    print('CPU forward/backward OK')


if __name__ == '__main__':
    main()
