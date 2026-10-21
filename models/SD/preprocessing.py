"""Separator-tree construction and correct PyG batching for the baseline.

Extracted from src/utils_vn_connect.py without its disk-cache/dataset wrapper.
The separator algorithms and VNData batching rules are unchanged.
"""

from collections import Counter
from itertools import chain, combinations

import networkx as nx
from networkx.algorithms.connectivity import build_auxiliary_node_connectivity
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow
import torch
from torch_geometric.data import Data


class VNData(Data):
    """실노드 번호와 VN 번호를 별도로 처리하는 PyG Data."""

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in {"vn_incidence", "vn_edge_index"}:
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key == "vn_incidence":
            return value.new_tensor([
                [self.num_nodes],
                [int(self.num_vn.item())],
            ])
        if key == "vn_edge_index":
            return int(self.num_vn.item())
        return super().__inc__(key, value, *args, **kwargs)


def _minimum_separator(graph):
    """Exact minimum node cut for a connected, non-complete undirected graph.

    Use the same node-splitting reduction and candidate pairs as NetworkX's
    global minimum_node_cut. Run the flows in SciPy's compiled Dinic solver;
    stop once the articulation-free lower bound of two is reached.
    """
    vertex = min(graph, key=lambda v: (graph.degree(v), v))
    best = set(graph[vertex])
    if len(best) == 1:
        return best
    articulation = next(nx.articulation_points(graph), None)
    if articulation is not None:
        return {articulation}
    # No articulation point: every separator has at least two vertices.
    if len(best) == 2:
        return best

    auxiliary = build_auxiliary_node_connectivity(graph)
    mapping = auxiliary.graph["mapping"]
    aux_nodes = list(auxiliary)
    aux_index = {node: index for index, node in enumerate(aux_nodes)}
    rows, cols = zip(*((aux_index[u], aux_index[v]) for u, v in auxiliary.edges()))
    capacity = csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)),
                          shape=(len(aux_nodes), len(aux_nodes)))
    neighbors = set(graph[vertex])
    pairs = chain(
        ((vertex, other) for other in sorted(set(graph) - neighbors - {vertex})),
        ((u, v) for u, v in combinations(sorted(neighbors), 2)
         if not graph.has_edge(u, v)),
    )
    for source, target in pairs:
        source_index = aux_index[f"{mapping[source]}B"]
        target_index = aux_index[f"{mapping[target]}A"]
        result = maximum_flow(capacity, source_index, target_index, method='dinic')
        if result.flow_value >= len(best):
            continue
        residual = capacity - result.flow
        residual.eliminate_zeros()
        reachable = {aux_nodes[i] for i in breadth_first_order(
            residual, source_index, directed=True, return_predecessors=False)}
        cut_edges = ((u, v) for u in reachable for v in auxiliary[u]
                     if v not in reachable)
        best = {auxiliary.nodes[v]["id"] for edge in cut_edges for v in edge}
        best -= {source, target}
        if len(best) == 2:
            break
    return best


def _balanced_articulation(graph):
    """Best single-vertex cut of a connected graph, or None.

    Score cuts by (largest remaining component, sum of squared component
    sizes). A weighted block-cut tree scores every articulation in O(V + E),
    without removing each candidate and traversing the graph again.
    """
    blocks = list(nx.biconnected_components(graph))
    if len(blocks) <= 1:
        return None

    counts = Counter(node for block in blocks for node in block)
    articulations = [node for node, count in counts.items() if count > 1]
    indices = {node: len(blocks) + i for i, node in enumerate(articulations)}
    adjacent = [[] for _ in range(len(blocks) + len(articulations))]
    weights = [0] * len(blocks) + [1] * len(articulations)
    for i, block in enumerate(blocks):
        for node in block:
            if node in indices:
                j = indices[node]
                adjacent[i].append(j)
                adjacent[j].append(i)
            else:
                weights[i] += 1

    parent = [-1] * len(adjacent)
    order = [0]
    for node in order:
        for neighbor in adjacent[node]:
            if neighbor != parent[node]:
                parent[neighbor] = node
                order.append(neighbor)
    subtree = weights.copy()
    for node in reversed(order[1:]):
        subtree[parent[node]] += subtree[node]

    def score(vertex):
        node = indices[vertex]
        sizes = [len(graph) - subtree[node]]
        sizes.extend(subtree[child] for child in adjacent[node]
                     if parent[child] == node)
        return max(sizes), sum(size * size for size in sizes)

    vertex = min(articulations, key=lambda v: (score(v), v))
    return vertex, score(vertex)


def _balanced_minimum_separator(graph):
    """Keep exact minimum cardinality and improve balance by vertex swaps.

    Single-vertex cuts are balanced optimally. For larger cuts, fix k-1 cut
    vertices and replace the remaining one with the best articulation of the
    remaining connected graph. Each accepted swap strictly improves balance.
    Two sweeps bound the extra work; global optimal balance is not guaranteed.
    """
    articulation = _balanced_articulation(graph)
    if articulation is not None:
        return {articulation[0]}

    cut = _minimum_separator(graph)
    nodes = set(graph)
    sizes = [len(component) for component in
             nx.connected_components(graph.subgraph(nodes - cut))]
    best_score = max(sizes), sum(size * size for size in sizes)
    for _ in range(2):
        improved = False
        for vertex in sorted(cut):
            fixed = cut - {vertex}
            # Removing k-1 vertices cannot disconnect a graph whose minimum
            # separator has size k. Its articulation cuts therefore retain k.
            candidate = _balanced_articulation(graph.subgraph(nodes - fixed).copy())
            if candidate is not None and candidate[1] < best_score:
                cut = fixed | {candidate[0]}
                best_score = candidate[1]
                improved = True
        if not improved:
            break
    return cut


def _merge_separator_tree(fields, merge_limit=16, balance_ratio=0.75):
    """Contract adjacent separator VNs to shorten an unbalanced tree.

    Absorb the largest child separator while its subtree exceeds the target
    fraction and the combined separator fits the merge limit. Leaves and
    empty connector roots stay intact. Existing separators larger than the
    limit are retained, but cannot absorb another separator. Only tree edges
    are contracted, so the tree diameter cannot increase.
    """
    count = fields['vn_kind'].numel()
    kinds = fields['vn_kind'].tolist()
    own = torch.bincount(fields['vn_incidence'][1], minlength=count).tolist()
    adjacent = [[] for _ in range(count)]
    for source, target in fields['vn_edge_index'].t().tolist():
        adjacent[source].append(target)
    children = [[] for _ in range(count)]
    parent = [-1] * count
    order = [0]
    for node in order:
        for neighbor in adjacent[node]:
            if neighbor != parent[node]:
                parent[neighbor] = node
                children[node].append(neighbor)
                order.append(neighbor)
    totals = own.copy()
    for node in reversed(order[1:]):
        totals[parent[node]] += totals[node]

    remap = [-1] * count
    new_kinds, edges = [], []
    pending = [(0, None)]
    while pending:
        old, new_parent = pending.pop()
        new = len(new_kinds)
        new_kinds.append(kinds[old])
        remap[old] = new
        if new_parent is not None:
            edges.extend([(new_parent, new), (new, new_parent)])
        frontier = list(children[old])
        size = own[old]
        if kinds[old] == 0:
            while frontier:
                largest = max(frontier, key=lambda v: (totals[v], -v))
                if totals[largest] <= balance_ratio * totals[old]:
                    break
                if kinds[largest] != 0 or size + own[largest] > merge_limit:
                    break
                frontier.remove(largest)
                frontier.extend(children[largest])
                size += own[largest]
                remap[largest] = new
        pending.extend((child, new) for child in sorted(frontier, reverse=True))

    return {
        'vn_incidence': torch.stack([
            fields['vn_incidence'][0],
            torch.tensor(remap, dtype=torch.long)[fields['vn_incidence'][1]],
        ]),
        'vn_edge_index': torch.tensor(edges, dtype=torch.long).reshape(-1, 2).t().contiguous(),
        'num_vn': torch.tensor([len(new_kinds)], dtype=torch.long),
        'vn_kind': torch.tensor(new_kinds, dtype=torch.long),
    }


SEPARATOR_STRATEGIES = ('minimum', 'balanced_minimum', 'balanced_merged')


def _validate_separator_options(strategy, merge_limit, balance_ratio):
    if strategy not in SEPARATOR_STRATEGIES:
        raise ValueError(f"separator_strategy must be one of {SEPARATOR_STRATEGIES}")
    if strategy == 'balanced_merged':
        if isinstance(merge_limit, bool) or not isinstance(merge_limit, int) or merge_limit < 1:
            raise ValueError('separator_merge_limit must be a positive integer')
        if (isinstance(balance_ratio, bool) or not isinstance(balance_ratio, (int, float))
                or not 0.5 <= balance_ratio < 1):
            raise ValueError('separator_balance_ratio must be in [0.5, 1)')


def separator_tree_fields(data, leaf_size=1, separator_strategy='minimum',
                          separator_merge_limit=16, separator_balance_ratio=0.75):
    """
    vn_incidence:  [2, N], (실노드 번호, 소속 VN 번호)
    vn_edge_index: [2, 2*(K-1)], VN 트리의 양방향 간선
    num_vn:        [1], VN 개수 K
    vn_kind:       [K], 0=separator, 1=leaf, 2=연결용 루트

    leaf_size 이하의 성분 또는 완전 그래프에서는 분할을 멈춥니다.
    """
    if leaf_size < 1:
        raise ValueError("leaf_size must be >= 1")
    _validate_separator_options(separator_strategy, separator_merge_limit, separator_balance_ratio)
    find_separator = (_minimum_separator if separator_strategy == 'minimum'
                      else _balanced_minimum_separator)

    n = int(data.num_nodes)
    if n == 0:
        raise ValueError("Empty graphs are not supported")

    # separator 계산에서는 무방향 구조만 사용합니다.
    # 원본 data.edge_index와 edge_attr는 변경하지 않습니다.
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(
        (u, v)
        for u, v in data.edge_index.cpu().t().tolist()
        if u != v
    )

    owner = torch.full((n,), -1, dtype=torch.long)
    kinds = []
    edges = []

    # 재귀 호출 대신 스택을 사용합니다.
    pending = [(set(graph), None)]

    def new_vn(nodes, kind, parent):
        vid = len(kinds)
        kinds.append(kind)

        if nodes:
            owner[sorted(nodes)] = vid

        if parent is not None:
            edges.extend([(parent, vid), (vid, parent)])

        return vid

    while pending:
        nodes, parent = pending.pop()
        # Materialize once: flow algorithms repeatedly traversing a filtered
        # subgraph view pay for membership checks on every adjacency access.
        subgraph = graph.subgraph(nodes).copy()
        components = sorted(
            nx.connected_components(subgraph),
            key=min,
        )

        # 처음부터 비연결이면 연결용 루트 아래에 각 성분을 배치합니다.
        if len(components) > 1:
            if parent is None:
                parent = new_vn(set(), kind=2, parent=None)

            pending.extend(
                (component, parent)
                for component in reversed(components)
            )
            continue

        size = len(nodes)
        is_complete = (
            subgraph.number_of_edges() == size * (size - 1) // 2
        )

        # 완전 그래프는 두 비어 있지 않은 성분으로 분리할 수 없습니다.
        if size <= leaf_size or is_complete:
            new_vn(nodes, kind=1, parent=parent)
            continue

        # 현재 성분에서 정점 수가 최소인 separator 하나를 선택합니다.
        separator = find_separator(subgraph)

        remaining = subgraph.subgraph(nodes - separator)
        components = sorted(
            nx.connected_components(remaining),
            key=min,
        )

        if not separator or len(components) < 2:
            raise RuntimeError("Invalid separator")

        # separator 정점들만 이 VN에 소속시킵니다.
        vid = new_vn(separator, kind=0, parent=parent)

        # 남은 성분에서 생성되는 VN들은 이 VN의 자식이 됩니다.
        pending.extend(
            (component, vid)
            for component in reversed(components)
        )

    assert bool((owner >= 0).all())

    fields = {
        "vn_incidence": torch.stack([torch.arange(n), owner]),
        "vn_edge_index": (
            torch.tensor(edges, dtype=torch.long)
            .reshape(-1, 2)
            .t()
            .contiguous()
        ),
        "num_vn": torch.tensor([len(kinds)], dtype=torch.long),
        "vn_kind": torch.tensor(kinds, dtype=torch.long),
    }
    if separator_strategy == 'balanced_merged':
        return _merge_separator_tree(fields, separator_merge_limit, separator_balance_ratio)
    return fields


def prepare_graph(data, cfg=None):
    """Return a VNData copy of one raw PPA graph, ready for PyG batching.

    Supply the experiment config to reproduce its VN settings. With no config,
    use the original minimum-separator defaults. Run before batching; moving
    the resulting batch to a GPU is supported. Node inputs are replaced with
    zeros because the PPA encoder has one embedding entry.
    """
    cfg = {} if cfg is None else cfg
    train_cfg = cfg.get('train', {})
    fields = separator_tree_fields(
        data,
        leaf_size=train_cfg.get('vn_leaf_size', cfg.get('data', {}).get('vn_leaf_size', 1)),
        separator_strategy=train_cfg.get('vn_separator', 'minimum'),
        separator_merge_limit=train_cfg.get('vn_separator_merge_limit', 16),
        separator_balance_ratio=train_cfg.get('vn_separator_balance_ratio', 0.75),
    )
    out = VNData(**data.clone().to_dict())
    device = out.edge_index.device
    out.x = torch.zeros(out.num_nodes, dtype=torch.long, device=device)
    for name, value in fields.items():
        out[name] = value.to(device)
    return out
