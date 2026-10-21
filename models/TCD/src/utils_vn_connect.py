import torch
from torch_geometric.data import Data

def get_size(n, edge, check, start):
    s = [(start, 0, False)]; size = [0] * n
    while s:
        node, prev, visited = s.pop()

        if visited:
            size[prev] += size[node]
        else:
            size[node] = 1; s.append((node, prev, True))
            for next in edge[node]:
                if prev == next or check[next]: continue
                s.append((next, node, False))

    return size

def get_centroid(edge, size, check, start):
    s = [(start, 0)]; cent = 0
    while s:
        node, prev = s.pop()
        
        for next in edge[node]:
            if prev == next or check[next]: continue
            if size[next] * 2 > size[start]: s.append((next, node))

        if not s: cent = node; break

    return cent

def get_centroid_tree(n, edge):
    s = [(0, -1)]; check = [0] * n; cent_tree = []
    while s:
        node, prev = s.pop()
        size = get_size(n, edge, check, node)
        cent = get_centroid(edge, size, check, node)

        if prev != -1: cent_tree.append((prev, cent))

        check[cent] = True
        for next in edge[cent]:
            if check[next]: continue

            s.append((next, cent))     

    return cent_tree

def get_vn_connect(n, m, bag):
    vn_connect = []
    for i in range(n, n+m):
        for node in bag[i - n]:
            vn_connect.append((i, node))
    return vn_connect

from heapq import heapify, heappop, heappush

def tree_decomposition(n, edges, max_width=None):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u != v:
            adj[u].add(v); adj[v].add(u)

    heap = [(len(adj[v]), v) for v in range(n)]
    heapify(heap)
    alive = [True] * n
    order, bag = [], [None] * n

    for i in range(n):
        while True:
            d, v = heappop(heap)
            if alive[v] and len(adj[v]) == d:
                break
        nb = list(adj[v])
        bag[v] = frozenset(nb + [v])
        for i, a in enumerate(nb):
            for b in nb[i+1:]:
                if b not in adj[a]:
                    adj[a].add(b); adj[b].add(a)
        for u in nb:
            adj[u].discard(v)
            heappush(heap, (len(adj[u]), u))
        alive[v] = False
        adj[v] = set()
        order.append(v)

    pos = {v: i for i, v in enumerate(order)}
    bags = [bag[v] for v in order]
    tree = [(i, pos[min(bag[v] - {v}, key=pos.get)])
            for i, v in enumerate(order) if len(bag[v]) > 1]
    return bags, tree

class VNData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'vn_edge_index':
            return int(self.num_vn)
        
        if key == 'vn_incidence':
            return torch.tensor([[self.num_nodes], [int(self.num_vn)]])
        
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ('vn_edge_index', 'vn_incidence'):
            return 1
        
        return super().__cat_dim__(key, value, *args, **kwargs)


def build_centroid_forest(adj):
    """bag tree/forest를 centroid tree/forest로 변환.

    adj: bag의 무방향 인접 리스트
    반환: VN 간 단방향 (parent, child) 목록
    """
    m = len(adj)
    removed = [False] * m
    centroid_edges = []

    # 분리된 component도 각각 처리
    for seed in range(m):
        if removed[seed]:
            continue

        pending = [(seed, -1)]

        while pending:
            start, parent_centroid = pending.pop()
            if removed[start]:
                continue

            # 현재 component의 DFS 순서와 부모
            parent = {start: -1}
            order = []
            stack = [start]

            while stack:
                v = stack.pop()
                order.append(v)

                for u in adj[v]:
                    if removed[u] or u in parent:
                        continue
                    parent[u] = v
                    stack.append(u)

            # subtree 크기
            sizes = {v: 1 for v in order}

            for v in reversed(order):
                p = parent[v]
                if p != -1:
                    sizes[p] += sizes[v]

            # 제거 시 가장 큰 component가 절반 이하인 centroid
            component_size = len(order)
            centroid = None

            for v in order:
                largest = component_size - sizes[v]

                for u in adj[v]:
                    if parent.get(u) == v:
                        largest = max(largest, sizes[u])

                if largest * 2 <= component_size:
                    centroid = v
                    break

            if centroid is None:
                raise ValueError('bag 연결 구조가 tree/forest인지 확인하세요.')

            if parent_centroid != -1:
                centroid_edges.append((parent_centroid, centroid))

            removed[centroid] = True

            for u in adj[centroid]:
                if not removed[u]:
                    pending.append((u, centroid))

    return centroid_edges


def add_ppa_virtual_nodes(data):
    """원본 PPA 그래프에 node feature와 VN 필드 추가."""
    n = int(data.num_nodes)
    if n == 0:
        raise ValueError('노드가 없는 그래프는 지원하지 않습니다.')

    # 기존 tree_decomposition() 재사용
    edges = data.edge_index.t().cpu().tolist()
    bags, bag_tree = tree_decomposition(n, edges)
    m = len(bags)

    # 1. bag tree의 무방향 인접 리스트
    adj = [[] for _ in range(m)]

    for u, v in bag_tree:
        adj[u].append(v)
        adj[v].append(u)

    # 2. centroid tree/forest 생성
    centroid_edges = build_centroid_forest(adj)

    # VN끼리 양방향 메시지를 주고받도록 역방향도 추가
    if centroid_edges:
        forward_edges = torch.tensor(
            centroid_edges, dtype=torch.long
        ).t().contiguous()

        vn_edge_index = torch.cat(
            [forward_edges, forward_edges.flip(0)],
            dim=1,
        )
    else:
        vn_edge_index = torch.empty((2, 0), dtype=torch.long)

    # 3. 원본 노드마다 VN 지정
    rows, cols = [], []
    for k, bag in enumerate(bags):
        for v in bag:
            rows.append(int(v))
            cols.append(k)
    vn_incidence = torch.tensor([rows, cols], dtype=torch.long)

    # 4. 원본 edge_index, edge_attr, y 등을 유지
    result = VNData(**data.to_dict())
    result.num_nodes = n
    if result.get('x', None) is None:
        result.x = torch.zeros(n, dtype=torch.long)

    # VN 번호는 원본 노드 번호와 별개로 0부터 시작
    result.vn_edge_index = vn_edge_index
    result.vn_incidence = vn_incidence
    result.num_vn        = torch.tensor(m)

    return result
