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

def get_vn_tree(n, edge):
    return get_centroid_tree(n, edge)

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
        if i % 50 == 0:
            print(i, "processing...")

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


from torch_geometric.data import InMemoryDataset
class CustomEasyDataset(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__(None)
        # collate 함수를 통해 리스트를 PyG 데이터셋 포맷으로 압축합니다.
        self.data, self.slices = self.collate(data_list)

def op(dataset):
    new_data_list = []
    
    for idx in range(len(dataset)):
        data = dataset[idx]
        n = data.num_nodes
        edge = data.edge_index.t().detach().cpu().tolist()
    
        print(type(edge), len(edge), len(edge[0]))
    
        print("starting tree decomposition")
        bags, bag_tree = tree_decomposition(n, edge)
        
        m = len(bags)
        adj_bag_tree = [[] for _ in range(m)]
        
        for u, v in bag_tree:
            adj_bag_tree[u].append(v)
            adj_bag_tree[v].append(u)
        
        base_vn_tree = get_vn_tree(m, adj_bag_tree)
        vn_edge = [(u + n, v + n) for u, v in base_vn_tree]
            
        new_tree = edge + vn_edge + get_vn_connect(n, m, bags)
        new_data_list.append(new_tree)

    return CustomEasyDataset(new_data_list)
