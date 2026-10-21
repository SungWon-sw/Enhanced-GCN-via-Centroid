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
    s = [(0, 0)]; check = [0] * n; cent_tree = []
    while s:
        node, prev = s.pop()
        size = get_size(n, edge, check, node)
        cent = get_centroid(edge, size, check, node)

        if prev: cent_tree.append((prev, cent))

        check[cent] = True
        for next in edge[cent]:
            if check[next]: continue

            s.append((next, cent))     

    return cent_tree
