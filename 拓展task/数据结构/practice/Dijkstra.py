graphA = {
    "start": {
        "A": 5,
        "B": 2
    },
    "A": {
        "C": 4,
        "D": 2
    },
    "B": {
        "A": 8,
        "D": 7
    },
    "C": {
        "D": 6,
        "END": 3
    },
    "D": {
        "END": 1
    },
    "END": {
    }
}
cost = {
    "A": 5,
    "B": 2,
    "C": float("inf"),
    "D": float("inf"),
    "END": float("inf")
}


costdemo = {
    "A": 5,
    "B": 2,
    "C": 1,
    "D": 8,
    "END": 6
}
parents = {
    "A": "start",
    "B": "start",
    "C": None,
    "D": None,
    "END": None
}
processed = []


def finlown(cost, processed):
    lowest_node = float("inf")
    node = None
    for n in cost:
        if cost[n] < lowest_node and n not in processed:
            lowest_node = cost[n]
            node = n
    return node


def comsho(parents, cost):
    node = parents[end]
    cost[node]


def finshortest(end, graph, cost, processed, parents):
    '''
    1.从起点开始，找最近的那条路->去最近的那个节点
    2.到达最近的节点后，更新该节点连接的邻居的距离，并在cost中更新
    3.标记这个节点到processed之中
    4.开启下一轮的循环更新
    '''
    shortest = float("inf")
    node = finlown(cost, processed)
    while node is not None:
        neighbors = graph[node]
        for n in neighbors:
            now_n = cost[node]+neighbors[n]
            if now_n < cost[n]:
                cost[n] = now_n
                parents[n] = node
        processed.append(node)
        node = finlown(cost, processed)
    shortest = cost[end]
    return shortest, parents


if __name__ == "__main__":
    len, path = finshortest('END', graphA, cost, processed, parents)
    print("最短路径长度为", len)
    print("最短路径为", path)
