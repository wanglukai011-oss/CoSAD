import collections
import random
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque


class UnionFind:
    """并查集：精简版，只保留必要方法"""

    def __init__(self, nodes):
        self.parent = {node: node for node in nodes}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 路径压缩
        return self.parent[x]

    def union(self, x, y):
        self.parent[self.find(y)] = self.find(x)  # 合并（简化写法）

    def is_connected(self, x, y):
        return self.find(x) == self.find(y)


def random_walk_subgraph(G, start_node, restart_prob, max_iter=200, epsilon=0.0001):
    """带重启的随机游走（用于确定k值）"""
    if start_node not in G.nodes:
        return {start_node}

    visited = {start_node}
    current, steps, prev_size = start_node, 0, 1  # 初始大小含start_node

    while steps < max_iter:
        neighbors = list(G.neighbors(current))
        current = start_node if (not neighbors or random.random() < restart_prob) else random.choice(neighbors)
        visited.add(current)
        steps += 1

        # 每5步检查稳定性
        if steps % 5 == 0:
            current_size = len(visited)
            growth_ratio = (current_size - prev_size) / prev_size if prev_size else 1.0
            if growth_ratio < epsilon:
                break
            prev_size = current_size

    return visited


def check_connectivity(temp_community, G):
    """检查临时社区是否全连通"""
    if not temp_community:
        return True
    uf = UnionFind(temp_community)
    for node in temp_community:
        for neighbor in G.neighbors(node):
            if neighbor in temp_community:
                uf.union(node, neighbor)
    first_node = next(iter(temp_community))
    return all(uf.is_connected(first_node, n) for n in temp_community)


def attribute_community_with_rrw(G, query_node, feature_df, restart_prob, max_iter=200, epsilon=0.0001, max_hop=50):
    """修复temp_uf未定义问题，确保替换逻辑正确"""
    # -------------------------- 前置检查：确保node_id列存在 --------------------------
    if 'node_id' not in feature_df.columns:
        raise ValueError("特征表中缺少'node_id'列，请检查输入")

    # -------------------------- 步骤1：用RRW确定k值 --------------------------
    rrw_visited = random_walk_subgraph(G, query_node, restart_prob, max_iter, epsilon)
    k = len(rrw_visited)
    if k <= 1:
        return [query_node]  # 极端情况直接返回

    # -------------------------- 步骤2：提取有效特征（第二列到倒数第二列） --------------------------
    feature_cols = feature_df.iloc[:, 1:-1]  # 第二列到倒数第二列

    # 检查查询节点是否在node_id列中
    if not feature_df['node_id'].isin([query_node]).any():
        raise ValueError(f"特征表的node_id列中缺少查询节点{query_node}")

    # 提取查询节点的特征向量
    query_row = feature_df[feature_df['node_id'] == query_node].iloc[0]
    query_feature = query_row[feature_cols.columns].values.reshape(1, -1)

    # -------------------------- 步骤3：初始化社区 --------------------------
    community = {query_node}
    node_similarity = {query_node: 1.0}
    uf = UnionFind(community)  # 初始并查集仅含查询节点

    # -------------------------- 步骤4：逐层扩展与筛选 --------------------------
    for hop in range(1, max_hop + 1):
        if len(community) == k:
            break

        # 找当前阶的候选节点（BFS）
        candidates = set()
        visited_bfs = set(community)
        queue = deque([(query_node, 0)])
        while queue:
            node, dist = queue.popleft()
            if dist == hop:
                candidates.add(node)
            elif dist < hop:
                for neighbor in G.neighbors(node):
                    if neighbor not in visited_bfs:
                        visited_bfs.add(neighbor)
                        queue.append((neighbor, dist + 1))

        # 筛选有效候选
        valid_candidates = [
            n for n in candidates
            if n not in community
               and feature_df['node_id'].isin([n]).any()
        ]
        if not valid_candidates:
            continue

        # 计算相似度并排序
        candidate_sims = []
        for node in valid_candidates:
            node_row = feature_df[feature_df['node_id'] == node].iloc[0]
            node_feat = node_row[feature_cols.columns].values.reshape(1, -1)
            sim = cosine_similarity(query_feature, node_feat)[0][0]
            candidate_sims.append((node, sim))
        candidate_sims.sort(key=lambda x: -x[1])

        # 筛选与替换
        for node, sim in candidate_sims:
            if len(community) == k:
                break

            # 检查与社区的连通性
            community_neighbors = [n for n in G.neighbors(node) if n in community]
            if not community_neighbors:
                continue

            # 尝试替换低相似度节点
            replaceable = [n for n in community if n != query_node]
            if replaceable:
                min_sim_node = min(replaceable, key=lambda x: node_similarity[x])
                min_sim = node_similarity[min_sim_node]

                if sim > min_sim:
                    # 定义临时社区
                    temp_community = (community - {min_sim_node}) | {node}
                    # 关键修复：初始化临时并查集（之前漏了这步）
                    temp_uf = UnionFind(temp_community)
                    # 检查连通性
                    if check_connectivity(temp_community, G):
                        # 执行替换
                        community.remove(min_sim_node)
                        del node_similarity[min_sim_node]
                        community.add(node)
                        node_similarity[node] = sim
                        # 更新并查集为临时并查集（包含新节点）
                        uf = temp_uf
                        continue

            # 直接加入新节点
            if len(community) < k:
                community.add(node)
                node_similarity[node] = sim
                # 将新节点添加到并查集
                uf.parent[node] = node
                # 与社区邻居合并
                for n in community_neighbors:
                    uf.union(query_node, n)
                    uf.union(n, node)

    # 按相似度排序返回
    return sorted(community, key=lambda x: -node_similarity[x])[:k]


# BFS对比方法
def bfs_sample_subgraph(G, start_node, max_nodes, max_depth=None):
    """
    基于BFS的子图采样
    从起始节点开始，逐层扩展直到达到节点数或深度限制
    """
    if start_node not in G.nodes:
        return {start_node}

    visited = set([start_node])
    queue = collections.deque([start_node])

    while queue and len(visited) < max_nodes:
        current = queue.popleft()

        # 获取当前节点的邻居
        neighbors = list(G.neighbors(current))
        random.shuffle(neighbors)  # 随机打乱邻居顺序

        for neighbor in neighbors:
            if neighbor not in visited and len(visited) < max_nodes:
                visited.add(neighbor)
                queue.append(neighbor)

    return visited


def importance_based_sample(G, start_node, importance_metric='degree', sample_size=50):
    """
    基于节点重要性采样子图
    """
    # 计算节点重要性
    if importance_metric == 'degree':
        importance_scores = dict(G.degree())
    elif importance_metric == 'pagerank':
        importance_scores = nx.pagerank(G)
    elif importance_metric == 'betweenness':
        importance_scores = nx.betweenness_centrality(G)

    visited = set([start_node])
    candidates = set([start_node])

    while len(visited) < sample_size and candidates:
        # 从候选集中选择最重要的节点
        current = max(candidates, key=lambda x: importance_scores.get(x, 0))
        candidates.remove(current)

        # 添加邻居到候选集
        neighbors = set(G.neighbors(current)) - visited
        candidates.update(neighbors)
        visited.add(current)

        # 限制候选集大小
        if len(candidates) > sample_size * 2:
            # 保留最重要的候选节点
            top_candidates = sorted(candidates,
                                    key=lambda x: importance_scores.get(x, 0),
                                    reverse=True)[:sample_size * 2]
            candidates = set(top_candidates)

    return visited


# 使用示例
if __name__ == "__main__":
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0), (1, 3), (3, 4)])

    feature_df = pd.DataFrame({
        'node_id': [0, 1, 2, 3, 4],  # 节点ID列
        'feat1': [0.9, 0.7, 0.2, 0.6, 0.8],
        'feat2': [0.8, 0.6, 0.3, 0.5, 0.9],
    })

    community = attribute_community_with_rrw(
        G=G, query_node=0, feature_df=feature_df,
        restart_prob=0.3, max_hop=3
    )
    print(f"社区节点: {community}")
