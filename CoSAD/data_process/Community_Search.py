import torch
import numpy as np
import ArgParser
import random
import networkx as nx
import pandas as pd
from torch_geometric.data import Data
from data_process.Commonity_search_via_RRW import attribute_community_with_rrw, random_walk_subgraph, \
    bfs_sample_subgraph, importance_based_sample

#
# def select_nodes_via_label(features_df, label, num_nodes):
#     """
#     从特征数据中筛选节点集合
#
#     参数：
#         features_df: DataFrame，包含节点数据，必须包含'node_id'和'label'列
#         label: 目标标签值（如3、4等）
#         num_nodes: 需要选择的节点数量
#
#     返回：
#         tuple: 两个集合
#             - 第一个集合：标签为label的节点ID（数量为num_nodes或全部可用节点）
#             - 第二个集合：标签不为label的节点ID（数量为num_nodes或全部可用节点）
#     """
#     # 1. 筛选标签为目标label的节点
#     label_nodes = features_df[features_df['label'] == label]
#     # 筛选标签不为目标label的节点
#     non_label_nodes = features_df[features_df['label'] != label]
#
#     # 2. 检查节点数量是否足够，不足则取全部
#     available_label = len(label_nodes)
#     available_non_label = len(non_label_nodes)
#
#     # 实际选择的节点数（不超过可用数量）
#     select_label_num = min(num_nodes, available_label)
#     select_non_label_num = min(num_nodes, available_non_label)
#
#     # 3. 随机选择节点（使用sample确保随机性，若数量为0则返回空列表）
#     selected_label_ids = label_nodes.sample(n=select_label_num, random_state=42)[
#         'node_id'].tolist() if available_label > 0 else []
#     selected_non_label_ids = non_label_nodes.sample(n=select_non_label_num, random_state=42)[
#         'node_id'].tolist() if available_non_label > 0 else []
#
#     # 4. 转换为列表并返回
#     label_set = list(selected_label_ids)
#     non_label_set = list(selected_non_label_ids)
#
#     # 打印提示信息（可选，方便验证）
#     # print(f"已选择标签为{label}的节点{select_label_num}个（共{available_label}个可用）")
#     # print(f"已选择标签不为{label}的节点{select_non_label_num}个（共{available_non_label}个可用）")
#
#     return label_set, non_label_set
def select_nodes_via_label(features_df, num_nodes, abnormal_label=1, normal_label=0, random_state=42):
    """
    从二分类标签中按比例抽取节点（异常优先占半）

    策略：
        - 目标异常节点数 = num_nodes // 2
        - 如果异常节点不够，全取异常，其余由正常节点补齐
        - 最终返回的异常+正常节点总数等于 num_nodes（假设总节点充足）

    参数：
        features_df: DataFrame，必须包含 'node_id' 和 'label' 列
        num_nodes: 需要抽取的总节点数
        abnormal_label: 异常标签值（默认为1）
        normal_label: 正常标签值（默认为0）
        random_state: 随机种子，保证可重复性

    返回：
        tuple: (异常节点ID列表, 正常节点ID列表)
    """
    # 1. 按标签分离数据
    abnormal_nodes = features_df[features_df['label'] == abnormal_label]
    normal_nodes = features_df[features_df['label'] == normal_label]

    avail_ab = len(abnormal_nodes)
    avail_norm = len(normal_nodes)

    # 2. 计算应取节点数
    target_ab = num_nodes // 2                     # 理想情况：异常占一半
    target_ab = min(target_ab, avail_ab)          # 不足则全取异常
    target_norm = num_nodes - target_ab           # 其余由正常补齐
    target_norm = min(target_norm, avail_norm)    # 防止正常也不够（极限情况）

    # 3. 随机抽样
    selected_ab = (abnormal_nodes.sample(n=target_ab, random_state=random_state)['node_id'].tolist()
                   if target_ab > 0 else [])
    selected_norm = (normal_nodes.sample(n=target_norm, random_state=random_state)['node_id'].tolist()
                     if target_norm > 0 else [])

    # 4. 输出信息（可选）
    # print(f"异常节点选取: {target_ab}/{avail_ab}, 正常节点选取: {target_norm}/{avail_norm}")

    return selected_ab, selected_norm


def build_data_from_subgraph(G, subgraph_nodes, features_df):
    """根据子图构建Data对象"""
    node_indices = [features_df[features_df.iloc[:, 0] == n].index[0] for n in subgraph_nodes]
    features = features_df.iloc[node_indices, 1:-1].values
    node_labels = features_df.iloc[node_indices, -1].values
    node_labels = torch.tensor(node_labels, dtype=torch.long)

    x = torch.tensor(features, dtype=torch.float)

    subgraph_node_index_map = {node: idx for idx, node in enumerate(subgraph_nodes)}
    edge_list = list(G.subgraph(subgraph_nodes).edges())
    if len(edge_list) > 0:
        edge_index = torch.tensor([[subgraph_node_index_map[u], subgraph_node_index_map[v]]
                                   for u, v in edge_list],
                                  dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, node_labels=node_labels)


def Communtiy_Search_randomwalk(edges_df, features_df, num_samples, com_ser=" ", seed=43, restart_prob=0.3,
                                epsilon=0.0001, max_iter=200):
    """社区搜索（接受重启概率和阈值参数）"""
    random.seed(seed)
    G = nx.from_pandas_edgelist(edges_df, source=edges_df.columns[0], target=edges_df.columns[1])

    ano_nodes, nor_nodes = select_nodes_via_label(features_df, 1, num_samples)
    all_nodes = ano_nodes + nor_nodes
    # all_nodes = ano_nodes
    data_list = []

    for start_node in random.sample(all_nodes, k=len(all_nodes)):

        if com_ser == "RWR":
            subgraph_nodes = random_walk_subgraph(G, start_node, restart_prob=restart_prob, epsilon=epsilon)
        elif com_ser == "PRWR":
            subgraph_nodes = attribute_community_with_rrw(G, start_node, features_df, restart_prob, max_iter=max_iter,
                                                          epsilon=epsilon, max_hop=50)
        elif com_ser == "BFS":
            subgraph_nodes = bfs_sample_subgraph(G, start_node, 29, max_depth=None)
        elif com_ser == "Importance":
            subgraph_nodes = importance_based_sample(G, start_node, importance_metric='degree', sample_size=29)

        if not subgraph_nodes:
            continue
        data = build_data_from_subgraph(G, subgraph_nodes, features_df)

        if (data.node_labels == 1).any():
            data.y = torch.tensor([1], dtype=torch.long)
        else:
            data.y = torch.tensor([0], dtype=torch.long)

        data_list.append(data)

    return data_list
