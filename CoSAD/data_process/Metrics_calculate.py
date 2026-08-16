import networkx as nx
import numpy as np
import itertools  # 用于生成参数组合
from data_process.Community_Search import Communtiy_Search_randomwalk
from data_process.Data_Query import Data_Query


def calculate_scene_metrics(data_list, abnormal_label=1, normal_label=0):
    """
    计算三个场景指标（仅区分正常/异常节点，无未知标签）：
    1. 异常节点聚集率：子图内异常节点数 / 子图总节点数
    2. 正常节点异常触达率：与异常节点相连的正常节点数 / 子图内正常节点总数
    3. 异常关联边网络渗透率：与异常节点相连的边数 / 子图总边数
    """
    known_cluster_rates = []  # 此处变量名保留，但实际已变为“异常节点聚集率”
    normal_reach_rates = []
    abnormal_edge_penetrations = []

    for data in data_list:
        num_nodes = data.node_labels.shape[0]  # 子图总节点数（无未知标签，全为正常/异常）
        node_labels = data.node_labels.numpy()
        edge_index = data.edge_index.numpy()

        # 1. 构建networkx图（节点属性仅含正常/异常标签）
        G = nx.Graph()
        for node_idx in range(num_nodes):
            G.add_node(node_idx, label=node_labels[node_idx])
        if edge_index.size > 0:
            edges = list(zip(edge_index[0], edge_index[1]))
            G.add_edges_from(edges)

        # 2. 计算：异常节点聚集率（原“已知节点聚集率”，无未知标签后简化）
        total_nodes = num_nodes  # 无未知标签，总节点数=有效节点数
        abnormal_count = sum(1 for node in G.nodes if G.nodes[node]["label"] == abnormal_label)
        # 处理子图无节点的极端情况（实际采样中极少出现）
        abnormal_cluster_rate = abnormal_count / total_nodes if total_nodes != 0 else 0.0

        # 3. 计算：正常节点异常触达率（仅筛选正常节点，无未知标签）
        # 子图内所有正常节点
        normal_nodes = [node for node in G.nodes if G.nodes[node]["label"] == normal_label]
        total_normal = len(normal_nodes)

        if total_normal == 0:
            # 子图无正常节点，触达率为0
            normal_reach_rate = 0.0
        else:
            # 子图内所有异常节点
            abnormal_nodes = [node for node in G.nodes if G.nodes[node]["label"] == abnormal_label]
            if not abnormal_nodes:
                # 子图无异常节点，触达率为0
                normal_reach_rate = 0.0
            else:
                # 异常节点的所有邻居（去重）
                related_nodes = set()
                for abn_node in abnormal_nodes:
                    related_nodes.update(G.neighbors(abn_node))
                # 邻居中的正常节点
                reachable_normal = [node for node in related_nodes if node in normal_nodes]
                reachable_normal_count = len(reachable_normal)
                normal_reach_rate = reachable_normal_count / total_normal

        # 4. 计算：异常关联边网络渗透率（逻辑不变，无未知标签影响）
        total_edges = G.number_of_edges()
        if total_edges == 0:
            abnormal_edge_penetration = 0.0
        else:
            abnormal_nodes = [node for node in G.nodes if G.nodes[node]["label"] == abnormal_label]
            # 统计与异常节点相连的边数
            abnormal_edge_count = sum(1 for u, v in G.edges if u in abnormal_nodes or v in abnormal_nodes)
            abnormal_edge_penetration = abnormal_edge_count / total_edges

        # 收集当前子图指标
        known_cluster_rates.append(abnormal_cluster_rate)  # 存入的是修改后的“异常节点聚集率”
        normal_reach_rates.append(normal_reach_rate)
        abnormal_edge_penetrations.append(abnormal_edge_penetration)

    # 计算所有子图的平均值（保留4位小数）
    avg_abnormal_cluster = round(np.mean(known_cluster_rates), 4) if known_cluster_rates else 0.0
    avg_normal_reach = round(np.mean(normal_reach_rates), 4) if normal_reach_rates else 0.0
    avg_abnormal_penetration = round(np.mean(abnormal_edge_penetrations), 4) if abnormal_edge_penetrations else 0.0

    # 返回值名称同步修改，更贴合实际含义
    return avg_abnormal_cluster, avg_normal_reach, avg_abnormal_penetration


# 主函数：网格搜索重启概率和阈值
def main():
    file_path = '../DataSet/Roman.npz'

    edges_df, features_df = Data_Query(file_path)

    # 定义网格搜索的参数组合（根据需要调整范围）
    restart_probs = [0.1, 0.15, 0.2, 0.25, 0.3]  # 重启概率候选值
    epsilons = [0.01, 0.05, 0.1, 0.15]  # 收敛阈值候选值
    num_samples = 400  # 每个参数组合采样子图数量
    seed = 43  # 固定种子保证可重复性

    # 打印表头
    print(
        f"{'重启概率':<10} {'阈值':<10} {'已知节点聚集率':<15} {'正常节点异常触达率':<20} {'异常关联边网络渗透率':<20}")
    print("-" * 80)

    # 遍历所有参数组合
    for rp, eps in itertools.product(restart_probs, epsilons):
        print(f"正在计算：重启概率={rp}, 阈值={eps}...")
        # 采样子图（传入当前参数组合）
        data_list = Communtiy_Search_randomwalk(edges_df, features_df, seed=seed,restart_prob=rp, epsilon=eps)
        # 计算指标
        avg_cluster, avg_reach, avg_penetration = calculate_scene_metrics(
            data_list=data_list,
            abnormal_label=1,
            normal_label=0,
        )
        # 打印当前组合的结果
        print(f"{rp:<10} {eps:<10} {avg_cluster:<15} {avg_reach:<20} {avg_penetration:<20}")


if __name__ == "__main__":
    main()
