import torch.nn.functional as F
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def visualize_homophily_violin(datasets, all_similarities):
    plt.figure(figsize=(16, 8))

    # 绘制小提琴图并显示所有统计线条
    positions = np.arange(1, len(datasets) + 1)
    parts = plt.violinplot(
        all_similarities,
        positions=positions,
        showmeans=True,  # 显示均值线
        showmedians=True,  # 显示中位数线
        showextrema=True,  # 显示极值线（最大值/最小值）
        widths=0.6,  # 调整小提琴图宽度
        bw_method='silverman'  # 核密度估计方法
    )

    # 自定义小提琴图颜色（调整为6个颜色，对应6个数据集）
    violin_colors = ["#adb2bb", "#f5ad92", "#F7AA58", "#FFD06F", "#72BCD5", "#528FAD"]
    for i, (pc, color) in enumerate(zip(parts['bodies'], violin_colors)):
        pc.set_facecolor(color)
        pc.set_edgecolor('#666666')  # 小提琴边缘颜色保持深灰
        pc.set_alpha(1)

    # 自定义统计线条样式（均值、中位数、极值线）
    for line_type in ['cmeans', 'cmedians', 'cmins', 'cmaxes']:
        if line_type in parts:
            if line_type == 'cmeans':
                parts[line_type].set_color('#ff0000')  # 均值线设为红色
                parts[line_type].set_linestyle('--')
                parts[line_type].set_linewidth(1.5)
            elif line_type == 'cmedians':
                parts[line_type].set_color('#ff0000')  # 中位数线设为黑色
                parts[line_type].set_linewidth(2)
            elif line_type == 'cmaxes':
                parts[line_type].set_color('#000000')  # 极值数线设为黑色
                parts[line_type].set_linewidth(2)
            elif line_type == 'cmins':  # 须线（极值线）颜色设置
                parts[line_type].set_color('#666666')  # 明确须线颜色
                parts[line_type].set_linewidth(1)

    # 创建虚拟绘图元素用于图例
    mean_line = plt.Line2D([0], [0], color='#ff0000', linestyle='--', linewidth=1.5, label='Mean')
    median_line = plt.Line2D([0], [0], color='#ff0000', linestyle='-', linewidth=2, label='Median')
    extrema_line = plt.Line2D([0], [0], color='#000000', linestyle='-', linewidth=1.2, label='Extrema')

    # 添加图例到右下角
    plt.legend(handles=[mean_line, median_line, extrema_line],
               loc='lower left',  # 修正为正确的右下角
               fontsize=20,
               frameon=False,  # 不显示图例边框
               framealpha=0.9)  # 若需要边框可调整此参数

    # 设置坐标轴标签
    plt.xlabel('Datasets', fontsize=20)
    plt.ylabel('Generalized Edge Homophily Ratio', fontsize=20)

    # 设置坐标轴刻度
    plt.xticks(positions, datasets, fontsize=20)  # 可根据情况添加rotation参数调整标签角度
    plt.yticks(fontsize=20)

    # 调整布局
    plt.tight_layout()

    # 保存为 .pdf 文件
    plt.savefig('homophily_ratio.pdf', format='pdf', dpi=300, bbox_inches='tight')

    # 显示图表
    plt.show()


def calculate_avg_nodes_edges(data_list):
    total_nodes = 0
    total_edges = 0
    num_data = len(data_list)

    for data in data_list:
        # 计算节点数，x的行数就是节点数
        nodes_count = data.x.shape[0]
        total_nodes += nodes_count
        # 计算边数，edge_index的列数就是边数
        edges_count = data.edge_index.shape[1]
        total_edges += edges_count

    avg_nodes = total_nodes / num_data if num_data != 0 else 0
    avg_edges = total_edges / num_data if num_data != 0 else 0

    return avg_nodes, avg_edges


def calculate_feature_heterogeneity(data_list):
    all_similarities = []
    for data in data_list:
        x = data.x  # 节点特征
        edge_index = data.edge_index  # 边索引
        similarities = []

        for edge in edge_index.t().tolist():  # 遍历每一条边
            u, v = edge
            x_u = x[u].unsqueeze(0)
            x_v = x[v].unsqueeze(0)

            # 计算余弦相似性
            similarity = F.cosine_similarity(x_u, x_v, dim=1).item()

            # 将余弦相似度映射到 [0, 1] 范围
            # mapped_similarity = (similarity + 1) / 2.0
            similarities.append(similarity)

        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            avg_similarity_rounded = round(avg_similarity, 2)
            all_similarities.append(avg_similarity_rounded)
        else:
            all_similarities.append(0.0)

    return all_similarities


# 示例使用
if __name__ == "__main__":
    # # 只保留6个数据集的计算
    minesweeper = torch.load("./minesweeper_data_list.pt", weights_only=False)
    minesweeper_sim = calculate_feature_heterogeneity(minesweeper)

    actor = torch.load("./actor_data_list.pt", weights_only=False)
    actor_sim = calculate_feature_heterogeneity(actor)

    arxiv = torch.load("./arxiv_data_list.pt", weights_only=False)
    arxiv_sim = calculate_feature_heterogeneity(arxiv)

    elliptic = torch.load("./elliptic_data_list.pt", weights_only=False)
    elliptic_sim = calculate_feature_heterogeneity(elliptic)

    dgraph = torch.load("./dgraph_data_list.pt", weights_only=False)
    dgraph_sim = calculate_feature_heterogeneity(dgraph)

    roman = torch.load("./roman_data_list.pt", weights_only=False)
    roman_sim = calculate_feature_heterogeneity(roman)
    # 调整数据集列表为6个
    datasets = ['Actor', 'Minesweeper', 'Roman', 'Arxiv', 'Elliptic', 'DGraph']

    # 调整相似度列表为6个，顺序与数据集列表对应
    all_similarities = [actor_sim, minesweeper_sim, roman_sim, arxiv_sim, elliptic_sim, dgraph_sim]
    visualize_homophily_violin(datasets, all_similarities)


# 保留的辅助函数调用示例（如需使用可取消注释）
# results = calculate_feature_heterogeneity(data_list)
# print("特征异质性结果:", results)

# avg_nodes, avg_edges = calculate_avg_nodes_edges(data_list)
# print(f"平均节点数: {avg_nodes}")
# print(f"平均边数: {avg_edges}")

# count = sum(1 for data in data_list if data.y.item() == 1)
# print(f"data.y 等于 1 的子图个数: {count}")
# count1 = sum(1 for data in data_list if data.y.item() == 0)
# print(f"data.y 等于 0 的子图个数: {count1}")
