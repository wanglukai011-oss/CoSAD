import numpy as np
import pandas as pd
import ArgParser


def Data_Query(file_path):
    # 加载 .npz 文件
    data = np.load(file_path)

    # 查看文件中的所有数组名称
    print("Arrays in the .npz file:", data.files)

    # 提取数据
    node_features = data['node_features']  # 节点特征
    edges = data['edges']  # 边结构

    total_cols = node_features.shape[1]
    feature_cols_count = total_cols - 2
    # 生成列名列表
    features_columns = ['node_id'] + [f'fea_{i}' for i in range(1, feature_cols_count + 1)] + ['label']

    features_df = pd.DataFrame(node_features, columns=features_columns)
    edges_df = pd.DataFrame(edges, columns=['node1', 'node2'])

    features_df['label'] = features_df['label'].astype(int)
    features_df['node_id'] = features_df['node_id'].astype(int)
    edges_df['node1'] = edges_df['node1'].astype(int)
    edges_df['node2'] = edges_df['node2'].astype(int)

    # # 可选：验证列名是否正确
    # print("\nfeatures_df列名：", features_df.columns.tolist())
    # print("edges_df列名：", edges_df.columns.tolist())
    #
    # # 打印示例
    # print("\nfeatures_df前5行：")
    # print(features_df.head())
    # print("\nedges_df前5行：")
    # print(edges_df.head())

    return edges_df, features_df


#
# file_path = "../DataSet/" + 'DGraph.npz'
# edge_df, fea_df = Data_Query(file_path)
# # 原有label统计逻辑（完整保留）
# label_counts = fea_df['label'].value_counts()
# print("各 label 的出现次数：")
# print(label_counts)
