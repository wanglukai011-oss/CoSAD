import pandas as pd
import numpy as np
# from ogb.nodeproppred import PygNodePropPredDataset
#
import ArgParser

#
# def ogb_load():
#     dataset = PygNodePropPredDataset(name="ogbn-arxiv")
#
#     split_idx = dataset.get_idx_split()
#     # train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
#     graph = dataset[0]  # pyg graph object
#
#     print(graph)
#     # 假设你的图数据对象名为 data（即 Data(...) 实例）
#     # 1. 转换为 NumPy 数组（若在 GPU 上，需先移到 CPU）
#     x_np = graph.x.cpu().numpy()  # 节点特征：(169343, 128)
#     edge_index_np = graph.edge_index.cpu().numpy().T  # 边索引：转置为 (1166243, 2)（每行一条边）
#     y_np = graph.y.cpu().numpy()  # 节点标签：(169343, 1)
#
#     # 2. 生成node_id（0到169342）
#     node_id = np.arange(x_np.shape[0]).reshape(-1, 1)  # 形状：(169343, 1)
#
#     # 3. 按顺序拼接：node_id → 特征 → 标签
#     combined = np.hstack([node_id, x_np, y_np])  # 拼接后形状：(169343, 1+128+1) = (169343, 130)
#
#     # 4. 设置列名
#     columns = ["node_id"] + [f"feature_{i}" for i in range(x_np.shape[1])] + ["label"]
#
#     # 5. 保存为CSV
#     combined_df = pd.DataFrame(combined, columns=columns)
#     combined_df.to_csv("Arxiv_features.csv", index=False)
#
#     # # 3. 保存 edge_index（边索引）
#     # # 列名设为 source（源节点）和 target（目标节点）
#     # edge_df = pd.DataFrame(edge_index_np, columns=["source", "target"])
#     # edge_df.to_csv("Arxiv_edges.csv", index=False)
#
#     print("保存完成！生成文件：node_features.csv, edges.csv, node_labels.csv")


def Actor_Data_Transform(edge_path, fea_path, save_path):  # 新增save_path参数，指定npz保存路径
    edge_df = pd.read_csv(edge_path, sep='\t')
    fea_df = pd.read_csv(fea_path, sep='\t')

    # 特征列名（你的数据中是'feature'）
    feature_col = 'feature'

    # 将特征列拆分为整数列表（增强鲁棒性：处理空值、None等情况）
    fea_df['feature_list'] = fea_df[feature_col].str.split(',').apply(
        lambda x: [int(i) for i in x] if (x and x[0] != '') else []
    )

    # 计算最大维度
    max_dim = max(len(feat) for feat in fea_df['feature_list'])
    print(f"自动计算的最大特征维度：{max_dim}")

    # 处理特征：超过截断、不足填充0，统一长度为max_dim
    def process_features(feat_list, target_dim):
        truncated = feat_list[:target_dim]
        padded = truncated + [0] * (target_dim - len(truncated))
        return padded

    # 应用处理函数
    fea_df['fixed_features'] = fea_df['feature_list'].apply(
        lambda x: process_features(x, max_dim)
    )

    # 拆分特征列表为列（fea_1~fea_max_dim）
    feat_expanded = fea_df['fixed_features'].apply(pd.Series)
    feat_expanded.columns = [f'fea_{i}' for i in range(1, max_dim + 1)]

    # 合并特征列到原始DataFrame，删除无用列（原始feature列+中间过程列）
    fea_df = pd.concat([fea_df, feat_expanded], axis=1)
    fea_df = fea_df.drop(columns=[feature_col, 'feature_list', 'fixed_features'])  # 关键：删除原始feature列

    # label转换逻辑（保留原有）
    fea_df.loc[fea_df['label'] != 0, 'label'] = 2
    fea_df.loc[fea_df['label'] == 0, 'label'] = 1
    fea_df.loc[fea_df['label'] == 2, 'label'] = 0

    # 核心修改：将label列从第二列移到最后一列
    label_col = fea_df.pop('label')  # 取出label列（原位置删除）
    fea_df['label'] = label_col  # 重新添加到最后一列

    # 转换为numpy数组，准备保存
    node_features = fea_df.values  # fea_df对应node_features（含所有特征列+最后一列label）
    edges = edge_df.values  # edge_df对应edges

    # 保存为.npz格式
    np.savez(save_path, node_features=node_features, edges=edges)
    print(f"数据已保存至 {save_path}，包含 'node_features' 和 'edges' 键")

    return edge_df, fea_df


def Arxiv_Data_Transform(edge_path, fea_path, save_path):  # 新增save_path参数，指定npz保存路径
    edge_df = pd.read_csv(edge_path)
    fea_df = pd.read_csv(fea_path)
    fea_df['label'] = fea_df['label'].astype(int)
    fea_df['node_id'] = fea_df['node_id'].astype(int)

    # 原有label转换逻辑（完整保留）
    fea_df.loc[fea_df['label'] == 0, 'label'] = 1
    fea_df.loc[fea_df['label'] != 1, 'label'] = 0

    # 原有label统计逻辑（完整保留）
    label_counts = fea_df['label'].value_counts()
    print("各 label 的出现次数：")
    print(label_counts)

    # 转换为numpy数组，准备保存
    node_features = fea_df.values  # fea_df 对应 npz 的 'node_features'
    edges = edge_df.values  # edge_df 对应 npz 的 'edges'

    # 保存为.npz格式
    np.savez(save_path, node_features=node_features, edges=edges)
    print(f"\n数据已保存至 {save_path}，包含键：'node_features'、'edges'")

    return edge_df, fea_df


def DGraph_Data_Transform():
    args = ArgParser.parse_args()
    main_path = args.filePath
    filename = main_path + 'dgraphfin.npz'

    # 1. 加载原始.npz文件数据
    data = np.load(filename)
    print("Arrays in the original .npz file:", data.files)

    # 提取原始数据
    features = data['x']  # 节点特征
    labels = data['y']  # 节点标签
    edges = data['edge_index']  # 边结构
    labels[(labels == 2) | (labels == 3)] = -1

    # ===================== 新增：打印Label标签信息 =====================
    # 1. 打印标签的唯一类型
    unique_labels = np.unique(labels)
    print("\n===== Label标签的唯一类型 =====")
    print(unique_labels)

    # 2. 打印每个标签的样本数量（统计分布）
    # label_counts = np.bincount(labels)
    # print("===== 各标签对应的节点数量 =====")
    # for label, count in enumerate(label_counts):
    #     print(f"标签 {label} : {count} 个节点")
    # ==================================================================

    n_nodes = features.shape[0]  # 节点数量（特征行数）

    # 生成node_id：从0开始递增（形状为 (n_nodes, 1)）
    node_ids = np.arange(n_nodes).reshape(-1, 1)  # 转换为列向量，方便拼接

    features_df = pd.DataFrame(features)

    # 在第一列插入node_id
    features_df.insert(0, 'node_id', node_ids)  # 插入后第一列为node_id，后续为原始特征列

    # 将labels转换为DataFrame（列名为'label'）
    labels_df = pd.DataFrame(labels, columns=['label'])

    # 按行拼接：features_df（含node_id+特征） + labels_df（label列）
    features_df = pd.concat([features_df, labels_df], axis=1)

    # 给features_df设置列名
    total_cols = features_df.shape[1]  # 总列数：1（node_id） + n_features + 1（label）
    n_feature_cols = total_cols - 2  # 中间特征列数量（减去node_id和label）
    # 生成列名列表
    feature_columns = ['node_id'] + [f'fea_{i}' for i in range(1, n_feature_cols + 1)] + ['label']
    features_df.columns = feature_columns  # 赋值列名

    edges_df = pd.DataFrame(edges, columns=['node1', 'node2'])  # 命名列

    # 5. 转换为numpy数组并保存为新的.npz文件
    # node_features：保留所有列（node_id + 特征 + label）
    node_features = features_df.values
    # edges：边数据（node1 + node2）
    edges = edges_df.values

    # 保存路径
    save_path = '../DataSet/DGraph.npz'
    np.savez(save_path, node_features=node_features, edges=edges)
    print(f"Processed data saved to {save_path} with keys: 'node_features', 'edges'")

    return edges_df, features_df


def Elliptic_Data_Transform(node_path, edge_path, fea_path, save_path):
    # 1. 获取数据路径
    node_filename = node_path
    edge_filename = edge_path
    feature_filename = fea_path

    # 2. 处理节点数据
    nodes_df = pd.read_csv(node_filename)
    nodes_df['class'].replace('unknown', '-1', inplace=True)
    nodes_df.rename(columns={'txId': 'node', 'class': 'label'}, inplace=True)
    nodes_df['label'] = nodes_df['label'].astype(int)

    # 3. 处理边数据
    edges_df = pd.read_csv(edge_filename)
    edges_df.rename(columns={'txId1': 'node1', 'txId2': 'node2'}, inplace=True)
    edges_df['node1'] = edges_df['node1'].astype(int)
    edges_df['node2'] = edges_df['node2'].astype(int)

    # 4. 处理特征数据
    features_df = pd.read_csv(feature_filename, header=None)
    features_df.columns = ['node_id', 'timestep'] + [f'feature_{i}' for i in range(1, features_df.shape[1] - 1)]
    features_df = features_df.iloc[:, :95]  # 保留前95列

    # 关联label
    features_df = pd.merge(
        features_df,
        nodes_df[['node', 'label']],
        left_on='node_id',
        right_on='node',
        how='left'
    )
    # 删除冗余的'node'列（来自nodes_df）和'timestep'列（按您原代码保留）
    features_df.drop(columns=['node', 'timestep'], inplace=True)

    # 5. 统一节点ID映射
    all_nodes = pd.concat([
        edges_df['node1'],
        edges_df['node2'],
        features_df['node_id']
    ]).unique()
    node_mapping = {node: idx for idx, node in enumerate(sorted(all_nodes))}

    # 6. 替换节点ID为映射后的索引
    features_df['node_id'] = features_df['node_id'].map(node_mapping)
    edges_df['node1'] = edges_df['node1'].map(node_mapping)
    edges_df['node2'] = edges_df['node2'].map(node_mapping)

    # 保留您的label转换逻辑
    features_df.loc[features_df['label'] == 2, 'label'] = 0

    # 7. 转换为numpy数组（保留node_id列）
    # 按node_id排序（确保节点顺序一致），直接取所有列转换为数组
    node_features = features_df.sort_values(by='node_id').values
    # edges：提取node1和node2
    edges = edges_df[['node1', 'node2']].values

    # 8. 保存为.npz格式
    np.savez(save_path, node_features=node_features, edges=edges)
    print(f"数据已保存至 {save_path}，包含 'node_features'（含node_id）和 'edges'")

    return edges_df, features_df


def Minesweeper_Data_Transform(file_path, save_path):
    # 加载 .npz 文件
    data = np.load(file_path)

    # 查看文件中的所有数组名称
    print("Arrays in the .npz file:", data.files)

    # 提取数据并转换为DataFrame
    node_features = data['node_features']  # 节点特征
    node_labels = data['node_labels']  # 节点标签
    edges = data['edges']  # 边结构

    # 处理特征列：添加标题fea_0到fea_i（i为特征列数-1）
    features_df = pd.DataFrame(node_features)
    # 生成特征列名：fea_0, fea_1, ..., fea_{n-1}（n为特征列数）
    feature_cols = [f'fea_{i}' for i in range(features_df.shape[1])]
    features_df.columns = feature_cols

    # 在第一列添加node_id（从0开始递增）
    features_df.insert(0, 'node_id', range(len(features_df)))  # 0表示插入到第一列

    # 处理标签并合并到特征表最后一列
    labels_df = pd.DataFrame(node_labels, columns=['label'])  # 给标签列命名为'label'
    # 合并特征和标签（标签放最后一列）
    features_df = pd.concat([features_df, labels_df], axis=1)

    # 处理边数据（保持原有结构）
    edges_df = pd.DataFrame(edges)

    # label_counts = features_df['label'].value_counts()
    #
    # # 打印结果
    # print("各 label 的出现次数：")
    # print(label_counts)

    # 打印处理后的特征表（包含node_id、特征列、label）
    # print(features_df)
    # print(edges_df)
    # 转换为numpy数组，准备保存
    node_features = features_df.values  # fea_df 对应 npz 的 'node_features'
    edges = edges_df.values  # edge_df 对应 npz 的 'edges'

    # 保存为.npz格式
    np.savez(save_path, node_features=node_features, edges=edges)
    print(f"\n数据已保存至 {save_path}，包含键：'node_features'、'edges'")

    return edges_df, features_df


def Roman_Data_Transform(file_path, save_path):
    # 加载 .npz 文件
    data = np.load(file_path)

    # 查看文件中的所有数组名称
    print("Arrays in the .npz file is:", data.files)

    # # 提取数据并转换为DataFrame
    node_features = data['node_features']  # 节点特征
    node_labels = data['node_labels']  # 节点标签
    edges = data['edges']  # 边结构

    # 处理特征列：添加标题fea_0到fea_i（i为特征列数-1）
    features_df = pd.DataFrame(node_features)
    # 生成特征列名：fea_0, fea_1, ..., fea_{n-1}（n为特征列数）
    feature_cols = [f'fea_{i}' for i in range(features_df.shape[1])]
    features_df.columns = feature_cols

    # 在第一列添加node_id（从0开始递增）
    features_df.insert(0, 'node_id', range(len(features_df)))  # 0表示插入到第一列

    # 处理标签并合并到特征表最后一列
    labels_df = pd.DataFrame(node_labels, columns=['label'])  # 给标签列命名为'label'
    # 合并特征和标签（标签放最后一列）
    features_df = pd.concat([features_df, labels_df], axis=1)

    # 处理边数据（保持原有结构）
    edges_df = pd.DataFrame(edges)
    features_df.loc[features_df['label'] == 1, 'label'] = 0
    features_df.loc[features_df['label'] == 15, 'label'] = 1
    features_df.loc[features_df['label'] == 16, 'label'] = 1
    features_df.loc[features_df['label'] != 1, 'label'] = 0

    node_features = features_df.values  # fea_df 对应 npz 的 'node_features'
    edges = edges_df.values  # edge_df 对应 npz 的 'edges'

    # 保存为.npz格式
    np.savez(save_path, node_features=node_features, edges=edges)
    print(f"\n数据已保存至 {save_path}，包含键：'node_features'、'edges'")

    # label_counts = features_df['label'].value_counts()
    #
    # # 打印结果
    # print("各 label 的出现次数：")
    # print(label_counts)

    # 打印处理后的特征表（包含node_id、特征列、label）
    # print(features_df)
    # print(edges_df)
    return edges_df, features_df

# data = np.load("../DataSet/Actor.npz")
# print("node_features shape:", data["node_features"].shape)  # 节点特征+label的形状
# print("edges shape:", data["edges"].shape)  # 边的形状
# edges_df, features_df = DGraph_Data_Transform()