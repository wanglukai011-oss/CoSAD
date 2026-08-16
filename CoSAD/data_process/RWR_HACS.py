import os
import torch
import numpy as np
from torch_geometric.data import Data
import random
from collections import defaultdict
from tqdm import tqdm
from data_process.Community_Search import select_nodes_via_label
import time
from sklearn.mixture import GaussianMixture
from scipy.stats import hypergeom
from sklearn.mixture import GaussianMixture
from scipy.stats import hypergeom
import time
from tqdm import tqdm

# ========== 调试配置：开启CUDA同步执行，精准定位报错行，调试必备 ==========
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# ====================== 全局基础配置 - 所有参数可按需调整 ======================
# 运行设备：优先使用GPU加速，无GPU自动使用CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 固定随机种子，保证实验结果可复现
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
# 数据类型：float32，兼顾精度和显存占用，GNN标准配置
DTYPE = torch.float32
EPS = 1e-8  # 极小值，防止除零错误和NaN值，所有除法必加

# ========== 全局缓存：减少重复计算，提升运行效率 ==========
_SIM_CACHE = {}
_DEGREE_CACHE = {}


# ====================== 图数据预处理函数【GPU加速版+内存优化】- 无改动+完整注释 ======================
def preprocess_graph_gpu(edges_df, features_df):
    """
    极速版：图数据预处理，构建 CSR 格式索引实现 O(1) 邻居查询
    """
    node_id_list = features_df['node_id'].tolist()
    node2idx = {node_id: idx for idx, node_id in enumerate(node_id_list)}
    idx2node = {idx: node_id for node_id, idx in node2idx.items()}
    N = len(node_id_list)

    edge_u = [node2idx[uid] for uid in edges_df.iloc[:, 0].tolist()]
    edge_v = [node2idx[vid] for vid in edges_df.iloc[:, 1].tolist()]
    all_edge_index = torch.tensor([edge_u, edge_v], dtype=torch.long, device=DEVICE)
    all_edge_index = torch.cat([all_edge_index, all_edge_index.flip(0)], dim=1)

    # ================= 🚀 核心优化：构建 CSR (Compressed Sparse Row) 索引 =================
    src, dst = all_edge_index
    # 按照 src 排序，保证同一节点的邻居在内存中连续
    sort_idx = torch.argsort(src)
    src = src[sort_idx]
    dst = dst[sort_idx]
    all_edge_index = torch.stack([src, dst])

    # 计算 ptr (指针数组)：ptr[i] 到 ptr[i+1] 的切片就是节点 i 的所有邻居
    degrees = torch.bincount(src, minlength=N)
    ptr = torch.zeros(N + 1, dtype=torch.long, device=DEVICE)
    ptr[1:] = torch.cumsum(degrees, dim=0)
    # ====================================================================================

    feature_cols = features_df.columns[1:-1]
    node_feat_np = features_df[feature_cols].values.astype(np.float32)
    node_feat_np = node_feat_np / (np.linalg.norm(node_feat_np, axis=1, keepdims=True) + EPS)
    node_feat_tensor = torch.tensor(node_feat_np, dtype=DTYPE, device=DEVICE)

    node_label_tensor = torch.tensor(features_df['label'].values, dtype=torch.long, device=DEVICE)

    # 注意：我们返回了 ptr 和 dst，彻底淘汰了慢速的 mask 查找
    return all_edge_index, node2idx, idx2node, node_feat_tensor, node_label_tensor, ptr, dst


# ==========================================================
# 🚀 极速两阶段贪心寻路 (支持自动化敏感性测试 + 严格防越界版)
# ==========================================================
def dap_arwr_single_query_balanced(q_id, node2idx, idx2node, all_edge_index,
                                   ptr, dst, node_feat_tensor, node_label_tensor,
                                   min_community_size, max_community_size, q_quantile,
                                   restart_prob=0.15, max_rwr_iter=100, rwr_tol=1e-6):
    """
    基于重启随机游走的异配感知社区搜索 (RWR-HACS) —— 单查询节点版本

    策略：
        1. 计算查询节点与所有节点的余弦相似度（属性亲和度）
        2. 以余弦相似度作为边转移权重，执行重启随机游走，得到稳态得分 rwr_scores
        3. 使用 rwr_scores 进行两阶段社区扩张：
           阶段一：动态阈值保底生长至 min_community_size
           阶段二：全局质量 Q(C) 守恒的贪心扩张至 max_community_size

    参数：
        q_id:                 查询节点 ID (原始 id)
        node2idx:             dict, 原始 id -> 全局索引
        idx2node:             dict, 全局索引 -> 原始 id (本函数未直接使用，保留接口)
        all_edge_index:       全图边索引 (2, E) 用于最终子图提取
        ptr, dst:             CSR 格式邻接信息 (ptr 为行偏移, dst 为列索引)
        node_feat_tensor:     节点特征矩阵 (N, d)
        node_label_tensor:    节点标签向量 (N,)
        min_community_size:   最小社区规模
        max_community_size:   最大社区规模
        q_quantile:           第一阶段动态阈值分位数 (0~1)
        restart_prob:         RWR 重启概率 (默认 0.15)
        max_rwr_iter:         RWR 最大迭代次数
        rwr_tol:              RWR 收敛容差 (L1 范数)

    返回：
        Data 对象，包含子图特征、边、节点标签、查询节点局部索引、
        社区质量 (平均 RWR 得分) 等信息；若查询节点无效则返回 None
    """
    # ==================== 0. 查询节点校验 ====================
    q_idx = node2idx.get(q_id, -1)
    if q_idx < 0:
        return None

    community_nodes = {q_idx}
    N = node_feat_tensor.size(0)

    # ==================== 1. 计算余弦相似度 (保留原有属性度量) ====================
    q_feat = node_feat_tensor[q_idx].unsqueeze(0)
    cos_sim = torch.nn.functional.cosine_similarity(q_feat, node_feat_tensor, dim=1).clamp(min=0.0)
    cos_sim[q_idx] = 1.0

    # ==================== 2. 构建转移矩阵并运行 RWR ====================
    # 2.1 重建边的源节点 (利用 ptr 与 dst)
    deg = ptr[1:] - ptr[:-1]  # 每个节点的出度
    src = torch.arange(N, device=DEVICE).repeat_interleave(deg)

    # 2.2 边权重 = 目标节点的余弦相似度
    edge_weights = cos_sim[dst]

    # 2.3 行归一化，保证每个节点的出边转移概率和为 1
    sum_w = torch.zeros(N, device=DEVICE)
    sum_w.index_add_(0, src, edge_weights)  # sum_w[i] = Σ_{j ∈ N(i)} cos_sim[j]
    sum_w = torch.where(sum_w > 0, sum_w, torch.ones_like(sum_w))  # 避免除零
    norm_weights = edge_weights / sum_w[src]

    # 2.4 幂迭代求解 RWR 稳态分布
    p = torch.zeros(N, device=DEVICE)
    p[q_idx] = 1.0
    for _ in range(max_rwr_iter):
        # 传播：p_new_partial = p @ M
        p_prop = torch.zeros(N, device=DEVICE)
        p_prop.index_add_(0, dst, p[src] * norm_weights)
        # 重启
        p_new = (1 - restart_prob) * p_prop
        p_new[q_idx] += restart_prob
        # 收敛检查
        if torch.abs(p_new - p).sum() < rwr_tol:
            p = p_new
            break
        p = p_new

    rwr_scores = p  # 最终稳态得分，替代原余弦相似度作为扩张度量

    # ==================== 3. 辅助函数：获取外部一阶邻居 ====================
    def get_external_neighbors(comm_set):
        if not comm_set:
            return torch.tensor([], dtype=torch.long, device=DEVICE)
        comm_tensor = torch.tensor(list(comm_set), dtype=torch.long, device=DEVICE)
        neighs = [dst[ptr[n]: ptr[n + 1]] for n in comm_tensor]
        if not neighs:
            return torch.tensor([], dtype=torch.long, device=DEVICE)
        all_n = torch.unique(torch.cat(neighs))
        return all_n[~torch.isin(all_n, comm_tensor)]

    # ==================== 4. 阶段一：保底生长至 min_community_size ====================
    while len(community_nodes) < min_community_size:
        # 硬上限保护
        if len(community_nodes) >= max_community_size:
            break
        neighbors = get_external_neighbors(community_nodes)
        if len(neighbors) == 0:
            break

        # 使用 RWR 得分
        sims = rwr_scores[neighbors]

        # 动态阈值
        if len(neighbors) <= 3:
            threshold = torch.mean(sims).item()
        else:
            threshold = torch.quantile(sims, q_quantile).item()

        valid_mask = sims >= threshold
        valid_candidates = neighbors[valid_mask]

        if len(valid_candidates) > 0:
            # 按得分降序加入，实时监控最大规模
            valid_sims = sims[valid_mask]
            sorted_idx = torch.argsort(valid_sims, descending=True)
            sorted_candidates = valid_candidates[sorted_idx]
            for candidate in sorted_candidates:
                if len(community_nodes) >= max_community_size:
                    break
                community_nodes.add(candidate.item())
        else:
            # 防止死循环：强制拉取 RWR 得分最高的邻居
            if len(community_nodes) < max_community_size:
                best_idx = torch.argmax(sims)
                community_nodes.add(neighbors[best_idx].item())

    # ==================== 5. 阶段二：质量守恒贪心扩张 ====================
    if min_community_size <= len(community_nodes) < max_community_size:
        current_comm_tensor = torch.tensor(list(community_nodes), dtype=torch.long, device=DEVICE)
        q_base = torch.mean(rwr_scores[current_comm_tensor]).item()  # 初始社区平均 RWR 得分

        while len(community_nodes) < max_community_size:
            neighbors = get_external_neighbors(community_nodes)
            if len(neighbors) == 0:
                break

            sims = rwr_scores[neighbors]
            sorted_indices = torch.argsort(sims, descending=True)

            added_in_this_round = False
            for idx in sorted_indices:
                candidate = neighbors[idx].item()
                candidate_sim = sims[idx].item()
                current_size = len(community_nodes)
                # 模拟加入后的新社区平均质量
                q_new = (q_base * current_size + candidate_sim) / (current_size + 1)
                if q_new >= q_base - EPS:  # 允许极小浮点误差
                    community_nodes.add(candidate)
                    q_base = q_new
                    added_in_this_round = True
                    break  # 每轮只加入一个最优节点，随后重新获取邻居拓扑
                else:
                    break  # 最优候选都拉低质量，直接停止

            if not added_in_this_round:
                break

    # ==================== 6. 子图打包 ====================
    community_idx = list(community_nodes)
    community_idx_tensor = torch.tensor(community_idx, dtype=torch.long, device=DEVICE)

    # 6.1 节点特征与标签
    x = node_feat_tensor[community_idx_tensor].detach()
    x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
    node_labels = node_label_tensor[community_idx_tensor].detach()
    local_q_idx = community_idx.index(q_idx)

    # 6.2 提取社区内部边
    mask = torch.isin(all_edge_index[0], community_idx_tensor) & \
           torch.isin(all_edge_index[1], community_idx_tensor)
    edge_index_global = all_edge_index[:, mask]

    if edge_index_global.shape[1] > 0:
        global2local = torch.zeros(N, dtype=torch.long, device=DEVICE)
        global2local[community_idx_tensor] = torch.arange(len(community_idx), device=DEVICE)
        edge_index = global2local[edge_index_global]
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=DEVICE)

    # 改为基于余弦相似度：
    # avg_quality = torch.mean(cos_sim[community_idx_tensor]).item()
    # 计算社区内部边的两端节点余弦相似度的均值 (同配性度量 S_avg)
    if edge_index.shape[1] > 0:
        # edge_index 已是局部索引，需要映射回全局索引以获得特征
        global_edge_index = edge_index_global  # 之前用来提取边的全局边索引
        src_global = global_edge_index[0]
        dst_global = global_edge_index[1]
        # 提取两端节点的特征
        feat_src = node_feat_tensor[src_global]
        feat_dst = node_feat_tensor[dst_global]
        # 计算每条边的余弦相似度
        edge_cos = torch.nn.functional.cosine_similarity(feat_src, feat_dst, dim=1)
        edge_cos = torch.clamp(edge_cos, min=0.0)  # 可选，避免负值
        avg_quality = torch.mean(edge_cos).item()
    else:
        avg_quality = 0.0  # 无边时的默认值

    # 6.4 返回 Data 对象 (标签 y 待后续标注流程填充)
    return Data(x=x, edge_index=edge_index, node_labels=node_labels,
                q_id=torch.tensor([local_q_idx], dtype=torch.long, device=DEVICE),
                y=torch.tensor([-1], dtype=torch.long, device=DEVICE),
                community_quality=torch.tensor([avg_quality], dtype=DTYPE, device=DEVICE),
                p_val=None)


def dap_arwr_first_stage_gpu(edges_df, features_df, num_samples,
                             min_community_size=5, max_community_size=50, q_quantile=0.75,
                             target_label=1, max_attempts=20, progress_callback=None,
                             restart_prob=0.15, max_rwr_iter=100, rwr_tol=1e-6):
    """
    批量生成查询社区的顶层函数（兼容新版 RWR-HACS）

    新增参数:
        restart_prob:  RWR 重启概率 (默认 0.15)
        max_rwr_iter:  RWR 最大迭代次数
        rwr_tol:       RWR 收敛容差 (L1 范数)
    """
    start_time = time.time()
    all_edge_index, node2idx, idx2node, node_feat_tensor, node_label_tensor, ptr, dst = preprocess_graph_gpu(
        edges_df, features_df
    )

    # 全局背景统计
    all_labels_np = node_label_tensor.cpu().numpy()
    valid_mask = all_labels_np != -1
    N_star = int(np.sum(valid_mask))
    K_star = int(np.sum(all_labels_np[valid_mask] == 1))

    print(f"\n🌍 全局背景统计: 已知标签节点 N*={N_star}, 异常节点 K*={K_star}")
    print(f"⚙️ 图采样参数: k_min={min_community_size}, k_max={max_community_size}, q={q_quantile}")
    print(f"🔁 RWR 参数: restart_prob={restart_prob}, max_iter={max_rwr_iter}, tol={rwr_tol}")

    # 选择查询节点（异常/正常各半）
    anomaly_nodes, normal_nodes = select_nodes_via_label(features_df, num_samples)
    query_node_list = anomaly_nodes + normal_nodes
    data_list = []

    # 阶段一：社区搜索
    pbar = tqdm(total=len(query_node_list), desc="🔍 两阶段贪心寻路", ncols=180,
                bar_format="{l_bar}{bar:60}{r_bar}", colour="green")
    for idx, q_id in enumerate(query_node_list):
        try:
            current_data = dap_arwr_single_query_balanced(
                int(q_id), node2idx, idx2node, all_edge_index, ptr, dst,
                node_feat_tensor, node_label_tensor,
                min_community_size, max_community_size, q_quantile,
                restart_prob=restart_prob, max_rwr_iter=max_rwr_iter, rwr_tol=rwr_tol
            )
            if current_data is not None:
                data_list.append(current_data)
        except Exception as e:
            pass

        pbar.update(1)
        if (idx + 1) % 10 == 0 and len(data_list) > 0:
            cur_nodes = sum(d.x.shape[0] for d in data_list)
            pbar.set_postfix({'有效图': len(data_list), '均节点': f"{cur_nodes / len(data_list):.1f}"})
    pbar.close()

    # 阶段二：Fisher 检验与 GMM 标签推断（完全保留原有设计）
    print("\n🧠 正在进行 Fisher 检验与 GMM 标签推断 (过滤未知节点)...")
    valid_p_vals, valid_indices = [], []

    for i, d in enumerate(data_list):
        comm_labels = d.node_labels.cpu().numpy()
        val_mask = comm_labels != -1
        n_labeled = np.sum(val_mask)
        k_labeled = np.sum(comm_labels[val_mask] == 1)

        if n_labeled == 0:
            d.p_val = 2.0
        else:
            p_val = 1.0 if k_labeled == 0 else hypergeom.sf(k_labeled - 1, N_star, K_star, n_labeled)
            d.p_val = p_val
            valid_p_vals.append(p_val)
            valid_indices.append(i)

    pos_count, neg_count, gray_count = 0, 0, 0
    if len(valid_p_vals) > 10:
        p_vals_np = np.clip(np.array(valid_p_vals), a_min=1e-300, a_max=1.0)
        scores = -np.log(p_vals_np).reshape(-1, 1)

        gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=SEED)
        gmm.fit(scores)

        means = gmm.means_.flatten()
        normal_comp_idx, anomaly_comp_idx = np.argmin(means), np.argmax(means)

        post_probs = gmm.predict_proba(scores)
        prob_anomaly = post_probs[:, anomaly_comp_idx]
        prob_normal = post_probs[:, normal_comp_idx]

        confidence_tau = 0.90

        for list_idx, p_anom, p_norm in zip(valid_indices, prob_anomaly, prob_normal):
            if p_anom >= confidence_tau:
                data_list[list_idx].y = torch.tensor([1], dtype=torch.long, device=DEVICE)
                pos_count += 1
            elif p_norm >= confidence_tau:
                data_list[list_idx].y = torch.tensor([0], dtype=torch.long, device=DEVICE)
                neg_count += 1
            else:
                data_list[list_idx].y = torch.tensor([-1], dtype=torch.long, device=DEVICE)
                gray_count += 1

        for d in data_list:
            if d.p_val == 2.0:
                d.y = torch.tensor([-1], dtype=torch.long, device=DEVICE)
                gray_count += 1
    else:
        for d in data_list:
            d.y = torch.tensor([-1], dtype=torch.long, device=DEVICE)
            gray_count += 1

    # 统计信息
    total_nodes = sum(d.x.shape[0] for d in data_list)
    total_edges = sum(d.edge_index.shape[1] for d in data_list)
    avg_nodes = total_nodes / len(data_list) if len(data_list) > 0 else 0.0
    avg_edges = total_edges / len(data_list) if len(data_list) > 0 else 0.0
    avg_quality = sum(d.community_quality.item() for d in data_list) / len(data_list) if len(data_list) > 0 else 0.0

    print(f"\n✅ 生成完毕 | 耗时: {time.time() - start_time:.2f}s | 总社区数: {len(data_list)}")
    print(f"📊 标签分布 | 监督样本(0/1): {pos_count + neg_count} | 灰区样本(-1): {gray_count}")
    print(f"📏 规模统计 | 平均节点数: {avg_nodes:.1f} | 平均边数: {avg_edges:.1f} | 纯度 (S_avg): {avg_quality:.4f}\n")

    return data_list
