import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, to_undirected
from torch_scatter import scatter_mean, scatter_max, scatter_add
from torch_geometric.utils import softmax as pyg_softmax
from torch_scatter import scatter_mean, scatter_max
from torch_geometric.utils import to_dense_batch


class PathAwareConv(MessagePassing):
    def __init__(self, in_dim, out_dim):
        super().__init__(aggr="add")
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.self_proj = nn.Linear(in_dim, out_dim)
        self.neigh_proj = nn.Linear(in_dim, out_dim)
        self.diff_proj = nn.Linear(in_dim, out_dim)

        self.fusion_proj = nn.Linear(out_dim * 2, out_dim)
        self.sim_scale = nn.Parameter(torch.tensor(1.0))

        # ==========================================================
        # 🚀 核心改动：新增可学习的节点异常打分器 (Learnable Scorer)
        # 替代原本僵硬的同异配比例公式，让辅助损失直接教它做人！
        # ==========================================================
        self.node_scorer = nn.Linear(out_dim, 1)

    def forward(self, x, edge_index, q_indices_global=None, batch=None):
        device = x.device
        num_nodes = x.size(0)

        if num_nodes == 0:
            return torch.zeros(0, self.out_dim, device=device), torch.zeros(0, device=device)

        mask = ((edge_index[0] >= 0) & (edge_index[0] < num_nodes) & (edge_index[1] >= 0) & (edge_index[1] < num_nodes))
        edge_index = edge_index[:, mask]
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        edge_index = to_undirected(edge_index)
        src, dst = edge_index

        raw_sim = F.cosine_similarity(x[src], x[dst], dim=-1)
        w = torch.sigmoid(F.softplus(self.sim_scale) * raw_sim)

        x_self = self.self_proj(x)
        x_neigh = self.neigh_proj(x)
        x_diff = self.diff_proj(x)

        self._cache = {"src": src, "dst": dst, "w": w, "x_self": x_self, "x_neigh": x_neigh, "x_diff": x_diff}

        out_msg = self.propagate(edge_index, size=(num_nodes, num_nodes))
        agg_homo, agg_hetero = out_msg.chunk(2, dim=-1)

        # 1. 完美特征融合 (包含了自特征、同配邻居、异配邻居的所有信息)
        concat_features = torch.cat([agg_homo, agg_hetero], dim=-1)
        out = x_self + F.gelu(self.fusion_proj(concat_features))

        # ==========================================================
        # 🚀 核心改动：数据驱动的异常打分 (Data-driven Anomaly Scoring)
        # 将融合后的高维特征直接映射为一个标量分数。
        # 外层的 node_focal_loss 会把真实的异常标签梯度传回这里，
        # 强迫这个 scorer 自动学会识别什么是伪装异常，什么是团伙异常！
        # ==========================================================
        node_rel_score = self.node_scorer(out).squeeze(-1)  # 维度：[num_nodes]

        return out, node_rel_score

    def message(self, edge_index):
        c = self._cache
        src, dst, w = c["src"], c["dst"], c["w"]

        w_homo = F.relu(w - 0.5) * 2.0
        w_hetero = F.relu(0.5 - w) * 2.0

        msg_homo = w_homo.unsqueeze(-1) * c["x_neigh"][src]
        msg_hetero = w_hetero.unsqueeze(-1) * (c["x_self"][dst] - c["x_diff"][src])

        return torch.cat([msg_homo, msg_hetero], dim=-1)

    def message(self, edge_index):
        c = self._cache
        src, dst, w = c["src"], c["dst"], c["w"]

        # 🚀【修改 4】: 回归你最喜欢的 0.5 绝对截断逻辑！
        # 如果 w = 0.8 (强同配): w_homo = 0.6, w_hetero = 0.0 (只发同配)
        # 如果 w = 0.2 (强异配): w_homo = 0.0, w_hetero = 0.6 (只发异配)
        # 如果 w = 0.5 (模棱两可): w_homo = 0.0, w_hetero = 0.0 (全部截断拦截，完美去除噪声！)
        w_homo = F.relu(w - 0.5) * 2.0
        w_hetero = F.relu(0.5 - w) * 2.0

        # 同配消息：同配权重 × 邻居特征
        msg_homo = w_homo.unsqueeze(-1) * c["x_neigh"][src]

        # 异配消息：异配权重 × (自特征 - 邻居差异特征)
        msg_hetero = w_hetero.unsqueeze(-1) * (c["x_self"][dst] - c["x_diff"][src])

        # 平行发送，互不干涉
        return torch.cat([msg_homo, msg_hetero], dim=-1)


# ===================== 2. 池化层 (接管宏观统计) =====================

class RelationAnomalyAwarePooling(nn.Module):
    """
    场景自适应池化层 (彻底告别超参数：动态软阈值重构版)
    """

    def __init__(self, hidden_dim, dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 3 种核心视图投影
        self.global_proj = nn.Linear(hidden_dim, hidden_dim)
        self.anom_proj = nn.Linear(hidden_dim, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)

        # ==========================================================
        # 🚀 核心黑科技：动态阈值预测器 (Dynamic Threshold Predictor)
        # 输入：图的宏观统计量 (mean, max)
        # 输出：专属于这张图的异常截断阈值 tau
        # ==========================================================
        self.tau_predictor = nn.Sequential(
            nn.Linear(2, 8),
            nn.GELU(),
            nn.Linear(8, 1)
        )
        # 陡峭度参数 (初始值设大一点，让 Sigmoid 表现得更像一个绝对的阶跃截断开关)
        self.steepness = nn.Parameter(torch.tensor(10.0))

        # 视图路由器
        self.router = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, 3)
        )

        self.final_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, node_rel_score, q_indices_global, batch):
        device = x.device
        num_nodes = x.size(0)
        batch_size = int(batch.max().item()) + 1

        # 提前计算图的宏观统计量 (后面测阈值和做路由都要用)
        rel_mean = scatter_mean(node_rel_score, batch, dim=0)
        rel_max, _ = scatter_max(node_rel_score, batch, dim=0)
        graph_stats = torch.stack([rel_mean, rel_max], dim=-1)  # [B, 2]

        # ==========================================
        # 视图 1：全局均值池化 (兜底大盘)
        # ==========================================
        h_global_raw = scatter_mean(x, batch, dim=0, dim_size=batch_size)
        h_global = F.gelu(self.global_proj(h_global_raw))

        # ==========================================
        # 视图 2：动态软阈值异常池化 (你提出的自适应逻辑！)
        # ==========================================
        # 1. 预测每张图的动态阈值 tau
        tau = self.tau_predictor(graph_stats)  # [B, 1]

        # 2. 扩展 tau 到每个节点，方便对齐相减
        tau_expanded = tau[batch].squeeze(-1)  # [N]

        # 3. 计算软掩码 (Soft Mask)
        # 如果 node_score > tau，mask 趋近 1；反之趋近 0
        gamma = F.softplus(self.steepness)  # 保证陡峭度为正
        anom_mask = torch.sigmoid((node_rel_score - tau_expanded) * gamma)  # [N]

        # 4. 使用掩码进行加权平均池化 (只聚合越过阈值的节点)
        masked_x = x * anom_mask.unsqueeze(-1)
        sum_masked_x = scatter_add(masked_x, batch, dim=0, dim_size=batch_size)  # [B, D]
        sum_mask = scatter_add(anom_mask, batch, dim=0, dim_size=batch_size).unsqueeze(-1)  # [B, 1]

        # 避免除以 0，加一个极小值
        h_anom_raw = sum_masked_x / (sum_mask + 1e-5)
        h_anom = F.gelu(self.anom_proj(h_anom_raw))

        # ==========================================
        # 视图 3：查询节点池化
        # ==========================================
        if q_indices_global.numel() > 0:
            q_indices_global = torch.clamp(q_indices_global, 0, num_nodes - 1)
            q_feat = x[q_indices_global]
        else:
            q_feat = torch.zeros(batch_size, self.hidden_dim, device=device)

        if q_feat.size(0) < batch_size:
            padding = torch.zeros(batch_size - q_feat.size(0), self.hidden_dim, device=device)
            q_feat = torch.cat([q_feat, padding], dim=0)

        h_query = F.gelu(self.query_proj(q_feat))

        # ==========================================
        # 智能自适应路由
        # ==========================================
        query_sim = F.cosine_similarity(q_feat, h_global_raw, dim=-1).unsqueeze(-1)
        meta_features = torch.cat([graph_stats, query_sim], dim=-1)  # [B, 3]

        route_logits = self.router(meta_features)
        route_weights = F.softmax(route_logits, dim=-1)  # [B, 3]

        h_fused = (
                route_weights[:, 0:1] * h_global +
                route_weights[:, 1:2] * h_anom +
                route_weights[:, 2:3] * h_query
        )

        return self.final_proj(h_fused)


# ===================== 3. 损失函数 =====================
class AdaptiveMultiClassFocalLoss(nn.Module):
    def __init__(self, num_classes=4, reduction='mean'):
        super().__init__()
        self.reduction = reduction

        # 可学习的 Gamma (通过 softplus 保证 > 0)
        self.gamma_param = nn.Parameter(torch.tensor(1.0))

        # 可学习的 Alpha 类别权重 (初始为全 1，模型会根据数据分布自动调节)
        self.alpha_param = nn.Parameter(torch.ones(num_classes))

    def forward(self, logits, targets):
        # 使用 softmax 保证权重之和为 1，防权重爆炸
        dynamic_alpha = F.softmax(self.alpha_param, dim=0)
        alpha_weight = dynamic_alpha[targets]

        # 动态 gamma (基础值为 1.0)
        dynamic_gamma = F.softplus(self.gamma_param) + 1.0

        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        focal_loss = alpha_weight * ((1.0 - pt) ** dynamic_gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# =====================================================================
# 🚀 优化 2：全自适应二分类 Focal Loss (节点级别)
# 自动平衡正常节点和异常节点的权重分配
# =====================================================================
class AdaptiveBinaryFocalLossWithLogits(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

        # 可学习的 Gamma
        self.gamma_param = nn.Parameter(torch.tensor(1.0))

        # 可学习的正样本权重 Alpha (标量，经 Sigmoid 映射到 0~1)
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, logits, targets):
        dynamic_alpha = torch.sigmoid(self.alpha_logit)
        dynamic_gamma = F.softplus(self.gamma_param) + 1.0

        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)

        alpha_weight = targets * dynamic_alpha + (1.0 - targets) * (1.0 - dynamic_alpha)
        focal_loss = alpha_weight * ((1.0 - pt) ** dynamic_gamma) * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# =====================================================================
# 🚀 优化 3：多任务自动平衡器 (Multi-Task Balancer)
# 基于同方差不确定性，自动调节 图损失 和 节点损失 的比例
# =====================================================================
class AutomaticMultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化两个任务的对数方差 (用 log 保证数值稳定)
        self.log_vars = nn.Parameter(torch.zeros(2))

    def forward(self, loss_graph, loss_node):
        var_graph = torch.exp(self.log_vars[0])
        var_node = torch.exp(self.log_vars[1])

        # 核心公式：损失难降时增加 var，降低该任务权重，并附加正则项
        loss_1 = (loss_graph / (2.0 * var_graph)) + (self.log_vars[0] / 2.0)
        loss_2 = (loss_node / (2.0 * var_node)) + (self.log_vars[1] / 2.0)

        return loss_1 + loss_2


#

# ===================== 顶层主模型 =====================
class PathAwareGNN(nn.Module):
    def __init__(self, feat_dim, hidden_dim, dropout_prob=0.5):
        super().__init__()
        self.conv1 = PathAwareConv(feat_dim, hidden_dim)
        self.conv2 = PathAwareConv(hidden_dim, hidden_dim)

        self.query_attention_pool = RelationAnomalyAwarePooling(hidden_dim=hidden_dim, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim // 2, 1)  # 🚀 纯二分类输出
        )

        self.graph_focal_loss = AdaptiveBinaryFocalLossWithLogits()
        self.node_focal_loss = AdaptiveBinaryFocalLossWithLogits()
        self.multi_task_balancer = AutomaticMultiTaskLoss()

    def forward(self, x, edge_index, q_indices_global, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        if x.numel() == 0:
            return torch.zeros(0, 1, device=x.device), torch.zeros(0, device=x.device)

        x1, _ = self.conv1(x, edge_index, q_indices_global, batch)
        x1 = self.dropout(x1)
        x2, node_rel_score = self.conv2(x1, edge_index, q_indices_global, batch)
        x2 = self.dropout(x2)

        graph_repr = self.query_attention_pool(x2, node_rel_score, q_indices_global, batch)
        logits = self.classifier(graph_repr)
        return logits, node_rel_score

    # 🚀 半监督联合优化 Loss 计算 (彻底修复 Batch 索引)
    def total_loss(self, data, augmentor=None):
        # 兼容单图与 Batch
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(data.x.size(0),
                                                                                                 dtype=torch.long,
                                                                                                 device=data.x.device)
        batch_size = int(batch.max().item()) + 1

        # 精准构建 ptr
        ptr = torch.cat(
            [torch.tensor([0], device=batch.device), torch.bincount(batch, minlength=batch_size).cumsum(dim=0)])

        # 🚀 修正点：图偏移量 (ptr[:-1]) + 局部索引 (q_local)
        q_local = data.q_id.view(-1)
        q_indices_global = ptr[:-1] + q_local

        logits, node_rel_score = self(data.x, data.edge_index, q_indices_global, batch)
        probs = torch.sigmoid(logits.squeeze(-1))
        labels = data.y.float()

        labeled_mask = (labels != -1)
        gray_mask = (labels == -1)

        # 1. 监督分支
        if labeled_mask.any():
            loss_graph_sup = self.graph_focal_loss(logits.squeeze(-1)[labeled_mask], labels[labeled_mask])
        else:
            loss_graph_sup = (logits.sum() * 0.0)

        # 2. 一致性正则化分支 (处理灰区 -1)
        loss_consistency = 0.0
        if gray_mask.any() and augmentor is not None:
            augmented_data = augmentor.random_augment_batch(data)

            # 同样需要为增强数据构建安全的 batch
            aug_batch = augmented_data.batch if hasattr(augmented_data,
                                                        'batch') and augmented_data.batch is not None else torch.zeros(
                augmented_data.x.size(0), dtype=torch.long, device=augmented_data.x.device)
            aug_logits, _ = self(augmented_data.x, augmented_data.edge_index, q_indices_global, aug_batch)
            aug_probs = torch.sigmoid(aug_logits.squeeze(-1))

            # 使用 detach 切断梯度，保证性能
            loss_consistency = F.mse_loss(aug_probs[gray_mask], probs[gray_mask].detach())

        # 联合图损失
        loss_graph = loss_graph_sup + 0.5 * loss_consistency

        # 3. 节点分支
        if hasattr(data, 'node_labels') and data.node_labels is not None:
            node_targets = data.node_labels.float().squeeze()
            if node_rel_score.size(0) == node_targets.size(0):
                valid_mask = (node_targets != -1)
                if valid_mask.any():
                    loss_node = self.node_focal_loss(node_rel_score[valid_mask], node_targets[valid_mask])
                else:
                    loss_node = (node_rel_score.sum() * 0.0)
            else:
                loss_node = (node_rel_score.sum() * 0.0)
        else:
            loss_node = (node_rel_score.sum() * 0.0)

        return self.multi_task_balancer(loss_graph, loss_node)

    # 🚀 预测模块同步修复 Batch 索引
    def predict(self, data):
        with torch.no_grad():
            batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(data.x.size(0),
                                                                                                     dtype=torch.long,
                                                                                                     device=data.x.device)
            batch_size = int(batch.max().item()) + 1
            ptr = torch.cat(
                [torch.tensor([0], device=batch.device), torch.bincount(batch, minlength=batch_size).cumsum(dim=0)])

            q_local = data.q_id.view(-1)
            q_indices_global = ptr[:-1] + q_local

            logits, _ = self(data.x, data.edge_index, q_indices_global, batch)

            if logits.numel() == 0:
                return torch.empty(0, dtype=torch.float, device=data.x.device)

            probs = torch.sigmoid(logits.squeeze(-1))
            return probs
