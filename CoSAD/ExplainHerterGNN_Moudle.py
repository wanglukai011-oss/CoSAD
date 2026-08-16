import torch
from torch_scatter import scatter_max
import math
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, GCNConv, global_max_pool
from torch_geometric.utils import add_self_loops
from collections import deque


class DHAGConv(MessagePassing):
    """增强版异质性处理卷积层（含动态阈值）"""

    def __init__(self, in_dim, out_dim, dropout_prob=0.5, num_heads=4):
        super().__init__(aggr='add')
        self.out_dim = out_dim
        self.num_heads = num_heads
        assert out_dim % num_heads == 0, "out_dim必须能被num_heads整除"

        # 特征变换层
        self.self_proj = nn.Linear(in_dim, out_dim)
        self.neigh_proj = nn.Linear(in_dim, out_dim)

        # Dropout层
        self.dropout = nn.Dropout(p=dropout_prob)

        # 移除原固定阈值参数
        # 新增动态阈值记录
        self.dynamic_threshold = None
        self.attention_weights = None

    def compute_dynamic_threshold(self, x, edge_index):
        """
        动态计算特征相似度阈值
        返回:
            threshold: 标量，全图节点与邻居的平均相似度
        """
        # 分离梯度以避免影响反向传播
        x_detached = x.detach()

        # 获取源节点和目标节点索引
        src, dst = edge_index

        # 提取对应特征
        src_feat = x_detached[src]  # (E, feat_dim)
        dst_feat = x_detached[dst]  # (E, feat_dim)

        # 批量计算余弦相似度（按边计算）
        similarity = F.cosine_similarity(src_feat, dst_feat, dim=-1)  # (E,)

        # 计算全局平均相似度（作为阈值）
        threshold = similarity.mean()

        return threshold

    def forward(self, x, edge_index):
        # 计算动态阈值（前向传播时实时计算）
        self.dynamic_threshold = self.compute_dynamic_threshold(x, edge_index)

        # 添加自环边
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # 特征投影
        x_self = self.self_proj(x)
        x_neigh = self.neigh_proj(x)

        # 应用Dropout
        x_self = self.dropout(x_self)
        x_neigh = self.dropout(x_neigh)

        # 消息传播
        aggr_out = self.propagate(edge_index, x=(x_self, x_neigh))

        # 残差连接
        return F.leaky_relu(aggr_out + x_self)

    def message(self, x_i, x_j):
        # 多头注意力计算
        head_dim = self.out_dim // self.num_heads
        query = x_i.view(-1, self.num_heads, head_dim)
        key = x_j.view(-1, self.num_heads, head_dim)

        # 计算注意力得分
        attn_logits = (query * key).sum(dim=-1) / math.sqrt(head_dim)
        attn_weights = F.softmax(attn_logits, dim=1)

        # 记录注意力权重（取平均）
        self.attention_weights = attn_weights.mean(dim=1).detach()

        # 应用动态阈值（重要修改点）
        current_threshold = self.dynamic_threshold.detach()  # 分离计算图
        mask = (self.attention_weights > current_threshold).float()
        attn_weights = attn_weights * mask.unsqueeze(1)

        # 结合门控和注意力权重
        weighted = (x_j.view(-1, self.num_heads, head_dim)
                    * attn_weights.unsqueeze(-1))

        return weighted.view(-1, self.out_dim)


class GatedLCFUnit(nn.Module):
    def __init__(self, hidden_dim, dropout_prob=0.5):
        super().__init__()

        # 门控融合机制 - 简化结构
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),  # 减少一层
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.bn = nn.BatchNorm1d(hidden_dim)

        self.global_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout_prob)
        )

    def forward(self, x1, x2, batch):
        # 避免重复计算门控值
        gate_value = self.gate(torch.cat([x1, x2], dim=-1))
        fused = gate_value * x1 + (1 - gate_value) * x2

        fused = self.bn(fused)
        fused = self.global_net(fused)

        # 使用多种池化方法获得更丰富的全局特征
        fused_global_max = global_max_pool(fused, batch)
        # fused_global_mean = global_mean_pool(fused, batch)
        # 保持原有维度，但融合两种池化结果
        # fused_global = (fused_global_max + fused_global_mean) / 2

        return fused, fused_global_max


class MASNetwork(nn.Module):
    def __init__(self, hidden_dim, use_graph_classifier=""):
        super().__init__()
        self.use_graph_classifier = use_graph_classifier

        # 预测网络 - 添加dropout防止过拟合
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.2),  # 添加dropout
            nn.Linear(hidden_dim // 2, 1)
        )

        # 全局与局部融合的注意力权重计算模块
        self.global_local_attention = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),  # 添加dropout
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, fused_local, fused_global, batch):
        if self.use_graph_classifier == "union":
            # 节点级预测
            node_scores = self.scorer(fused_local).squeeze(-1)
            local_scores, max_indices = scatter_max(node_scores, batch, dim=0)
            max_features = fused_local[max_indices]

            # 图级预测
            graph_scores = self.scorer(fused_global).squeeze(-1)

            # 计算注意力权重
            alpha = self.global_local_attention(
                torch.cat([max_features, fused_global], dim=-1)
            ).squeeze(-1)

            # final_anomaly_scores = 0.5 * local_scores + 0.5 * graph_scores
            final_anomaly_scores = alpha * local_scores + (1 - alpha) * graph_scores

            return final_anomaly_scores, local_scores, graph_scores, alpha

        elif self.use_graph_classifier == "graph":
            graph_scores = self.scorer(fused_global).squeeze(-1)
            final_anomaly_scores = graph_scores
            return final_anomaly_scores, 0, graph_scores, 0

        else:
            node_scores = self.scorer(fused_local).squeeze(-1)
            local_scores, _ = scatter_max(node_scores, batch, dim=0)
            final_anomaly_scores = local_scores
            return final_anomaly_scores, local_scores, 0, 0


# 可解释的异质图异常检测模型（双任务：图级监督+节点级无监督异常检测）
class ExplainableGNN(nn.Module):
    def __init__(self, feat_dim, hidden_dim=512, dropout_prob=0.5,
                 use_conv_layers=True, use_enhanced_fusion=True, use_MASNet="union"):
        super().__init__()
        self.use_conv_layers = use_conv_layers
        self.use_enhanced_fusion = use_enhanced_fusion
        self.use_MASNet = use_MASNet

        # 输入投影层
        self.input_proj = nn.Linear(feat_dim, hidden_dim)

        # 卷积层配置
        if self.use_conv_layers:
            self.conv1 = DHAGConv(hidden_dim, hidden_dim, dropout_prob)
            self.conv2 = DHAGConv(hidden_dim, hidden_dim, dropout_prob)

        else:
            self.conv1 = GCNConv(hidden_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # 特征融合模块
        if self.use_enhanced_fusion:
            self.fusion = GatedLCFUnit(hidden_dim, dropout_prob)

        # 综合分类模块
        self.classifier = MASNetwork(hidden_dim, use_graph_classifier=self.use_MASNet)

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x, edge_index, batch):
        # 输入投影
        x = self.input_proj(x)

        # 卷积层
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        #
        # x1 = self.conv1(x, edge_index, batch)  # 添加 batch 参数
        # x2 = self.conv2(x1, edge_index, batch)  # 添加 batch 参数

        # 特征融合
        if self.use_enhanced_fusion:
            fused_local, fused_global = self.fusion(x1, x2, batch)
        else:
            fused_local = x2
            fused_global = global_max_pool(x2, batch)

        # 分类
        anomalous_scores, node_scores, graph_scores, alpha = self.classifier(
            fused_local, fused_global, batch)

        return anomalous_scores, node_scores, graph_scores, alpha

    def total_loss(self, data, weight_decay=1e-5):
        pred, node_scores, graph_scores, alpha = self.forward(
            data.x, data.edge_index, data.batch)

        # 主损失
        if self.use_MASNet == "union":
            main_loss = F.binary_cross_entropy_with_logits(pred, data.y.float())
        elif self.use_MASNet == "graph":
            main_loss = F.binary_cross_entropy_with_logits(graph_scores, data.y.float())
        elif self.use_MASNet == "local":
            main_loss = F.binary_cross_entropy_with_logits(node_scores, data.y.float())

        # 可选：添加L2正则化
        l2_reg = torch.tensor(0.).to(pred.device)
        for param in self.parameters():
            l2_reg += torch.norm(param)

        return main_loss + weight_decay * l2_reg


# 跨层特征融合模块（保持不变）
class FeatureFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

    def forward(self, x1, x2):
        gate = self.gate(torch.cat([x1, x2], dim=-1))
        return gate * x1 + (1 - gate) * x2  # 门控加权融合


class GCNClassifier(torch.nn.Module):
    def __init__(self, feat_dim, hidden_dim=512):
        super().__init__()
        self.conv1 = GCNConv(feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.graph_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, batch):
        # 图卷积层
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)

        # 全局平均池化
        fusion_x = global_mean_pool(x2, batch)  # 聚合为图级别表示

        # 分类层
        pred = self.graph_scorer(fusion_x).squeeze()
        return pred, x1, x2, edge_index

    def total_loss(self, data):
        pred, _, _, _ = self.forward(data.x, data.edge_index, data.batch)
        # 主分类损失
        cls_loss = F.binary_cross_entropy_with_logits(pred, data.y.float())

        return cls_loss
