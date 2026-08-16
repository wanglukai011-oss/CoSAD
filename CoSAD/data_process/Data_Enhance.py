import torch
import copy
from torch_geometric.utils import to_undirected
import torch.nn.functional as F
import random


class UltimateTripleStreamAugmentor:
    def __init__(
            self,
            feature_noise=0.02,
            edge_drop_rate=0.15,
            path_jitter_rate=0.2,
            feature_pull_ratio=0.3,
            homophily_th=0.7,
            path_dropout_rate=0.3,
            camouflage_only_for_anomaly=True
    ):
        self.feature_noise = feature_noise
        self.edge_drop_rate = edge_drop_rate
        self.path_jitter_rate = path_jitter_rate
        self.feature_pull_ratio = feature_pull_ratio
        self.homophily_th = homophily_th
        self.path_dropout_rate = path_dropout_rate
        self.camouflage_only_for_anomaly = camouflage_only_for_anomaly

    def feature_noise_aug(self, data):
        aug = copy.deepcopy(data)
        noise = torch.randn_like(aug.x) * self.feature_noise
        aug.x = aug.x + noise
        return aug

    def edge_drop_aug(self, data):
        aug = copy.deepcopy(data)
        edge_index = aug.edge_index
        if edge_index.numel() == 0:
            return aug
        src, dst = edge_index
        keep_mask = torch.rand(src.size(0), device=src.device) > self.edge_drop_rate
        keep_mask = keep_mask | (src == dst)
        aug.edge_index = to_undirected(edge_index[:, keep_mask])
        return aug

    def path_jitter_aug(self, data):
        aug = copy.deepcopy(data)
        if not hasattr(aug, "node_path_weight"):
            return aug
        scale = 1.0 + (torch.rand_like(aug.node_path_weight) - 0.5) * 2 * self.path_jitter_rate
        aug.node_path_weight = torch.clamp(aug.node_path_weight * scale, min=0.0)
        return aug

    # ======================================================
    # 🚀 核心新增：即时随机增强 (专为一致性正则化设计)
    # 保证在 Batch 内原图和增强图维度完美对齐
    # ======================================================
    def random_augment_batch(self, batch_data):
        """
        在训练循环内部，对整个 Batch 进行即时、随机的单流扰动。
        """
        aug_type = random.choice(['feature', 'edge', 'path'])
        if aug_type == 'feature':
            return self.feature_noise_aug(batch_data)
        elif aug_type == 'edge':
            return self.edge_drop_aug(batch_data)
        else:
            return self.path_jitter_aug(batch_data)
