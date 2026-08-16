import numpy as np
import torch


class RobustEarlyStopping:
    def __init__(self, patience=20, min_delta=0.001, smoothing=3,
                 restore_best_weights=True, verbose=True):
        """
        参数:
            smoothing: 平滑窗口大小。只有当连续窗口内的平均值没有提升时，才增加计数。
                       这可以过滤掉图数据随机采样带来的指标抖动。
        """
        self.patience = patience
        self.min_delta = min_delta
        self.smoothing = smoothing
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.counter = 0
        self.best_score = -np.inf
        self.best_weights = None
        self.recent_scores = []  # 用于存储最近几个 epoch 的分值进行平滑

    def __call__(self, score, model):
        # 1. 异常值保护
        if np.isnan(score):
            if self.verbose: print("⚠️ 警告: 验证集指标为 NaN，跳过此次记录。")
            return False

        self.recent_scores.append(score)
        if len(self.recent_scores) > self.smoothing:
            self.recent_scores.pop(0)

        # 2. 计算平滑分值（抗抖动核心）
        smoothed_score = np.mean(self.recent_scores)

        # 3. 判断改进逻辑
        # 使用相对改进或绝对改进。对于 AUC 这种 0-1 的指标，绝对改进更好。
        if smoothed_score > self.best_score + self.min_delta:
            if self.verbose and smoothed_score > self.best_score:
                print(f"📈 验证集指标提升: {self.best_score:.4f} -> {smoothed_score:.4f}. 保存模型权重。")

            self.best_score = smoothed_score
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"⏳ 验证集无显著提升 ({self.counter}/{self.patience}).")

        # 4. 触发早停
        should_stop = self.counter >= self.patience
        if should_stop:
            if self.verbose: print(f"🛑 早停触发! 最佳验证集分值: {self.best_score:.4f}")
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)

        return should_stop