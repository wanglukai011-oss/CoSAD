import random
import torch
import ArgParser
import numpy as np
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
from data_process.Community_Search import Communtiy_Search_randomwalk
from data_process.Data_Query import Data_Query
from ExplainHerterGNN_Moudle import ExplainableGNN, GCNClassifier
from data_process.Eearly_Stopping import MultiMetricEarlyStopping


def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for data in loader:
        # 确保数据在正确的设备上
        data = data.to(device, non_blocking=True)
        optimizer.zero_grad()

        total_loss = model.total_loss(data)

        # 反向传播
        total_loss.backward()

        optimizer.step()

        # 记录损失
        total_loss += total_loss.item()

    return total_loss / len(loader)


def test(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in loader:
            # 确保数据在正确的设备上
            data = data.to(device)
            pred, _, _, _ = model(data.x, data.edge_index, data.batch)
            # 确保模型输出是1维的
            if pred.dim() == 0:  # 如果输出是标量（0维），则将其扩展为1维
                pred = pred.unsqueeze(0)
            y_true.append(data.y.cpu().numpy())
            y_pred.append(pred.cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # 检查 y_true 是否只有一个类别
    if len(np.unique(y_true)) < 2:
        print("Warning: Only one class present in y_true. Skipping ROC AUC calculation.")
        return 0  # 或者返回一个固定值，比如 0 或 -1
    return roc_auc_score(y_true, y_pred)


# 自监督数据划分函数
def self_supervised_split(data_list, batch_size=128, seed=42):
    """自监督数据划分：1:7:1:1 (有标签训练:无标签:验证:测试)
    Args:
        data_list: 图数据列表
        batch_size: DataLoader批次大小
        seed: 随机种子
    Returns:
        loaders: (labeled_train, unlabeled, val, test) 的DataLoader元组
    """
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)

    # 获取标签
    labels = np.array([data.y.item() for data in data_list])

    # 首先划分测试集 (1/10)
    test_splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    for train_val_idx, test_idx in test_splitter.split(data_list, labels):
        test_data = [data_list[i] for i in test_idx]
        remaining_data = [data_list[i] for i in train_val_idx]
        remaining_labels = labels[train_val_idx]
        break

    # 然后从剩余数据中划分验证集 (1/9)
    val_splitter = StratifiedKFold(n_splits=9, shuffle=True, random_state=seed + 1)
    for train_unlabeled_idx, val_idx in val_splitter.split(remaining_data, remaining_labels):
        val_data = [remaining_data[i] for i in val_idx]
        train_unlabeled_data = [remaining_data[i] for i in train_unlabeled_idx]
        train_unlabeled_labels = remaining_labels[train_unlabeled_idx]
        break

    # 最后从训练+无标签数据中划分有标签训练集 (3/8)
    labeled_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + 2)
    for unlabeled_idx, labeled_idx in labeled_splitter.split(train_unlabeled_data, train_unlabeled_labels):
        labeled_data = [train_unlabeled_data[i] for i in labeled_idx]
        unlabeled_data = [train_unlabeled_data[i] for i in unlabeled_idx]
        break

    # 为无标签数据创建副本并掩码标签
    unlabeled_data_masked = []
    for data in unlabeled_data:
        # 创建数据副本
        masked_data = data.clone()
        # 掩码标签（设置为-1表示无标签）
        masked_data.y = torch.tensor([-1], dtype=torch.float)
        unlabeled_data_masked.append(masked_data)

    # 创建DataLoader
    labeled_loader = DataLoader(labeled_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    unlabeled_loader = DataLoader(unlabeled_data_masked, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

    # 返回原始无标签数据（用于评估伪标签质量）和掩码后的无标签数据
    return labeled_loader, unlabeled_loader, val_loader, test_loader, unlabeled_data


# 5折数据划分函数（保持原有结构，但修改为使用自监督划分）
def stratified_split(data_list, val_ratio=0.1,
                     n_splits=5, batch_size=300, seed=42):
    """改进后的分层五折交叉验证 - 使用自监督划分
    Args:
        data_list: 图数据列表
        val_ratio: 验证集占训练集的比例 (默认10%)
        n_splits: 交叉验证折数 (默认5)
        batch_size: DataLoader批次大小
        seed: 随机种子
    Returns:
        fold_loaders: 各折的DataLoader元组列表 [(labeled_train, unlabeled, val, test, orig_unlabeled), ...]
    """
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)

    # 获取标签并初始化分层划分器
    labels = np.array([data.y.item() for data in data_list])
    fold_loaders = []

    # 主分层KFold划分（用于生成测试集）
    main_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # 生成五折划分
    for fold, (train_val_idx, test_idx) in enumerate(main_kfold.split(data_list, labels)):
        # 使用自监督划分
        fold_data = [data_list[i] for i in train_val_idx]
        labeled_loader, unlabeled_loader, val_loader, _, orig_unlabeled = self_supervised_split(
            fold_data, batch_size=batch_size, seed=seed + fold)

        # 创建测试集DataLoader
        test_data = [data_list[i] for i in test_idx]
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

        fold_loaders.append((labeled_loader, unlabeled_loader, val_loader, test_loader, orig_unlabeled))

    return fold_loaders


def compute_val_loss(model, val_loader, device):
    """计算验证集损失"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in val_loader:
            # 确保数据在正确的设备上
            data = data.to(device)
            loss = model.total_loss(data)
            total_loss += loss.item()
    return total_loss / len(val_loader)


def generate_pseudo_labels(model, unlabeled_loader, device, threshold_high=0.7, threshold_low=0.3):
    """双向伪标签生成：同时利用高置信度正常和异常样本"""
    model.eval()
    pseudo_labeled_data = []
    anomaly_samples = []
    normal_samples = []

    with torch.no_grad():
        for data in unlabeled_loader:
            data = data.to(device)
            pred, _, _, _ = model(data.x, data.edge_index, data.batch)

            # 确保pred是1维张量
            if pred.dim() == 0:
                pred = pred.unsqueeze(0)

            # 应用sigmoid获取概率
            probabilities = torch.sigmoid(pred)

            # 筛选高置信度异常样本（概率 > threshold_high）
            high_anomaly_mask = probabilities > threshold_high
            high_anomaly_indices = torch.where(high_anomaly_mask)[0]

            # 筛选高置信度正常样本（概率 < threshold_low）
            high_normal_mask = probabilities < threshold_low
            high_normal_indices = torch.where(high_normal_mask)[0]

            # 收集异常样本
            for idx in high_anomaly_indices:
                if hasattr(data, '__getitem__'):
                    new_data = data[idx].clone()
                else:
                    new_data = data.clone()
                new_data.y = torch.tensor([1], dtype=torch.long)
                new_data = new_data.cpu()
                anomaly_samples.append(new_data)

            # 收集正常样本
            for idx in high_normal_indices:
                if hasattr(data, '__getitem__'):
                    new_data = data[idx].clone()
                else:
                    new_data = data.clone()
                new_data.y = torch.tensor([0], dtype=torch.long)
                new_data = new_data.cpu()
                normal_samples.append(new_data)

    # 平衡采样：取两类中较小的数量，确保平衡
    min_samples = min(len(anomaly_samples), len(normal_samples))
    if min_samples > 0:
        selected_anomaly = random.sample(anomaly_samples, min_samples)
        selected_normal = random.sample(normal_samples, min_samples)
        pseudo_labeled_data = selected_anomaly + selected_normal
        print(f"平衡采样: 异常样本 {min_samples}, 正常样本 {min_samples}")
    else:
        # 如果没有足够的样本，使用所有可用的
        pseudo_labeled_data = anomaly_samples + normal_samples
        print(f"非平衡采样: 异常样本 {len(anomaly_samples)}, 正常样本 {len(normal_samples)}")

    return pseudo_labeled_data


def self_supervised_train_iteration(model, labeled_loader, unlabeled_loader, optimizer, device,
                                    threshold_high, threshold_low, orig_unlabeled_data):
    """执行一次自监督训练迭代"""
    # 1. 使用当前模型生成伪标签 - 现在使用两个阈值
    pseudo_labeled_data = generate_pseudo_labels(model, unlabeled_loader, device, threshold_high, threshold_low)

    # 2. 随机采样50%的无标签数据用于下一轮
    remaining_unlabeled = []
    for data in orig_unlabeled_data:
        # 确保数据在CPU上进行比较
        data_cpu = data.cpu()
        is_pseudo = False
        for pseudo_data in pseudo_labeled_data:
            pseudo_data_cpu = pseudo_data.cpu()
            if torch.equal(data_cpu.x, pseudo_data_cpu.x) and torch.equal(data_cpu.edge_index,
                                                                          pseudo_data_cpu.edge_index):
                is_pseudo = True
                break
        if not is_pseudo:
            remaining_unlabeled.append(data)

    # 随机采样50%
    if len(remaining_unlabeled) > 0:
        sample_size = max(1, int(0.5 * len(remaining_unlabeled)))
        sampled_unlabeled = random.sample(remaining_unlabeled, sample_size)

        # 创建掩码版本
        sampled_unlabeled_masked = []
        for data in sampled_unlabeled:
            masked_data = data.clone()
            masked_data.y = torch.tensor([-1], dtype=torch.float)
            # 确保数据在CPU上
            sampled_unlabeled_masked.append(masked_data.cpu())

        # 更新无标签DataLoader
        new_unlabeled_loader = DataLoader(sampled_unlabeled_masked, batch_size=unlabeled_loader.batch_size,
                                          shuffle=True, pin_memory=True)
    else:
        new_unlabeled_loader = unlabeled_loader
        sampled_unlabeled = []

    # 3. 合并原始标签数据和伪标签数据
    combined_train_data = []

    # 处理原始标签数据
    for data in labeled_loader.dataset:
        combined_train_data.append(data.cpu())

    # 处理伪标签数据
    for pseudo_data in pseudo_labeled_data:
        combined_train_data.append(pseudo_data.cpu())

    # 创建新的训练DataLoader
    new_labeled_loader = DataLoader(combined_train_data, batch_size=labeled_loader.batch_size,
                                    shuffle=True, pin_memory=True)

    # 4. 使用合并后的数据训练模型
    train_loss = train(model, new_labeled_loader, optimizer, device)

    return train_loss, new_labeled_loader, new_unlabeled_loader, sampled_unlabeled, len(pseudo_labeled_data)


# 5折交叉验证训练与测试 - 自监督版本（优化参数）
# 5折交叉验证训练与测试 - 自监督版本（完整修改版）
def cross_val_self_supervised_train_and_test(data_list, params, device,
                                             num_epochs=300, k_folds=5,
                                             batch_size=300, save_best_model_path="best_model.pth"):
    """自监督K折交叉验证 - 双向伪标签版本"""

    folds = stratified_split(data_list, n_splits=k_folds, batch_size=batch_size)
    test_aucs = []
    best_model = None
    best_test_auc = -float('inf')

    # 自监督训练参数 - 双向阈值
    initial_threshold_high = 0.7  # 异常样本阈值
    initial_threshold_low = 0.4  # 正常样本阈值
    min_threshold_high = 0.5  # 最低异常阈值
    min_threshold_low = 0.2  # 最低正常阈值
    threshold_decay = 0.005  # 衰减速度
    self_training_patience = 10  # 自训练耐心值
    initial_train_epochs = 15  # 初始训练轮数

    start_time = datetime.now()
    print(f"学习率: {params['learning_rate']} -- 隐藏层: {params['hidden_dim']}")

    for fold, (labeled_loader, unlabeled_loader, val_loader, test_loader, orig_unlabeled) in enumerate(folds):
        print(f"Fold {fold + 1}/{k_folds}:", end=" ")

        model = ExplainableGNN(
            feat_dim=data_list[0].x.shape[1],
            hidden_dim=params['hidden_dim'],
            dropout_prob=0.5,
            use_conv_layers=True,
            use_enhanced_fusion=True)
        # model = GCNClassifier(feat_dim=data_list[0].x.shape[1], hidden_dim=params['hidden_dim'])
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=params['learning_rate'],
                                     weight_decay=1e-5)

        # 使用学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10)

        # 初始化早停
        early_stopping = MultiMetricEarlyStopping(
            patience=20,
            min_delta=0.001,
            metrics=['auc', 'loss'],
            mode='any',
            verbose=False
        )

        fold_best_val_auc = -float('inf')
        fold_best_model = None
        fold_best_epoch = 0
        last_lr = params['learning_rate']

        # 自监督训练变量 - 双向阈值
        current_threshold_high = initial_threshold_high
        current_threshold_low = initial_threshold_low
        self_training_iteration = 0
        no_improvement_count = 0
        prev_val_auc = -float('inf')

        # 初始纯监督训练阶段
        print("初始纯监督训练...")
        for epoch in range(initial_train_epochs):
            train_loss = train(model, labeled_loader, optimizer, device)
            val_auc = test(model, val_loader, device)

            if val_auc > fold_best_val_auc:
                fold_best_val_auc = val_auc
                fold_best_model = model.state_dict()
                fold_best_epoch = epoch

        # 自监督训练阶段
        for epoch in range(initial_train_epochs, num_epochs):
            # 自监督训练迭代 - 使用双向阈值
            train_loss, labeled_loader, unlabeled_loader, orig_unlabeled, num_pseudo = self_supervised_train_iteration(
                model, labeled_loader, unlabeled_loader, optimizer, device,
                current_threshold_high, current_threshold_low, orig_unlabeled)

            # 更新阈值 - 双向独立更新
            current_threshold_high = max(min_threshold_high,
                                         initial_threshold_high - threshold_decay * self_training_iteration)
            current_threshold_low = min(min_threshold_low,
                                        initial_threshold_low + threshold_decay * self_training_iteration)
            self_training_iteration += 1

            # 评估
            train_auc = test(model, labeled_loader, device)
            val_auc = test(model, val_loader, device)
            val_loss = compute_val_loss(model, val_loader, device)

            # 学习率调度
            scheduler.step(val_auc)

            # 检查学习率是否变化
            current_lr = optimizer.param_groups[0]['lr']
            if current_lr != last_lr:
                last_lr = current_lr

            # 早停检查
            metrics_dict = {
                'auc': val_auc,
                'loss': -val_loss
            }

            if early_stopping(metrics_dict, model):
                break

            # 自训练停止条件：验证集AUC不再提升
            if val_auc <= prev_val_auc + 0.001:  # 微小提升忽略
                no_improvement_count += 1
            else:
                no_improvement_count = 0
                prev_val_auc = val_auc

            if no_improvement_count >= self_training_patience:
                print(
                    f"自训练停止于迭代 {self_training_iteration}, 最终阈值: 高{current_threshold_high:.3f}/低{current_threshold_low:.3f}")
                break

            # 记录最佳模型
            if val_auc > fold_best_val_auc:
                fold_best_val_auc = val_auc
                fold_best_model = model.state_dict()
                fold_best_epoch = epoch

            # 每5轮输出一次进度
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1:3d} | "
                      f"伪标签: {num_pseudo} | "
                      f"阈值: 高{current_threshold_high:.3f}/低{current_threshold_low:.3f} | "
                      f"Train AUC: {train_auc:.4f} | "
                      f"Val AUC: {val_auc:.4f}")

        # 测试最佳模型
        if fold_best_model is not None:
            model.load_state_dict(fold_best_model)

        test_auc = test(model, test_loader, device)
        test_aucs.append(test_auc)

        print(f"Test AUC: {test_auc:.4f} (最佳epoch: {fold_best_epoch})")

        if test_auc > best_test_auc:
            best_test_auc = test_auc
            best_model = model.state_dict()

    # 输出最终结果
    avg_test_auc = np.mean(test_aucs)
    std_test_auc = np.std(test_aucs)
    end_time = datetime.now()

    # 计算模型运行时间
    running_time = end_time - start_time

    # 格式化运行时间
    hours, remainder = divmod(running_time.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    print(f"=== 最终结果 ===")
    print(f"平均测试AUC: {avg_test_auc:.4f} ± {std_test_auc:.4f}")
    print(f"各折结果: {[f'{auc:.4f}' for auc in test_aucs]}")

    # 保存最佳模型
    if best_model is not None:
        torch.save(best_model, save_best_model_path)
        print(f"最佳模型已保存至: {save_best_model_path}")

    return params, avg_test_auc, std_test_auc


# 主函数
def main():
    # 模型存储路径
    args = ArgParser.parse_args()
    args.dataset = "Actor"
    file_path = args.filePath + args.dataset + ".npz"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(
        f"{args.dataset} --lr -{args.lr}  --batch_size -{args.batch_size}  --hidden_dim -{args.hidden_dim} --dropout -{args.dropout}")
    now = datetime.now().strftime("%m-%d-%H-%M")

    edges_df, features_df = Data_Query(file_path)
    data_list = Communtiy_Search_randomwalk(edges_df, features_df)
    count = sum(1 for data in data_list if data.y.item() == 1)
    print(f"data.y 等于 1 的子图个数: {count}")
    model_path = args.bestModelPath + f"{args.dataset}_best_model_{now}.pth"

    # 定义超参数网格[1e-4,3e-4,1e-3,3e-3,1e-2]
    param_grid = {
        'learning_rate': [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        'hidden_dim': [512],
    }

    # 初始化一个列表来存储每次循环的结果
    results = []
    for params in ParameterGrid(param_grid):
        # 使用自监督训练版本
        params, avg_test_auc, std_test_auc = cross_val_self_supervised_train_and_test(
            data_list, params, device, batch_size=args.batch_size, save_best_model_path=model_path)
        results.append((params, avg_test_auc, std_test_auc))

    for id, (params, avg_test_auc, std_test_auc) in enumerate(results):
        print(f"实验 {id + 1}:")
        print(f"学习率: {params['learning_rate']} -- 隐藏层: {params['hidden_dim']}")
        print(f"平均测试 AUC: {avg_test_auc * 100:.2f}±{std_test_auc * 100:.2f}")
        print("-" * 50)


if __name__ == "__main__":
    main()
