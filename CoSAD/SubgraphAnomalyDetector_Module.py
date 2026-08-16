import random
import torch
import ArgParser
import numpy as np
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, ParameterGrid, KFold
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
from PathAwareGNN import PathAwareGNN
from data_process.Community_Search import Communtiy_Search_randomwalk
from data_process.Community_Search_via_DPA_ARWR import run_DAP_ARWR_batch
from data_process.Data_Enhance import UltimateTripleStreamAugmentor
from data_process.Data_Query import Data_Query
from ExplainHerterGNN_Moudle import ExplainableGNN, GCNClassifier
from tqdm import tqdm
from data_process.Eearly_Stopping import RobustEarlyStopping
from data_process.RWR_HACS import dap_arwr_first_stage_gpu
import os
from itertools import product
from sklearn.model_selection import ParameterGrid  # 假设您已经导入

os.environ["OMP_NUM_THREADS"] = "6"


def train(model, loader, optimizer, device, augmentor=None):
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device, non_blocking=True)
        optimizer.zero_grad()

        # 🚀 自动判定并执行一致性正则化
        has_gray_samples = (data.y == -1).any().item()
        if has_gray_samples and augmentor is not None:
            loss = model.total_loss(data, augmentor=augmentor)
        else:
            loss = model.total_loss(data)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def test(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            # 1. 究极防爆：无限解包取出真实的 Tensor
            out = model.predict(data)
            while isinstance(out, (tuple, list)):
                out = out[0]
            pred = out

            # 2. 维度安全保护
            if pred.dim() == 0:
                pred = pred.unsqueeze(0)

            # 3. 过滤掉灰区数据（测试和验证集决不包含 -1 的杂质！）
            valid_mask = (data.y != -1)
            if valid_mask.sum() > 0:
                y_true.append(data.y[valid_mask].cpu().numpy())
                y_pred.append(pred[valid_mask].cpu().numpy())

    if len(y_true) == 0: return 0.0

    y_true = np.concatenate(y_true).ravel()
    y_pred = np.concatenate(y_pred)

    if y_pred.ndim == 2:
        y_pred = y_pred[:, 1] if y_pred.shape[1] >= 2 else y_pred.ravel()

    y_true_binary = (y_true > 0).astype(int)

    if len(np.unique(y_true_binary)) < 2:
        return 0.0

    return roc_auc_score(y_true_binary, y_pred)


# 5折数据划分函数
def stratified_split(data_list, val_ratio=0.2,
                     n_splits=5, batch_size=128, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    labels = np.array([data.y.item() for data in data_list])
    fold_loaders = []

    # 🚀 安全机制：检查每个类别的样本数，如果最少类别的数量 < n_splits，退化为普通的 KFold
    unique_classes, counts = np.unique(labels, return_counts=True)
    min_count = np.min(counts)

    if min_count < n_splits:
        tqdm.write(
            f"⚠️ 警告: 类别 {unique_classes[np.argmin(counts)]} 只有 {min_count} 个样本，无法进行严格的分层划分。自动切换为普通 KFold。")
        main_kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        inner_splitter_class = KFold
    else:
        main_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        inner_splitter_class = StratifiedKFold

    for fold, (train_val_idx, test_idx) in enumerate(main_kfold.split(data_list, labels)):
        train_labels = labels[train_val_idx]

        # 内部验证集划分
        inner_splitter = inner_splitter_class(n_splits=int(1 / val_ratio), shuffle=True, random_state=seed + fold)

        for train_idx, val_idx in inner_splitter.split(train_val_idx, train_labels):
            real_train_idx = train_val_idx[train_idx]
            real_val_idx = train_val_idx[val_idx]
            break

        train_data = [data_list[i] for i in real_train_idx]
        val_data = [data_list[i] for i in real_val_idx]
        test_data = [data_list[i] for i in test_idx]

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=False)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=False)

        fold_loaders.append((train_loader, val_loader, test_loader))

    return fold_loaders


def compute_val_loss(model, val_loader, device):
    """计算验证集损失"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            loss = model.total_loss(data)
            total_loss += loss.item()
    return total_loss / len(val_loader)


#
# def cross_val_train_and_test(data_list, params, device, num_epochs=300, k_folds=5, batch_size=300,
#                              save_best_model_path="best_model.pth"):
#     folds = stratified_split(data_list, n_splits=k_folds, batch_size=batch_size)
#     test_aucs = []
#     best_overall_model = None
#     best_overall_test_auc = -float('inf')
#
#     warmup_epochs = 15
#     patience_val = 25
#     total_steps = k_folds * num_epochs
#
#     global_pbar = tqdm(total=total_steps, desc="🚀 训练进度", ncols=180, unit="step",
#                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, {postfix}]")
#
#     # 🚀 实例化全能的增强器
#     augmentor = UltimateTripleStreamAugmentor(
#         feature_noise=0.02, edge_drop_rate=0.15, path_jitter_rate=0.2
#     )
#
#     for fold in range(k_folds):
#         train_loader, val_loader, test_loader = folds[fold]
#
#         model = PathAwareGNN(feat_dim=data_list[0].x.shape[1], hidden_dim=params['hidden_dim'], dropout_prob=0.6).to(
#             device)
#         optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=5e-4)
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
#
#         early_stopping = RobustEarlyStopping(patience=patience_val, min_delta=0.001, smoothing=10,
#                                              restore_best_weights=True, verbose=False)
#
#         fold_best_val_auc = -float('inf')
#         fold_best_epoch = 0
#         stop_training = False
#
#         for epoch in range(num_epochs):
#             # 🚀 传入 augmentor，激活一致性正则化
#             train_loss = train(model, train_loader, optimizer, device, augmentor)
#             val_auc = test(model, val_loader, device)
#
#             scheduler.step(val_auc)
#             current_lr = optimizer.param_groups[0]['lr']
#
#             if val_auc > fold_best_val_auc:
#                 fold_best_val_auc = val_auc
#                 fold_best_epoch = epoch + 1
#
#             if epoch >= warmup_epochs:
#                 if early_stopping(val_auc, model):
#                     stop_training = True
#
#             global_pbar.set_postfix({'折数': f'{fold + 1}/{k_folds}', 'Epoch': f'{epoch + 1}/{num_epochs}',
#                                      'Loss': f'{train_loss:.4f}', 'ValAUC': f'{val_auc:.4f}', 'LR': f'{current_lr:.2e}',
#                                      '最佳Epoch': f'{fold_best_epoch}', '当前最佳ValAUC': f'{fold_best_val_auc:.4f}'})
#             global_pbar.update(1)
#
#             if stop_training:
#                 global_pbar.update(num_epochs - epoch - 1)
#                 break
#
#         test_auc = test(model, test_loader, device)
#         test_aucs.append(test_auc)
#
#         if test_auc > best_overall_test_auc:
#             best_overall_test_auc = test_auc
#             best_overall_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
#
#         tqdm.write(f"\n✅ Fold {fold + 1} 完成 | Test AUC: {test_auc:.4f} | 最佳Epoch: {fold_best_epoch}")
#
#     global_pbar.close()
#
#     tqdm.write(f"\n{'=' * 80}")
#     tqdm.write(f"📈 最终统计 (半监督一致性正则化 SSL Mode)")
#     tqdm.write(f"平均测试 AUC: {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")
#     tqdm.write(f"{'=' * 80}")
#
#     if best_overall_model is not None:
#         torch.save(best_overall_model, save_best_model_path)
#     return params, np.mean(test_aucs), np.std(test_aucs)
#
# def main():
#     # 模型存储路径
#     torch.autograd.set_detect_anomaly(True)
#     torch.backends.cudnn.benchmark = True
#     torch.backends.cudnn.deterministic = False
#     num_samples = 0
#     args = ArgParser.parse_args()
#     args.dataset = "DGraph"
#     args.com_ser = "BFS"
#     file_path = args.filePath + args.dataset + ".npz"
#     device = torch.device(args.device if torch.cuda.is_available() else "cpu")
#
#     # 初始化进度条输出格式
#     tqdm.write(f"\n{'=' * 80}")
#     tqdm.write(f"🚀 开始训练 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     tqdm.write(f"{'=' * 80}")
#     tqdm.write(f"数据集: {args.dataset}")
#     tqdm.write(f"运行设备: {device}")
#     tqdm.write(f"社区搜索方法: {args.com_ser}")
#     tqdm.write(
#         f"超参数: LR={args.lr} | Batch Size={args.batch_size} | Hidden Dim={args.hidden_dim} | Dropout={args.dropout}")
#     tqdm.write(f"{'=' * 80}\n")
#
#     now = datetime.now().strftime("%m-%d-%H-%M")
#
#     edges_df, features_df = Data_Query(file_path)
#     if args.dataset == "Actor":
#         # num_samples = int(len(features_df) * 0.1)
#         num_samples = 500
#     elif args.dataset == "Arxiv":
#         # num_samples = int(len(features_df) * 0.005)
#         num_samples = 800
#     elif args.dataset == "Minesweeper":
#         # num_samples = int(len(features_df) * 0.1)
#         num_samples = 1000
#     elif args.dataset == "Roman":
#         # num_samples = int(len(features_df) * 0.06)
#         num_samples = 500
#     elif args.dataset == "Elliptic":
#         # num_samples = int(len(features_df) * 0.003)
#         num_samples = 1000
#     elif args.dataset == "DGraph":
#         # num_samples = int(len(features_df) * 0.0002)
#         num_samples = 1000
#     else:
#         num_samples = 500
#     # data_list = Communtiy_Search_randomwalk(edges_df, features_df, num_samples, com_ser=args.com_ser)
#     data_list = dap_arwr_first_stage_gpu(edges_df, features_df, num_samples)
#     # save_path = "./" + args.dataset + ".pt"
#     # # 保存data_list（会完整保存所有Data对象的属性）
#     # torch.save(data_list, save_path)
#     # data_list = torch.load(save_path, weights_only=False)
#     # print(f"Data_list已保存")
#
#     # 统计数据信息
#     # total_nodes = sum(data.x.shape[0] for data in data_list)
#     # total_graphs = len(data_list)
#     # pos_graphs = sum(1 for data in data_list if data.y.item() == 1)
#     # neg_graphs = total_graphs - pos_graphs
#     #
#     # # 输出数据统计信息（通过tqdm.write避免打断进度条）
#     # tqdm.write(f"\n📊 数据统计:")
#     # tqdm.write(f"- 子图总数: {total_graphs}")
#     # tqdm.write(f"- 正样本子图: {pos_graphs} ({pos_graphs / total_graphs * 100:.1f}%)")
#     # tqdm.write(f"- 负样本子图: {neg_graphs} ({neg_graphs / total_graphs * 100:.1f}%)")
#     # tqdm.write(f"- 总节点数: {total_nodes}")
#     # tqdm.write(f"- 平均子图大小: {total_nodes / total_graphs:.1f} 节点/子图\n")
#     # ================= 🚀 替换主函数中的统计打印逻辑 =================
#
#     total_nodes = sum(data.x.shape[0] for data in data_list)
#     total_graphs = len(data_list)
#
#     pos_graphs = sum(1 for data in data_list if data.y.item() == 1)
#     neg_graphs = total_graphs - pos_graphs
#
#     # 输出基于费希尔检验的数据统计信息
#     tqdm.write(f"\n📊 费希尔检验高置信数据统计:")
#     tqdm.write(f"- 子图总数: {total_graphs}")
#     tqdm.write(f"- 【0】负样本 (正常社区, p > 0.50): {neg_graphs} ({neg_graphs / total_graphs * 100:.1f}%)")
#     tqdm.write(f"- 【1】正样本 (异常社区, p < 0.01): {pos_graphs} ({pos_graphs / total_graphs * 100:.1f}%)")
#     tqdm.write(f"- 总节点数: {total_nodes}")
#     tqdm.write(f"- 平均子图大小: {total_nodes / total_graphs:.1f} 节点/子图\n")
#     # return 1
#
#     model_path = args.bestModelPath + f"{args.dataset}_best_model_{now}.pth"
#
#     # 定义超参数网格[1e-4,3e-4,1e-3,3e-3,1e-2]
#     param_grid = {
#         'learning_rate': [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
#         'hidden_dim': [32, 64, 128, 256, 512],
#     }
#     # param_grid = {
#     #     'learning_rate': [3e-4],
#     #     'hidden_dim': [64],
#     # }
#
#     # 初始化一个列表来存储每次循环的结果
#     results = []
#     for params in ParameterGrid(param_grid):
#         params, avg_test_auc, std_test_auc = cross_val_train_and_test(data_list, params, device,
#                                                                       batch_size=args.batch_size,
#                                                                       save_best_model_path=model_path)
#         results.append((params, avg_test_auc, std_test_auc))
#
#     for id, (params, avg_test_auc, std_test_auc) in enumerate(results):
#         tqdm.write(f"\n实验 {id + 1}:")
#         tqdm.write(f"学习率: {params['learning_rate']} -- 隐藏层: {params['hidden_dim']}")
#         tqdm.write(f"平均测试 AUC: {avg_test_auc * 100:.2f}±{std_test_auc * 100:.2f}")
#         tqdm.write("-" * 50)
#
#
# if __name__ == "__main__":
#     main()
def cross_val_train_and_test(data_list, params, device, num_epochs=300, k_folds=5, batch_size=300,
                             save_best_model_path="best_model.pth"):
    folds = stratified_split(data_list, n_splits=k_folds, batch_size=batch_size)
    test_aucs = []
    best_overall_model = None
    best_overall_test_auc = -float('inf')

    warmup_epochs = 15
    patience_val = 25

    start_time = datetime.now()

    # 🚀 强力打印实验参数
    print("\n" + "★" * 80)
    print(f"★ 启动新实验 | 学习率 LR: {params['learning_rate']} | 隐藏层 Hidden: {params['hidden_dim']}")
    print("★" * 80)

    global_pbar = tqdm(total=k_folds * num_epochs, desc="🚀 训练进度", ncols=180, unit="step",
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, {postfix}]")

    augmentor = UltimateTripleStreamAugmentor(feature_noise=0.02, edge_drop_rate=0.15, path_jitter_rate=0.2)

    for fold in range(k_folds):
        train_loader, val_loader, test_loader = folds[fold]

        model = PathAwareGNN(feat_dim=data_list[0].x.shape[1], hidden_dim=params['hidden_dim'], dropout_prob=0.6).to(
            device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
        early_stopping = RobustEarlyStopping(patience=patience_val, min_delta=0.001, smoothing=10,
                                             restore_best_weights=True, verbose=False)

        fold_best_val_auc = -float('inf')
        fold_best_epoch = 0
        stop_training = False

        for epoch in range(num_epochs):
            train_loss = train(model, train_loader, optimizer, device, augmentor)
            val_auc = test(model, val_loader, device)

            scheduler.step(val_auc)
            current_lr = optimizer.param_groups[0]['lr']

            if val_auc > fold_best_val_auc:
                fold_best_val_auc = val_auc
                fold_best_epoch = epoch + 1

            if epoch >= warmup_epochs:
                if early_stopping(val_auc, model):
                    stop_training = True

            global_pbar.set_postfix({'折数': f'{fold + 1}/{k_folds}', 'Epoch': f'{epoch + 1}/{num_epochs}',
                                     'Loss': f'{train_loss:.4f}', 'ValAUC': f'{val_auc:.4f}', 'LR': f'{current_lr:.2e}',
                                     '最佳Epoch': f'{fold_best_epoch}', '最佳ValAUC': f'{fold_best_val_auc:.4f}'})
            global_pbar.update(1)

            if stop_training:
                global_pbar.update(num_epochs - epoch - 1)
                break

        test_auc = test(model, test_loader, device)
        test_aucs.append(test_auc)

        if test_auc > best_overall_test_auc:
            best_overall_test_auc = test_auc
            best_overall_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        tqdm.write(f"\n✅ Fold {fold + 1} 完成 | Test AUC: {test_auc:.4f} | 最佳Epoch: {fold_best_epoch}")

    global_pbar.close()

    # 🚀 强力汇总当前实验的最终参数与结果
    res_str = f"🏁 实验结束 -> LR: {params['learning_rate']} | Hidden: {params['hidden_dim']} | 最终AUC: {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}"
    print("\n" + "=" * len(res_str))
    print(res_str)
    print("=" * len(res_str) + "\n")

    if best_overall_model is not None:
        torch.save(best_overall_model, save_best_model_path)
    return params, np.mean(test_aucs), np.std(test_aucs)


def main():
    torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    args = ArgParser.parse_args()
    args.dataset = "DGraph"
    args.com_ser = "BFS"
    file_path = args.filePath + args.dataset + ".npz"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    now = datetime.now().strftime("%m-%d-%H-%M")

    print(f"\n{'=' * 80}")
    print(f"🚀 开始自动化敏感性测试流水线 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据集: {args.dataset} | 运行设备: {device}")
    print(f"{'=' * 80}\n")

    edges_df, features_df = Data_Query(file_path)

    # =====================================================================
    # 🚀🚀🚀 [核心控制台]：在这里修改您想跑的实验阶段 🚀🚀🚀
    # STAGE 1: 测图拓扑规模 (k_min, k_max)
    # STAGE 2: 测自适应过滤阈值 (q_quantile)
    # STAGE 3: 测训练基础参数 (num_samples, lr, hidden_dim)
    # STAGE 4: 测RWR参数 (restart_prob, max_rwr_iter)          <-- 新增
    # =====================================================================
    EXPERIMENT_STAGE = 1  # <---- 每次测试只需修改这个数字！(1, 2, 3, 4)

    # ========== 全局最优基准参数 (用于控制变量法) ==========
    BEST_NUM_SAMPLES = 2500
    BEST_K_MIN = 5
    BEST_K_MAX = 50
    BEST_Q = 0.75
    BEST_LR = 0.001
    BEST_HIDDEN = 256
    BEST_RESTART_PROB = 0.15  # 新增 RWR 基准值
    BEST_MAX_RWR_ITER = 100  # 新增 RWR 基准值
    # ======================================================

    results = []  # 记录所有结果

    # ---------------------------------------------------------------------
    # 🌟 阶段一：图拓扑规模测试 (k_min 与 k_max)
    # ---------------------------------------------------------------------
    if EXPERIMENT_STAGE == 1:
        print("\n" + "🔥" * 40)
        print("执行阶段 1：图社区规模 (k_min, k_max) 敏感性测试")
        print("🔥" * 40)

        k_min_list = [5, 10, 15, 20]
        k_max_list = [30, 50, 70, 90]

        for k_min, k_max in product(k_min_list, k_max_list):
            if k_min >= k_max: continue
            print(f"\n👉 [当前参数] k_min: {k_min}, k_max: {k_max}")

            data_list = dap_arwr_first_stage_gpu(
                edges_df, features_df, num_samples=BEST_NUM_SAMPLES,
                min_community_size=k_min, max_community_size=k_max, q_quantile=BEST_Q,
                restart_prob=BEST_RESTART_PROB, max_rwr_iter=BEST_MAX_RWR_ITER  # 固定 RWR 参数
            )

            avg_nodes = np.mean([d.x.shape[0] for d in data_list]) if data_list else 0
            avg_quality = np.mean([d.community_quality.item() for d in data_list]) if data_list else 0

            params = {'learning_rate': BEST_LR, 'hidden_dim': BEST_HIDDEN}
            model_path = args.bestModelPath + f"stage1_{now}_kmin{k_min}_kmax{k_max}.pth"
            _, avg_test_auc, std_test_auc = cross_val_train_and_test(
                data_list, params, device, batch_size=args.batch_size, save_best_model_path=model_path
            )

            results.append({
                'k_min': k_min, 'k_max': k_max, '|C|_avg': avg_nodes, 'S_avg': avg_quality,
                'AUC': avg_test_auc * 100, 'STD': std_test_auc * 100
            })

    # ---------------------------------------------------------------------
    # 🌟 阶段二：自适应过滤阈值测试 (q_quantile)
    # ---------------------------------------------------------------------
    elif EXPERIMENT_STAGE == 2:
        print("\n" + "🔥" * 40)
        print("执行阶段 2：自适应过滤分位数 (q_quantile) 敏感性测试")
        print("🔥" * 40)

        q_list = [0.5, 0.6, 0.75, 0.85, 0.9]

        for q in q_list:
            print(f"\n👉 [当前参数] q_quantile: {q}")
            data_list = dap_arwr_first_stage_gpu(
                edges_df, features_df, num_samples=BEST_NUM_SAMPLES,
                min_community_size=BEST_K_MIN, max_community_size=BEST_K_MAX, q_quantile=q,
                restart_prob=BEST_RESTART_PROB, max_rwr_iter=BEST_MAX_RWR_ITER
            )

            avg_nodes = np.mean([d.x.shape[0] for d in data_list]) if data_list else 0
            avg_quality = np.mean([d.community_quality.item() for d in data_list]) if data_list else 0

            params = {'learning_rate': BEST_LR, 'hidden_dim': BEST_HIDDEN}
            model_path = args.bestModelPath + f"stage2_{now}_q{q}.pth"
            _, avg_test_auc, std_test_auc = cross_val_train_and_test(
                data_list, params, device, batch_size=args.batch_size, save_best_model_path=model_path
            )

            results.append({
                'q': q, '|C|_avg': avg_nodes, 'S_avg': avg_quality,
                'AUC': avg_test_auc * 100, 'STD': std_test_auc * 100
            })

    # ---------------------------------------------------------------------
    # 🌟 阶段三：训练基础参数测试 (原版的三维网格搜索)
    # ---------------------------------------------------------------------
    elif EXPERIMENT_STAGE == 3:
        print("\n" + "🔥" * 40)
        print("执行阶段 3：样本量 x 学习率 x 隐藏维度 三维网格搜索")
        print("🔥" * 40)

        num_samples_list = [2000, 2500, 3000, 3500]
        param_grid = {
            'learning_rate': [0.0001, 0.0003, 0.001, 0.003],
            'hidden_dim': [32, 64, 128, 256],
        }

        for num_samples in num_samples_list:
            print(f"\n👉 开始构建图数据, 样本数: {num_samples}")
            data_list = dap_arwr_first_stage_gpu(
                edges_df, features_df, num_samples=num_samples,
                min_community_size=BEST_K_MIN, max_community_size=BEST_K_MAX, q_quantile=BEST_Q,
                restart_prob=BEST_RESTART_PROB, max_rwr_iter=BEST_MAX_RWR_ITER
            )

            avg_nodes = np.mean([d.x.shape[0] for d in data_list]) if data_list else 0
            avg_quality = np.mean([d.community_quality.item() for d in data_list]) if data_list else 0

            for params in ParameterGrid(param_grid):
                model_path = args.bestModelPath + f"stage3_{now}_n{num_samples}_lr{params['learning_rate']}_h{params['hidden_dim']}.pth"
                _, avg_test_auc, std_test_auc = cross_val_train_and_test(
                    data_list, params, device, batch_size=args.batch_size, save_best_model_path=model_path
                )

                results.append({
                    'Num_Samples': num_samples, 'LR': params['learning_rate'], 'Hidden': params['hidden_dim'],
                    '|C|_avg': avg_nodes, 'S_avg': avg_quality,
                    'AUC': avg_test_auc * 100, 'STD': std_test_auc * 100
                })

    # ---------------------------------------------------------------------
    # 🌟 阶段四：RWR 超参数测试 (restart_prob, max_rwr_iter)   <-- 全新阶段
    # ---------------------------------------------------------------------
    elif EXPERIMENT_STAGE == 4:
        print("\n" + "🔥" * 40)
        print("执行阶段 4：RWR 参数 (重启概率, 最大迭代次数) 敏感性测试")
        print("🔥" * 40)

        restart_probs = [0.05, 0.1, 0.15, 0.2, 0.3]  # 重启概率候选
        max_iters = [50, 100, 150, 200]  # 最大迭代次数候选

        for rp, mi in product(restart_probs, max_iters):
            print(f"\n👉 [当前参数] restart_prob: {rp}, max_rwr_iter: {mi}")

            # 用最优社区搜索参数 + 当前 RWR 参数构造数据
            data_list = dap_arwr_first_stage_gpu(
                edges_df, features_df, num_samples=BEST_NUM_SAMPLES,
                min_community_size=BEST_K_MIN, max_community_size=BEST_K_MAX, q_quantile=BEST_Q,
                restart_prob=rp, max_rwr_iter=mi, rwr_tol=1e-6  # tol 固定默认
            )

            avg_nodes = np.mean([d.x.shape[0] for d in data_list]) if data_list else 0
            avg_quality = np.mean([d.community_quality.item() for d in data_list]) if data_list else 0

            # 固定最优训练参数
            params = {'learning_rate': BEST_LR, 'hidden_dim': BEST_HIDDEN}
            model_path = args.bestModelPath + f"stage4_{now}_rp{rp}_mi{mi}.pth"
            _, avg_test_auc, std_test_auc = cross_val_train_and_test(
                data_list, params, device, batch_size=args.batch_size, save_best_model_path=model_path
            )

            results.append({
                'restart_prob': rp, 'max_iter': mi,
                '|C|_avg': avg_nodes, 'S_avg': avg_quality,
                'AUC': avg_test_auc * 100, 'STD': std_test_auc * 100
            })

    # =====================================================================
    # 🏆 统一的结果输出面板
    # =====================================================================
    print("\n" + "🏆" * 50)
    print(f"🏆 实验阶段 {EXPERIMENT_STAGE} 测试结果汇总 (按 AUC 降序排列) 🏆")
    print("🏆" * 50)

    results.sort(key=lambda x: x['AUC'], reverse=True)

    for i, res in enumerate(results):
        # 动态生成参数部分，跳过固定的监控指标列
        metric_keys = {'AUC', 'STD', '|C|_avg', 'S_avg'}
        param_str = " | ".join([f"{k}: {v}" for k, v in res.items() if k not in metric_keys])
        metric_str = f"|C|_avg: {res['|C|_avg']:5.1f} | S_avg: {res['S_avg']:.4f} | Test AUC: {res['AUC']:.2f}% ± {res['STD']:.2f}%"
        print(f"Rank {i + 1:2d} | {param_str} | {metric_str}")

    print("🏆" * 50 + "\n")


if __name__ == "__main__":
    import time

    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"\n⏱️  main() 总耗时: {elapsed:.2f} 秒 ({elapsed / 60:.2f} 分钟)")
