import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run program")

    parser.add_argument('--device', nargs='?', default='cuda:0', help="GPU名称")
    parser.add_argument('--bestModelPath', nargs='?', default='model_file/', help="模型存储文件路径")
    parser.add_argument('--filePath', nargs='?', default='./DataSet/', help="数据集路径")
    parser.add_argument('--output', nargs='?', default='Output_', help="预处理完的数据的输出路径")
    parser.add_argument('--dataset', type=str, default='Elliptic',
                        choices=['Elliptic', 'DGraph', 'Actor', 'Arxiv', 'Minesweeper', 'Roman'],
                        help="The dataset to use for training and evaluation.")
    parser.add_argument('--com_ser', type=str, default='Elliptic',
                        choices=['BFS', 'RRW', 'PRRW',],
                        help="The dataset to use for training and evaluation.")
    parser.add_argument('--epochs', type=int, default=300, help="Number of training epochs.")
    parser.add_argument('--hidden_dim', type=int, default=512, help="Number of hidden_dim.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument('--dropout', type=float, default=0.5, help="dropout")
    parser.add_argument('--k_folds', type=int, default=5, help="Number of folds for cross-validation.")
    parser.add_argument('--batch_size', type=int, default=300, help="Batch size for training and validation.")
    parser.add_argument('--save_best_model', action='store_true', help="Flag to save the best model during training.")
    return parser.parse_args()
