import os
import sys
import torch
import random
import argparse
import numpy as np

from dataloader import get_dataloader
from model.poger import Trainer as POGERTrainer
from model.poger_mix import Trainer as POGERMixTrainer
from model.poger_wo_context import Trainer as POGERWOContextTrainer
from model.poger_mix_wo_context import Trainer as POGERMixWOContextTrainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--model')
    parser.add_argument('--n-classes', type=int, default=8)
    parser.add_argument('--n-feat', type=int, default=7)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--data-name', type=str)
    parser.add_argument('--pretrain-model', default='roberta-base')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-len', type=int, default=512)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--model-save-dir', default='./params')
    parser.add_argument('--test', type=str)
    return parser.parse_args()

def set_seed(seed):
    """
    设置随机种子，以确保结果的可重复性和稳定性。

    Args:
    seed (int): 随机种子值。

    Returns:
    None
    """

    # 设置 Python 内置的随机数生成器的种子
    random.seed(seed)

    # 设置 Python 环境变量中的哈希种子，用于控制哈希对象的随机化顺序
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 设置 NumPy 库的随机数生成器的种子
    np.random.seed(seed)

    # 设置 PyTorch 的随机数生成器的种子
    torch.manual_seed(seed)

    # 设置 PyTorch CUDA 随机数生成器的种子（如果使用GPU加速的话）
    torch.cuda.manual_seed(seed)

    # 禁用 PyTorch 的内置 cuDNN 自动调整功能，确保每次结果一致
    torch.backends.cudnn.benchmark = False

    # 启用 PyTorch 的 determinstic 模式，确保每次结果一致
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False

def main(args):
    set_seed(args.seed)
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'

    if not os.path.isdir(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    model_save_path = os.path.join(args.model_save_dir, 'params_%s_%s.pt' % (args.model, args.data_name))

    train_dataloader, test_dataloader = get_dataloader(args.model, args.data_dir, args.pretrain_model, args.batch_size, args.max_len, args.k)

    if args.model == 'poger':
        trainer = POGERTrainer(device, args.pretrain_model, train_dataloader, test_dataloader, args.epoch, args.lr, model_save_path, args.n_classes, args.n_feat, args.k)
    elif args.model == 'poger_wo_context':
        trainer = POGERWOContextTrainer(device, args.pretrain_model, train_dataloader, test_dataloader, args.epoch, args.lr, model_save_path, args.n_classes, args.n_feat, args.k)
    elif args.model == 'poger_mix':
        trainer = POGERMixTrainer(device, args.pretrain_model, train_dataloader, test_dataloader, args.epoch, args.lr, model_save_path, args.n_classes, args.n_feat)
    elif args.model == 'poger_mix_wo_context':
        trainer = POGERMixWOContextTrainer(device, args.pretrain_model, train_dataloader, test_dataloader, args.epoch, args.lr, model_save_path, args.n_classes, args.n_feat)
    else:
        print('There is no model called "%s"' % args.model)
        return -1

    if args.test is None:
        trainer.train()
    else:
        # 加载测试模型的参数
        trainer.model.load_state_dict(torch.load(args.test))
        # 进行测试并输出结果
        results = trainer.test(test_dataloader)
        # print(results)
        print('acc = %.4f, f1 = %.4f, auc_ovo = %.4f' % (results['accuracy'], results['f1'], results['auc_ovo']))
        print('P/R per class: ', end='')
        for i in range(args.n_classes):
            print('%.2f/%.2f ' % (results['precision'][i] * 100, results['recall'][i] * 100), end='')
        print()
        print('F1 per class: ', end='')
        for i in range(args.n_classes):
            print('%.2f ' % (results['detail_f1'][i] * 100), end='')
        print()

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
