import os
import sys
import torch
import random
import argparse
import numpy as np
from jsonlines import jsonlines

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

def detect_aigc_non_chinese(text):

    model_save_path = 'params/params_poger_full_data.pt'
    model='poger_mix_wo_context'
    batch_size=64
    seed=2024
    device ='cpu'
    data_dir='../get_feature'
    pretrain_model='roberta-base'
    max_len=512
    k=10
    lr=1e-4
    n_classes=8
    n_feat=7
    data = [{
        "text": text,
    }]

    # 3. 写入到 JSON Lines 文件
    jsonl_file = '../get_feature/detect_feature.jsonl'
    with jsonlines.open(jsonl_file, mode='w') as writer:
        writer.write(data)
    train_dataloader, test_dataloader = get_dataloader(model, data_dir, pretrain_model,batch_size, max_len, k)
    epoch=20
    trainer = POGERMixWOContextTrainer(device, pretrain_model, test_dataloader, epoch, lr, model_save_path, n_classes, n_feat)


        # 定义标签映射关系
    labels = {
        'human': 0,
        'gpt2': 1,
        'gpt-j': 2,
        'Llama': 3,
        'vicuna': 4,
        'alpaca': 5,
        'gpt-3.5': 6,
        'gpt-4': 7
    }
    # 加载测试模型的参数
    # model = torch.load('path_to_your_model.pth', map_location=torch.device('cpu'))
    trainer.model.load_state_dict(torch.load('params/params_poger_data.pt', map_location=torch.device('cpu')))
    # 进行测试并输出结果
    results = trainer.detect()
    #
    reverse_labels = {v: k for k, v in labels.items()}


    if results in reverse_labels:
        if results == labels['human']:
            textfalse = 0
            result_label='human'
        else:
            textfalse = 1
            result_label = reverse_labels[results]


    return textfalse,result_label


