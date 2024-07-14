import warnings

import cv2
import jsonlines
from PIL.Image import Image
from pytesseract import pytesseract

warnings.filterwarnings("ignore")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.vit import interpolate_pos_embed
from transformers import BertTokenizerFast, BertTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer

import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import logging
from types import MethodType
from tools.env import init_dist
from tqdm import tqdm

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from models import box_ops
from tools.multilabel_metrics import AveragePrecisionMeter, get_multi_label

from models.HAMMER import HAMMER

def setlogger(log_file):
    filehandler = logging.FileHandler(log_file)
    streamhandler = logging.StreamHandler()

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    def epochInfo(self, set, idx, loss, acc):
        self.info('{set}-{idx:d} epoch | loss:{loss:.8f} | auc:{acc:.4f}%'.format(
            set=set,
            idx=idx,
            loss=loss,
            acc=acc
        ))

    logger.epochInfo = MethodType(epochInfo, logger)

    return logger


def text_input_adjust(text_input, fake_word_pos, device):
    # input_ids adaptation
    input_ids_remove_SEP = [x[:-1] for x in text_input.input_ids]
    maxlen = max([len(x) for x in text_input.input_ids])-1
    input_ids_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in input_ids_remove_SEP] # only remove SEP as HAMMER is conducted with text with CLS
    text_input.input_ids = torch.LongTensor(input_ids_remove_SEP_pad).to(device) 

    # attention_mask adaptation
    attention_mask_remove_SEP = [x[:-1] for x in text_input.attention_mask]
    attention_mask_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in attention_mask_remove_SEP]
    text_input.attention_mask = torch.LongTensor(attention_mask_remove_SEP_pad).to(device)

    # fake_token_pos adaptation
    fake_token_pos_batch = []
    subword_idx_rm_CLSSEP_batch = []
    for i in range(len(fake_word_pos)):
        fake_token_pos = []

        fake_word_pos_decimal = np.where(fake_word_pos[i].numpy() == 1)[0].tolist() # transfer fake_word_pos into numbers

        subword_idx = text_input.word_ids(i)
        subword_idx_rm_CLSSEP = subword_idx[1:-1]
        subword_idx_rm_CLSSEP_array = np.array(subword_idx_rm_CLSSEP) # get the sub-word position (token position)

        subword_idx_rm_CLSSEP_batch.append(subword_idx_rm_CLSSEP_array)
        
        # transfer the fake word position into fake token position
        for i in fake_word_pos_decimal: 
            fake_token_pos.extend(np.where(subword_idx_rm_CLSSEP_array == i)[0].tolist())
        fake_token_pos_batch.append(fake_token_pos)

    return text_input, fake_token_pos_batch, subword_idx_rm_CLSSEP_batch


@torch.no_grad()
def evaluation(args, model, data_loader, tokenizer, device, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'


    print_freq = 200

    y_true, y_pred, IOU_pred, IOU_50, IOU_75, IOU_95 = [], [], [], [], [], []
    cls_nums_all = 0
    cls_acc_all = 0



    multi_label_meter = AveragePrecisionMeter(difficult_examples=False)
    multi_label_meter.reset()

    for i, (image, label, text, fake_image_box, fake_word_pos, W, H) in enumerate(
            metric_logger.log_every(args, data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)

        text_input = tokenizer(text, max_length=128, truncation=True, add_special_tokens=True,
                                   return_attention_mask=True, return_token_type_ids=False)

        text_input, fake_token_pos, _ = text_input_adjust(text_input, fake_word_pos, device)

        logits_real1_fake, logits1_multicls, output_coord, logits_tok = model(image, label, text_input, fake_image_box,
                                                                                fake_token_pos, is_train=False)
        ##================= real/fake cls ========================##
        cls_label = torch.ones(len(label), dtype=torch.long).to(image.device)

        real_label_pos = np.where(np.array(label) == 'orig')[0].tolist()

        cls_label[real_label_pos] = 0

        y_true.extend(cls_label.cpu().flatten().tolist())
        cls_nums_all += cls_label.shape[0]
         
        # ----- multi metrics -----
        target, _ = get_multi_label(label, image)
        ##================= bbox cls ========================##
        boxes1 = box_ops.box_cxcywh_to_xyxy(output_coord)
        # boxes2 = box_ops.box_cxcywh_to_xyxy(fake_image_box)
        print(boxes1)
        # print(boxes2)
            # 预测的tensor([[0.2696, 0.1382, 0.4156, 0.4262]])
            # tensor([[0.2638, 0.1328, 0.4174, 0.4492]])
            ##================= token cls ========================##
        token_label = text_input.attention_mask[:, 1:].clone()  # [:,1:] for ingoring class token
        token_label[token_label == 0] = -100  # -100 index = padding token
        token_label[token_label == 1] = 0

        for batch_idx in range(len(fake_token_pos)):
            fake_pos_sample = fake_token_pos[batch_idx]
            if fake_pos_sample:
                for pos in fake_pos_sample:
                    token_label[batch_idx, pos] = 1

            logits_tok_reshape = logits_tok.view(-1, 2)
            logits_tok_pred = logits_tok_reshape.argmax(1)
            print(logits_tok_pred)

    return logits_tok_pred,boxes1
class Args:
    def __init__(self):
        self.config = './configs/test.yaml'
        self.checkpoint = ''
        self.resume = False
        self.output_dir = 'results'
        self.text_encoder = 'bert-base-uncased'
        self.device = 'cpu'
        self.seed = 777
        self.distributed = False
        self.rank = 0
        self.world_size = 1
        self.dist_url = 'tcp://127.0.0.1:23451'
        self.dist_backend = 'nccl'
        self.launcher = 'pytorch'
        self.log_num = 20240709  # Replace with your desired value for log_num
        self.model_save_epoch = 5
        self.token_momentum = False
        self.test_epoch = 'best'
        self.gpu = 0
        self.log = True


def main_worker(gpu, args, config):


    eval_type = os.path.basename(config['val_file'][0]).split('.')[0]
    if eval_type == 'test':
        eval_type = 'all'
    log_dir = 'results'
    os.makedirs(log_dir, exist_ok=True)
    log_file = 'results/shell_{eval_type}.txt'
    logger = setlogger(log_file)



    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Model ####
    # tokenizer = BertTokenizerFast.from_pretrained(args.text_encoder)
    tokenizer =BertTokenizerFast.from_pretrained('bertbaseuncased',
                                              do_lower_case=True,local_files_only=True)

    print(f"Creating HAMMER")
    model = HAMMER(args=args, config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, init_deit=True)

    model = model.to('cpu')

    checkpoint_dir = f'{args.output_dir}/{args.log_num}/checkpoint_{args.test_epoch}.pth'

    checkpoint = torch.load(checkpoint_dir, map_location='cpu')
    state_dict = checkpoint['model']

    pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
    state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped

    # model.load_state_dict(state_dict)
    print('load checkpoint from %s' % checkpoint_dir)
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

        #### Dataset ####
    print("Creating dataset")
    _, val_dataset = create_dataset(config)

    if args.distributed:
        samplers = create_sampler([val_dataset], [True], args.world_size, args.rank) + [None]
    else:
        samplers = [None]

    val_loader = create_loader([val_dataset],
                               samplers,
                               batch_size=[config['batch_size_val']],
                               num_workers=[4],
                               is_trains=[False],
                               collate_fns=[None])[0]

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    print("Start evaluation")

    logits_tok_pred,boxes1=evaluation(args, model_without_ddp, val_loader, tokenizer, device,
                                                                config)
    return logits_tok_pred,boxes1



# 定义一个函数来手动解析 YAML 文件
def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
 
def detect_aigc_mul(path):
    args_instance = Args()
    # 使用定义的函数加载配置文件

    img = Image.open(path)

    # 使用 pytesseract 进行 OCR
    text = pytesseract.image_to_string(img)
    # 2. 构造 JSON Lines 数据
    data = [{

        "id": 1,
        "image": path,
        "text": text,
        "fake_cls": " ",
        "fake_image_box": [],
        "fake_text_pos": [],
        "mtcnn_boxes": []

    }]

    # 3. 写入到 JSON Lines 文件
    jsonl_file = '../datasets/DGM4/metadata/val2.json'
    with jsonlines.open(jsonl_file, mode='w') as writer:
        writer.write(data)
    config = load_yaml_config(args_instance.config)
    # main_worker(0, args, config)
    logits_tok_pred, boxes1 = main_worker(0, args_instance, config)

    tensor = torch.tensor(boxes1)
    box1 = tensor.tolist()
    count_of_ones = logits_tok_pred.count(1)

    # Step 2: 计算百分比
    total_elements = len(logits_tok_pred)
    percentage_of_text= (count_of_ones / total_elements) * 100

    # 获取图像的宽度和高度
    height, width = img.shape[:2]

    # 计算边界框的像素坐标
    xmin = int(box1[0][0] * width)
    ymin = int(box1[0][1] * height)
    xmax = int(box1[0][2] * width)
    ymax = int(box1[0][3] * height)

    # 绘制边界框
    cv2.rectangle(path, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # 这里的(0, 255, 0)是绿色，2是线条宽度
    cv2.imwrite('pictureresults/bounding_box_result.jpg', img)
    print('图片已存入结果文件夹')
    return text,percentage_of_text

