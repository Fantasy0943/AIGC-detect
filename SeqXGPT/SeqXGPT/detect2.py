import os
import sys
import json
from collections import namedtuple

import torch
import numpy as np
import warnings
import torch.nn.functional as F
import torch.nn as nn


from tqdm import tqdm, trange
from sklearn.metrics import precision_score, recall_score
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from ..dataset.gen_features import get_features
warnings.filterwarnings('ignore')

project_path = os.path.abspath('')
if project_path not in sys.path:
    sys.path.append(project_path)
import backend_model_info
from dataloader import DataManager
from model import ModelWiseCNNClassifier, ModelWiseTransformerClassifier, TransformerOnlyClassifier



class SupervisedTrainer:
    def __init__(self, data, model, en_labels, id2label, args):
        self.data = data
        self.model = model
        self.en_labels = en_labels
        self.id2label =id2label

        self.seq_len = args.seq_len
        self.num_train_epochs = args.num_train_epochs
        self.weight_decay = args.weight_decay
        self.lr = args.lr
        self.warm_up_ratio = args.warm_up_ratio

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.model.to(self.device)
        self._create_optimizer_and_scheduler()

    def _create_optimizer_and_scheduler(self):
        num_training_steps = len(
            self.data.train_dataloader) * self.num_train_epochs
        no_decay = ["bias", "LayerNorm.weight"]

        named_parameters = self.model.named_parameters()
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in named_parameters
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                self.weight_decay,
            },
            {
                "params": [
                    p for n, p in named_parameters
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0,
            },
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.lr,
            betas=(0.9, 0.98),
            eps=1e-8,
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warm_up_ratio * num_training_steps,
            num_training_steps=num_training_steps)

    def train(self, ckpt_name='linear_en.pt'):
        for epoch in trange(int(self.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_steps = 0
            # train
            for step, inputs in enumerate(
                    tqdm(self.data.train_dataloader, desc="Iteration")):
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                with torch.set_grad_enabled(True):
                    labels = inputs['labels']
                    output = self.model(inputs['features'], inputs['labels'])
                    logits = output['logits']
                    loss = output['loss']
                    # print(loss.item())
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    tr_loss += loss.item()
                    nb_tr_steps += 1

            loss = tr_loss / nb_tr_steps
            print(f'epoch {epoch+1}: train_loss {loss}')
            # test
            # self.test()
            print('*' * 120)
            torch.save(self.model.cpu(), ckpt_name)
            self.model.to(self.device)

        torch.save(self.model.cpu(), ckpt_name)
        saved_model = torch.load(ckpt_name)
        self.model.load_state_dict(saved_model.state_dict())
        return
    
    def test(self, content_level_eval=False):
        self.model.eval()
        for step, inputs in enumerate(
                tqdm(self.data.test_dataloader, desc="Iteration")):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.device)
            with torch.no_grad():
                # labels = inputs['labels']
                output = self.model(inputs['features'], inputs['labels'])
                # logits = output['logits']
                preds = output['preds']
                return preds


    
    def content_level_eval(self, texts, true_labels, pred_labels):
        from collections import Counter

        true_content_labels = []
        pred_content_labels = []
        for text, true_label, pred_label in zip(texts, true_labels, pred_labels):
            true_label = np.array(true_label)
            pred_label = np.array(pred_label)
            mask = true_label != -1
            true_label = true_label[mask].tolist()
            pred_label = pred_label[mask].tolist()
            true_common_tag = self._get_most_common_tag(true_label)
            true_content_labels.append(true_common_tag[0])
            pred_common_tag = self._get_most_common_tag(pred_label)
            pred_content_labels.append(pred_common_tag[0])
        
        true_content_labels = [self.en_labels[label] for label in true_content_labels]
        pred_content_labels = [self.en_labels[label] for label in pred_content_labels]
        result = self._get_precision_recall_acc_macrof1(true_content_labels, pred_content_labels)
        return result

    def sent_level_eval(self, texts, true_labels, pred_labels):
        """
        """
        true_sent_labels = []
        pred_sent_labels = []
        for text, true_label, pred_label in zip(texts, true_labels, pred_labels):
            true_sent_label = self.get_sent_label(text, true_label)
            pred_sent_label = self.get_sent_label(text, pred_label)
            true_sent_labels.extend(true_sent_label)
            pred_sent_labels.extend(pred_sent_label)
        
        true_sent_labels = [self.en_labels[label] for label in true_sent_labels]
        # print(true_sent_labels)
        pred_sent_labels = [self.en_labels[label] for label in pred_sent_labels]
        result = self._get_precision_recall_acc_macrof1(true_sent_labels, pred_sent_labels)
        return result

    def get_sent_label(self, text, label):
        import nltk
        sent_separator = nltk.data.load('tokenizers/punkt/english.pickle')
        sents = sent_separator.tokenize(text)
        
        offset = 0
        sent_label = []
        for sent in sents:
            start = text[offset:].find(sent) + offset
            end = start + len(sent)
            offset = end
            
            split_sentence = self.data.split_sentence
            end_word_idx = len(split_sentence(text[:end]))
            if end_word_idx > self.seq_len:
                break
            word_num = len(split_sentence(text[start:end]))
            start_word_idx = end_word_idx - word_num
            tags = label[start_word_idx:end_word_idx]
            most_common_tag = self._get_most_common_tag(tags)
            print(f"判别的原文本为{text},识别的标签为{most_common_tag}")
            # print("most_common_tag",most_common_tag)
            sent_label.append(most_common_tag[0])
        
        if len(sent_label) == 0:
            print("empty sent label list")
        # print("sent_label-------------------------",sent_label)
        return sent_label
    
    def _get_most_common_tag(self, tags):
        """most_common_tag is a tuple: (tag, times)"""
        from collections import Counter
        print("self.id2label",self.id2label)
        print("tag",tags)
        # for tag in tags:
        #     if tag == -1:
        #         tag = 25
        #     tags.append(self.id2label[tag])
        tags = [self.id2label[tag] if tag != -1 else self.id2label[24] for tag in tags]
        # tags = [self.id2label[tag] for tag in tags]
        tags = [tag.split('-')[-1] for tag in tags]
        tag_counts = Counter(tags)
        most_common_tag = tag_counts.most_common(1)[0]
        print("most_common_tag",most_common_tag)

        return most_common_tag

    def _get_precision_recall_acc_macrof1(self, true_labels, pred_labels):
        accuracy = accuracy_score(true_labels, pred_labels)
        macro_f1 = f1_score(true_labels, pred_labels, average='macro')
        print("Accuracy: {:.1f}".format(accuracy*100))
        print("Macro F1 Score: {:.1f}".format(macro_f1*100))

        precision = precision_score(true_labels, pred_labels, average=None)
        recall = recall_score(true_labels, pred_labels, average=None)
        print("Precision/Recall per class: ")
        precision_recall = ' '.join(["{:.1f}/{:.1f}".format(p*100, r*100) for p, r in zip(precision, recall)])
        print(precision_recall)

        result = {"precision":precision, "recall":recall, "accuracy":accuracy, "macro_f1":macro_f1}
        return result


def construct_bmes_labels(labels):
    prefix = ['B-', 'M-', 'E-', 'S-']
    id2label = {}
    counter = 0

    for label, id in labels.items():
        for pre in prefix:
            id2label[counter] = pre + label
            counter += 1
    
    return id2label

def split_dataset(data_path, train_path, test_path, train_ratio=0.9):
    file_names = [file_name for file_name in os.listdir(data_path) if file_name.endswith('.jsonl')]
    print('*'*32)
    print('The overall data sources:')
    print(file_names)
    file_paths = [os.path.join(data_path, file_name) for file_name in file_names]

    total_samples = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            samples = [json.loads(line) for line in f]
            total_samples.extend(samples)
    
    import random
    random.seed(0)
    random.shuffle(total_samples)

    split_index = int(len(total_samples) * train_ratio)
    train_data = total_samples[:split_index]
    test_data = total_samples[split_index:]

    def save_dataset(fpath, data_samples):
        with open(fpath, 'w', encoding='utf-8') as f:
            for sample in tqdm(data_samples):
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    save_dataset(train_path, train_data)
    save_dataset(test_path, test_data)
    print()
    print("The number of train dataset:", len(train_data))
    print("The number of test  dataset:", len(test_data))
    print('*'*32)
    pass

import argparse
def parse_args():
    args_dict = {
        'model': 'Transformer',
        'gpu': '0',
        'train_mode': 'classify',
        'batch_size': 16,
        'seq_len': 1024,
        'train_ratio': 0.9,
        'split_dataset': False,
        'data_path': '/root/autodl-tmp/SeqXGPT/SeqXGPT/encoder_Data',
        'train_path': '/root/autodl-tmp/SeqXGPT/SeqXGPT/train.jsonl',
        'test_path': '/root/autodl-tmp/SeqXGPT/SeqXGPT/test_2.jsonl',
        'num_train_epochs': 20,
        'weight_decay': 0.1,
        'lr': 5e-5,
        'warm_up_ratio': 0.1,
        'do_test': True,
        'test_content': False
    }

    Args = namedtuple('Args', args_dict.keys())
    args = Args(**args_dict)

    return args

# python ./Seq_train/train.py --gpu=0 --split_dataset
# python ./Seq_train/train.py --gpu=0
def detect_aigc_chinese(text):
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # 准备要写入的数据字典
    data = {
        'text': text
    }

    # 指定要写入的文件路径
    file_path = 'test_1.json'

    # 打开文件进行写入操作，并将 data 写入文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

    get_features(type='cn', input_file='../test_1.jsonl', output_file='../test_2.jsonl')
    if args.split_dataset:
        print("Log INFO: split dataset...")
        split_dataset(data_path=args.data_path, train_path=args.train_path, test_path=args.test_path, train_ratio=args.train_ratio)

    # en_labels = backend_model_info.en_labels
    # en_labels = {
    #     'gpt2': 0,
    #     'gptneo': 1,
    #     'gptj': 2,
    #     'llama': 3,
    #     'gpt3re': 4,
    #     # 'gpt3sum': 3,
    #     'human': 5
    # }
    # en_labels = {
    #     'kimi': 0,
    #     'human': 1,
    # }
    cn_labels = {
        'wenzhong': 0,
        'sky_text': 1,
        'damo': 2,
        'chatglm': 3,
        'gpt3': 4,
        'human': 5,
        'moss': 6,
        'kimi': 7
    }

    id2label = construct_bmes_labels(cn_labels)
    label2id = {v: k for k, v in id2label.items()}

    data = DataManager(train_path=args.train_path, test_path=args.test_path, batch_size=args.batch_size, max_len=args.seq_len, human_label='human', id2label=id2label)
    
    """linear classify"""
    print('-' * 32 + 'classify' + '-' * 32)

    classifier = ModelWiseTransformerClassifier(id2labels=id2label, seq_len=args.seq_len)
    ckpt_name = ''

    trainer = SupervisedTrainer(data, classifier, cn_labels, id2label, args)


    ckpt_name = "save_checkpoint.pt"
    print("Log INFO: do test...")
    saved_model = torch.load(ckpt_name)
    trainer.model.load_state_dict(saved_model.state_dict())
    label=trainer.test(content_level_eval=args.test_content)
    if label==5:
        return 0,'human'
    else :
        predicted_label = id2label[label]
        return 1,predicted_label


