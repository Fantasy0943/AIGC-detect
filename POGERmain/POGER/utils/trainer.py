import torch
import numpy as np
from tqdm import tqdm
from transformers import RobertaTokenizer

from utils.utils import Averager, metrics

class Trainer:
    def __init__(self, device, pretrain_model, train_dataloader, test_dataloader, epoch, lr, model_save_path, n_classes):
        self.device = device
        self.epoch = epoch
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.n_classes = n_classes

        self.tokenizer = RobertaTokenizer.from_pretrained('../get_feature/roberta-base')
        # Need to define self.model_save_path, self.model, self.criterion and self.optimizer in derived class

    def get_loss(self, batch):
        # Need to be implemented in derived class
        return None

    def get_output(self, batch):
        # Need to be implemented in derived class
        return None

    def train(self):
        for epoch in range(self.epoch):
            print('----epoch %d----' % (epoch+1))
            self.model.train()
            avg_loss = Averager()
            for i, batch in enumerate(tqdm(self.train_dataloader)):
                self.optimizer.zero_grad()
                loss = self.get_loss(batch)
                loss.backward()
                self.optimizer.step()
                avg_loss.add(loss.item())
            results = self.test(self.test_dataloader)
            print('epoch %d: loss = %.4f, acc = %.4f, f1 = %.4f, auc_ovo = %.4f' % (epoch+1, avg_loss.get(), results['accuracy'], results['f1'], results['auc_ovo']))
            print('P/R per class: ', end='')
            for i in range(self.n_classes):
                print('%.2f/%.2f ' % (results['precision'][i] * 100, results['recall'][i] * 100), end='')
            print()
            print('F1 per class: ', end='')
            for i in range(self.n_classes):
                print('%.2f ' % (results['detail_f1'][i] * 100), end='')
            print()

            torch.save(self.model.state_dict(), self.model_save_path)

    def test(self, dataloader):
        self.model.eval()

        y_true = torch.empty(0)
        y_score = torch.empty((0, self.n_classes))
        for i, batch in enumerate(tqdm(dataloader)):
            output = self.get_output(batch).cpu()
            output = torch.softmax(output, dim=1)
            y_score = torch.cat((y_score, output))
            y_true = torch.cat((y_true, batch['label']))

        results = metrics(y_true, y_score)
        return results
    def detect(self):
        self.model.eval()

        y_true = torch.empty(0)
        y_score = torch.empty((0, self.n_classes))
        item={"text": "Yes By leaving them behind. It's not about standing up to them. Unless you are being physically threatened as your life is on the line. If they are a sociopath or psychopath you standing up against them verbally will put you further into their web of abuse. I can guarantee they will use this against you. They absolutely love it when this happens. This is the only love they can feel. Which is your destruction. DON'T DO IT In some cases your life will be put into danger and they will stalk and harass you for YEARS after you leave. If you have children or large assets get the best lawyer possible and only communicate through them. Standing up to a low level narcissist is a different story. They are extremely WEAK and confronting them will sometimes work. But, my guidance is that you are playing with your life if you do this. Time to start not thinking anything about this person and put all of your energy into yourself. Drop everything about them from your life. What you win is your life back", "label": "human", "label_int": 0, "est_prob_list": [[3.772119104862213, 4.605170185988091, 4.605170185988091, 4.605170185988091, 4.494445025920868, 4.605170185988091, 4.605170185988091, 4.605170185988091, 4.605170185988091, 4.605170185988091], [4.605170185988091, 4.605170185988091, 4.605170185988091, 4.605170185988091, 4.605170185988091, 4.605170185988091, 4.5370761087694165, 4.605170185988091, 4.605170185988091, 4.38629100471735], [4.605170185988091, 0.02701049883539048, 4.605170185988091, 2.6695696115493774, 4.605170185988091, 4.157551439993909, 2.2973719896000473, 4.605170185988091, 4.605170185988091, 3.431089524684804], [4.605170185988091, 0.009539705113628873, 4.605170185988091, 2.1506745614018676, 2.3025848865509033, 3.0323371092478433, 4.605170185988091, 3.64686009950108, 4.605170185988091, 2.101405367255211], [4.605170185988091, 0.012140400451401457, 4.605170185988091, 4.300578413476046, 4.605170185988091, 3.2241251468658447, 1.9387789579670844, 4.605170185988091, 4.605170185988091, 4.243873760104179], [4.605170185988091, 3.7534179752515073, 2.8371272433773522, 1.467640000573843, 4.605170185988091, 2.9802280870180256, 1.114360645636249, 4.605170185988091, 0.24187253642048673, 4.605170185988091], [4.605170185988091, 4.1588830833596715, 1.6739764335716716, 1.7165360479904674, 4.605170185988091, 1.4180430594344708, 0.44531101665536404, 4.605170185988091, 0.5212969236332861, 4.605170185988091]], "target_roberta_idx": [39, 44, 64, 82, 95, 127, 130, 143, 153, 161], "target_prob_idx": [37, 42, 62, 80, 92, 123, 126, 138, 147, 155]}
        inputs = self.tokenizer(item['text'], max_length=512, padding='max_length', truncation=True)

        mix_prob = torch.tensor(item['est_prob_list'])  # (n_feat, seq_len)
        mix_prob = mix_prob.t()  # (seq_len, n_feat)
        # fill mix_prob to (max_len, n_feat) and get bool mask
        mix_prob = mix_prob[:512]
        mix_prob_mask = torch.cat([torch.zeros(mix_prob.shape[0]), torch.ones(512 - mix_prob.shape[0])],
                                  dim=0).bool()
        mix_prob = torch.cat([mix_prob, torch.zeros(512 - mix_prob.shape[0], mix_prob.shape[1])], dim=0)

        target_roberta_idx = torch.tensor(item['target_roberta_idx'])  # (10, )
        target_roberta_idx = torch.cat([target_roberta_idx, torch.zeros(10 - target_roberta_idx.shape[0])],
                                       dim=0).long()

        target_prob_idx = torch.tensor(item['target_prob_idx'])  # (10, )
        target_prob_idx = torch.cat([target_prob_idx, torch.zeros(10- target_prob_idx.shape[0])], dim=0).long()
        batch={
            'input_ids': 1,
            'attention_mask': torch.tensor(inputs['attention_mask']),
            'mix_prob': mix_prob,
            'mix_prob_mask': mix_prob_mask,
            'target_roberta_idx': target_roberta_idx,
            'target_prob_idx': target_prob_idx,
            'label': torch.tensor(item['label_int']),
        }
        output = self.get_output(batch).cpu()
        print(output)
        return output
