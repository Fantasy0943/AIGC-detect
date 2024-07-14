import os
import math
import torch
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import nn, optim
from transformers import RobertaModel
from typing import List, Tuple

from utils.functions import MLP
from utils.trainer import Trainer as TrainerBase

class Attention(torch.nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(torch.nn.Module):
    """
    多头注意力机制模块，接受模型大小和头的数量作为参数。
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0  # 确保模型大小能够被头的数量整除

        # 假设 d_v 总是等于 d_k
        self.d_k = d_model // h  # 每个头的键（key）和查询（query）的维度
        self.h = h  # 注意力头的数量

        # 线性变换层，包括三个线性变换：query、key、value
        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = torch.nn.Linear(d_model, d_model)  # 输出的线性变换层
        self.attention = Attention()  # 注意力计算实例

        self.dropout = nn.Dropout(p=dropout)  # Dropout层，用于防止过拟合

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)  # 获取batch的大小
        if mask is not None:
            mask = mask.repeat(1, self.h, 1, 1)  # 复制mask以适应所有头的注意力计算

        # 1) 对query、key、value进行线性投影
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) 在批量中应用所有投影向量的注意力计算
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) 使用view进行“串联”操作，并应用最终的线性变换
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x), attn


class SelfAttentionFeatureExtract(torch.nn.Module):
    def __init__(self, multi_head_num, input_size, output_size=None):
        super(SelfAttentionFeatureExtract, self).__init__()
        self.attention = MultiHeadedAttention(multi_head_num, input_size)
        # self.out_layer = torch.nn.Linear(input_size, output_size)

    def forward(self, inputs, query, mask=None):
        if mask is not None:
            mask = mask.view(mask.size(0), 1, 1, mask.size(-1))  # 将mask的形状调整为(batch_size, 1, 1, sequence_length)

        # 使用多头注意力模块进行特征提取
        feature, attn = self.attention(query=query,
                                       value=inputs,
                                       key=inputs,
                                       mask=mask
                                       )
        return feature, attn
class ConvFeatureExtractionModel(nn.Module):

    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        conv_dropout: float = 0.0,
        conv_bias: bool = False,
    ):
        super().__init__()

        def block(n_in, n_out, k, stride=1, conv_bias=False):
            padding = k // 2  # 计算卷积的填充大小，确保输出大小与输入大小一致
            return nn.Sequential(
                nn.Conv1d(in_channels=n_in, out_channels=n_out, kernel_size=k, stride=stride, padding=padding, bias=conv_bias),
                nn.Dropout(conv_dropout),  # 添加Dropout层，用于防止过拟合
                nn.ReLU()  # 使用ReLU作为激活函数
            )

        in_d = 1  # 输入通道数，初始为1（假设输入是一维的）
        self.conv_layers = nn.ModuleList()
        for _, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            # 添加卷积块到层列表中
            self.conv_layers.append(
                block(in_d, dim, k, stride=stride, conv_bias=conv_bias))
            in_d = dim  # 更新下一层的输入通道数

    def forward(self, x):
        # x = x.unsqueeze(1)  # 如果输入x的维度不是[B, C, L]，可能需要调整维度
        for conv in self.conv_layers:
            x = conv(x)  # 逐层通过卷积块进行前向传播
        return x


class Model(nn.Module):

    def __init__(self, nfeat, nclasses, dropout=0.2, k=10):
        super(Model, self).__init__()
        self.nfeat = nfeat  # 输入特征的维度

        # 定义卷积特征提取模型，使用了一系列卷积层
        feature_enc_layers = [(64, 5, 1)] + [(128, 3, 1)] * 3 + [(64, 3, 1)]
        self.conv = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            conv_dropout=0.0,
            conv_bias=False,
        )

        embedding_size = nfeat * 64  # 计算特征提取后的嵌入维度

        # 定义Transformer编码器层
        self.encoder_layer = TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )

        # 堆叠两层Transformer编码器
        self.encoder = TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=2
        )

        seq_len = k  # 序列长度
        # 创建位置编码矩阵，用于为输入序列中的每个位置提供位置信息
        self.position_encoding = torch.zeros((seq_len, embedding_size))
        for pos in range(seq_len):
            for i in range(0, embedding_size, 2):
                # 根据位置和维度计算位置编码的正弦和余弦值
                self.position_encoding[pos, i] = torch.sin(
                    torch.tensor(pos / (10000 ** ((2 * i) / embedding_size)))
                )
                self.position_encoding[pos, i + 1] = torch.cos(
                    torch.tensor(pos / (10000 ** ((2 * (i + 1)) / embedding_size)))
                )

        self.norm = nn.LayerNorm(embedding_size)  # 应用Layer Normalization进行归一化
        self.dropout = nn.Dropout(0.1)  # 定义Dropout层，用于防止过拟合

        self.reducer = MLP(768, [384], embedding_size, dropout)
        # 使用MLP模型进行特征降维，输入维度为768，输出维度为embedding_size，中间隐藏层维度为384，dropout率为dropout。

        self.cross_attention_context = SelfAttentionFeatureExtract(1, embedding_size)
        self.cross_attention_prob = SelfAttentionFeatureExtract(1, embedding_size)
        # 初始化两个SelfAttentionFeatureExtract模型，用于计算概率特征和语义特征之间的交叉注意力，每个模型的输入维度为embedding_size。

        self.classifier = MLP(embedding_size * 2, [128, 32], nclasses, dropout)
        # 使用MLP模型进行最终的分类任务，输入维度为embedding_size * 2，中间隐藏层维度分别为128和32，输出维度为nclasses，dropout率为dropout。

    def conv_feat_extract(self, x):
        out = self.conv(x)
        out = out.transpose(1, 2)
        return out

    # 定义了一个卷积特征提取的函数，接收输入x并通过卷积层处理后将维度进行转置。

    def forward(self, prob_feature, sem_feature, target_roberta_idx):
        # extract sem_feature of target_roberta_idx
        context_feature = sem_feature.gather(1, target_roberta_idx.unsqueeze(-1).expand(-1, -1, sem_feature.shape[-1]))
        # 根据target_roberta_idx从sem_feature中提取对应的语义特征，形状为(batch_size, 10, 768)

        # reduce sem_feature
        context_feature = self.reducer(context_feature)
        # 使用reducer模型对语义特征进行降维处理，形状变为(batch_size, 10, embedding_size)

        # cnn
        prob_feature = prob_feature.transpose(1, 2)
        prob_feature = torch.cat([self.conv_feat_extract(prob_feature[:, i:i + 1, :]) for i in range(self.nfeat)],
                                 dim=2)
        # 将概率特征prob_feature通过卷积特征提取函数处理，得到形状为(batch_size, 10, embedding_size)

        prob_feature = prob_feature + self.position_encoding.cuda()
        # 将位置编码加到prob_feature上

        prob_feature = self.norm(prob_feature)
        prob_feature = self.encoder(prob_feature)
        prob_feature = self.dropout(prob_feature)
        # 对prob_feature进行Layer Normalization、Transformer编码和Dropout处理

        # reweight prob_feature
        prob_feature, _ = self.cross_attention_prob(prob_feature, context_feature)
        # 使用cross_attention_prob模型计算prob_feature和context_feature之间的注意力加权，得到形状为(batch_size, 10, embedding_size)

        # reweight context_feature
        context_feature, _ = self.cross_attention_context(context_feature, prob_feature)
        # 使用cross_attention_context模型计算context_feature和prob_feature之间的注意力加权，得到形状为(batch_size, 10, embedding_size)

        # concat
        merged = torch.cat([prob_feature, context_feature], dim=-1)
        # 将prob_feature和context_feature沿最后一个维度进行拼接，得到形状为(batch_size, 10, embedding_size * 2)

        # classify
        merged = self.classifier(merged)
        # 使用分类器模型对merged进行分类，得到形状为(batch_size, 10, nclasses)

        # mean
        output = merged.mean(dim=1)
        # 对merged在第二维度上取平均，得到形状为(batch_size, nclasses)，作为最终的输出结果

        return output

class Trainer(TrainerBase):
    def __init__(self, device, pretrain_model, train_dataloader, test_dataloader, epoch, lr, model_save_path, n_classes, n_feat, k):
        super(Trainer, self).__init__(device, pretrain_model, train_dataloader, test_dataloader, epoch, lr, model_save_path, n_classes)
        self.pretrain = RobertaModel.from_pretrained('../get_feature/roberta-base').to(device)
        self.model_save_path = model_save_path
        self.model = Model(nfeat=n_feat, nclasses=n_classes, k=k).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def get_loss(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        sem_feature = self.pretrain(input_ids, attention_mask).last_hidden_state.detach()
        prob_feature = batch['est_prob'].to(self.device)
        target_roberta_idx = batch['target_roberta_idx'].to(self.device)
        label = batch['label'].to(self.device)

        output = self.model(prob_feature, sem_feature, target_roberta_idx)
        loss = self.criterion(output, label)
        return loss

    def get_output(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        sem_feature = self.pretrain(input_ids, attention_mask).last_hidden_state.detach()
        prob_feature = batch['est_prob'].to(self.device)
        target_roberta_idx = batch['target_roberta_idx'].to(self.device)
        with torch.no_grad():
            output = self.model(prob_feature, sem_feature, target_roberta_idx)
        return output
