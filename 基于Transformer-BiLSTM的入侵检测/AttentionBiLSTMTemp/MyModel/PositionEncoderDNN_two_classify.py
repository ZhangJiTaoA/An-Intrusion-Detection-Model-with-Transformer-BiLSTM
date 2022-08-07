"""
-*- coding:utf-8 -*-
@Time : 2022/2/24 9:53
@Author : 597778912@qq.com
@File : PositionEncoderDNN_two_classify.py

该模型可作为Transformer+DNN和Position+Transformer+DNN的对比实验
"""
import torch
import torch.nn as nn
from ChildModel.Embedding import Embedding
from ChildModel.PositionEmbedding import PositionEmbedding
from ChildModel.MultiHeadSelfAttention import MultiHeadSelfAttention
from ChildModel.Encoder import Encoder
from ChildModel.BiLSTM import BiLSTM
from ChildModel.DNN import DNN
from config import *





class PositionEncoderDNN_Two_Classify(nn.Module):
    def __init__(self):
        super(PositionEncoderDNN_Two_Classify, self).__init__()
        self.emb = Embedding(EmbSize)
        self.positionEmb = PositionEmbedding(SeqLen, EmbSize)
        self.encoder = Encoder()

        self.linear1 = nn.Linear(126 * 32, 126)
        self.linear1_act = nn.ReLU()
        self.linear1_drop = nn.Dropout()
        self.linear2 = nn.Linear(126, 32)
        self.linear2_act = nn.ReLU()
        self.linear2_drop = nn.Dropout()
        self.linear3 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.emb(x)
        x2 = self.positionEmb()
        x = x +x2
        x = self.encoder(x)  # (512,126,32)
        x = x.view(-1, 126 * 32)
        x = self.linear1(x)
        x = self.linear1_act(x)
        x = self.linear1_drop(x)
        x = self.linear2(x)
        x = self.linear2_act(x)
        x = self.linear2_drop(x)
        x = self.linear3(x)
        return x

    def _get_name(self):
        return "PositionEncoderDNN_Two_Classify"
