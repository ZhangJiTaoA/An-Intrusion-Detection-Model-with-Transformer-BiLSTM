"""
-*- coding:utf-8 -*-
@Time : 2022/2/24 10:05
@Author : 597778912@qq.com
@File : PositionEncoderDNN_five_classify.py
"""
import torch
import torch.nn as nn
from ChildModel.Embedding import Embedding
from ChildModel.MultiHeadSelfAttention import MultiHeadSelfAttention
from ChildModel.PositionEmbedding import PositionEmbedding
from ChildModel.Encoder import Encoder
from ChildModel.BiLSTM import BiLSTM
from ChildModel.DNN import DNN
from config import *


class EncoderDNN_Five_Classify(nn.Module):
    def __init__(self):
        super(EncoderDNN_Five_Classify, self).__init__()
        self.emb = Embedding(EmbSize)
        self.encoder = Encoder()

        self.linear1 = nn.Linear(126 * 32, 126)
        self.linear1_act = nn.ReLU()
        self.linear1_drop = nn.Dropout()
        self.linear2 = nn.Linear(126, 32)
        self.linear2_act = nn.ReLU()
        self.linear2_drop = nn.Dropout()
        self.linear3 = nn.Linear(32, 5)

    def forward(self, x):
        x = self.emb(x)
        x = self.encoder(x)  # -->(512,126,32)
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
        return "EncoderDNN_Five_Classify"
