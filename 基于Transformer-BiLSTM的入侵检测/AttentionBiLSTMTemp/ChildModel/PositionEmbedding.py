"""
-*- coding:utf-8 -*-
@Time : 2022/2/22 15:00
@Author : 597778912@qq.com
@File : PositionEmbedding.py
输入维度：(seq_len,1)
输出维度：(seq_len,emb_size)
"""
import numpy as np
import torch
import torch.nn as nn
from config import *

class PositionEmbedding(nn.Module):

    def __init__(self, seq_len, emb_size):
        super(PositionEmbedding, self).__init__()

        # 首先获得位置数组(seq_len,1)
        pos = np.expand_dims(np.arange(seq_len), 1)
        # 计算sin和cos里面的公式
        pe = pos / np.power(10000, 2 * np.expand_dims(np.arange(emb_size) // 2, 0) / emb_size)
        # 偶数位置用sin编码
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        # 奇数位置用cos编码
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        # pe [seq_len,emb_size]

        # # 加上batch_size
        # pe = np.expand_dims(pe,0).repeat(batch_size,axis=0)
        # 将numpy转换为Tensor
        self.pe = torch.FloatTensor(pe).to(Device)

    def forward(self):
        return self.pe
