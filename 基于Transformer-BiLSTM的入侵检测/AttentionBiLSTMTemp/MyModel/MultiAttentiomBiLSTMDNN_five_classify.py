import torch
import torch.nn as nn
from ChildModel.Embedding import Embedding
from ChildModel.MultiHeadSelfAttention import MultiHeadSelfAttention
from ChildModel.Encoder import Encoder
from ChildModel.BiLSTM import BiLSTM
from ChildModel.DNN import DNN
from config import *


class MultiAttentionBiLSTMDNN_Five_Classify(nn.Module):
    def __init__(self):
        super(MultiAttentionBiLSTMDNN_Five_Classify, self).__init__()
        self.emb = Embedding(EmbSize)
        self.multi_attention = MultiHeadSelfAttention(AttentionHeadNum, AttentionHeadSize, AttentionDropout)
        self.bilstm = BiLSTM(BiLSTMInputSize, BiLSTMHiddenSize, BiLSTMDropout)
        self.dnn = DNN(DNNInputSize, DNNHiddenSize, FiveClassify, DNNDropout)

    def forward(self, x):
        x = self.emb(x)
        x = self.multi_attention(x)
        x = self.bilstm(x)
        x = self.dnn(x)
        return x

    def _get_name(self):
        return "MultiAttentionBiLSTMDNN_Five_Classify"
