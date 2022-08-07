import torch
import torch.nn as nn
from ChildModel.Embedding import Embedding
from ChildModel.MultiHeadSelfAttention import MultiHeadSelfAttention
from ChildModel.Encoder import Encoder
from ChildModel.BiLSTM import BiLSTM
from ChildModel.DNN import DNN
from config import *
# 该模型可作为MultiAttention+BiLSTM+DNN模型的对比试验

class MultiAttentionBiLSTMDNN_Two_Classify(nn.Module):
    def __init__(self):
        super(MultiAttentionBiLSTMDNN_Two_Classify, self).__init__()
        self.emb = Embedding(EmbSize)
        self.multi_attention = MultiHeadSelfAttention(AttentionHeadNum, AttentionHeadSize, AttentionDropout)
        self.bilstm = BiLSTM(BiLSTMInputSize, BiLSTMHiddenSize, BiLSTMDropout)
        self.dnn = DNN(DNNInputSize, DNNHiddenSize, TwoClassify, DNNDropout)

    def forward(self, x):
        x = self.emb(x)
        x = self.multi_attention(x)
        x = self.bilstm(x)
        x = self.dnn(x)
        return x

    def _get_name(self):
        return "MultiAttentionBiLSTMDNN_Two_Classify"
