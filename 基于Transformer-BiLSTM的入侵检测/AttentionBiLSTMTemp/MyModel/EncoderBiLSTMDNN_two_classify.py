import torch
import torch.nn as nn
from ChildModel.Embedding import Embedding
from ChildModel.MultiHeadSelfAttention import MultiHeadSelfAttention
from ChildModel.Encoder import Encoder
from ChildModel.BiLSTM import BiLSTM
from ChildModel.DNN import DNN
from config import *
# 正常的对比实验

class EncoderBiLSTMDNN_Two_Classify(nn.Module):
    def __init__(self):
        super(EncoderBiLSTMDNN_Two_Classify, self).__init__()
        self.emb = Embedding(EmbSize)
        self.encoder = Encoder()
        self.bilstm = BiLSTM(BiLSTMInputSize, BiLSTMHiddenSize, BiLSTMDropout)
        self.dnn = DNN(DNNInputSize, DNNHiddenSize, TwoClassify, DNNDropout)

    def forward(self, x):
        x = self.emb(x)
        x = self.encoder(x)
        x = self.bilstm(x)
        x = self.dnn(x)
        return x

    def _get_name(self):
        return "EncoderBiLSTMDNN_Two_Classify"
