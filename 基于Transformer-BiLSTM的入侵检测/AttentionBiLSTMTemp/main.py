import numpy as np
import torch
import math
import time
from MyModel.EncoderBiLSTMDNN_five_classify import EncoderBiLSTMDNN_Five_Classify
from MyModel.MultiAttentiomBiLSTMDNN_two_classify import MultiAttentionBiLSTMDNN_Two_Classify
from config import *
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

if __name__ == '__main__':

    # model = nn.Sequential(
    #     nn.Linear(1,32,bias=True)
    # )
    #
    # input = torch.Tensor([0])
    # print(model(input))
    # 首先获得位置数组(seq_len,1)
    # pos = np.expand_dims(np.arange(4), 1)
    # # 计算sin和cos里面的公式
    # pe = pos / np.power(10000, 2 * np.expand_dims(np.arange(5) // 2, 0) / 5)
    # # 偶数位置用sin编码
    # pe[:, 0::2] = np.sin(pe[:, 0::2])
    # # 奇数位置用cos编码
    # pe[:, 1::2] = np.cos(pe[:, 1::2])
    # # pe [seq_len,emb_size]
    # pe = np.expand_dims(pe,0)
    # pe = pe.repeat(2,axis=0)
    # print(pe.shape)
    # # 将numpy转换为Tensor
    # pe = torch.from_numpy(pe).type(torch.float32)
    # print(pe)
    print([0.01**(i+1) for i in range(4)])