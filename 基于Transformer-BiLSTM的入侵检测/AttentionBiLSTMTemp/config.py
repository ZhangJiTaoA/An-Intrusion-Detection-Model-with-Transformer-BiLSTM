import time
import torch
import torch.nn as nn
from Data.data_handler import *
X_train, X_test, Y_train, Y_test = get_NSLKDD(2)
SeqLen = X_train.size()[1]
# SeqLen = 95
# 训练超参
BatchSize = 512
TwoClassifyEpoch = 20
TwoClassifyMilestones = []
FiveClassifyEpoch = 40
FiveClassifyMilestones = []
TwoClassifyLR = 1e-4
FiveClassifyLR = 1e-4

ALL_Dropout = 0.5
# emb参数
EmbSize = 32

# Attention参数,AttentionHeadNum能被EmbSize整除
AttentionHeadNum = 4
AttentionHeadSize = EmbSize // AttentionHeadNum  # 除法以后是float型，//则是整型
AttentionDropout = ALL_Dropout
# transformer中FeedForward参数
FFInputSize = EmbSize  # 输出仍然为FFInputSize
FFIntermediateSize = EmbSize * 2
FFDrop = ALL_Dropout

# BiLSTM参数
BiLSTMInputSize = EmbSize
BiLSTMHiddenSize = BiLSTMInputSize * 2  # LSTM的输出维度，如果是双向的还要*2
BiLSTMDropout = ALL_Dropout

# DNN参数
DNNInputSize = 2 * BiLSTMHiddenSize
#DNNInputSize = BiLSTMHiddenSize
DNNHiddenSize = EmbSize
DNNDropout = ALL_Dropout
TwoClassify = 2
FiveClassify = 5

# 显卡
device_num = 1  # 使用第几块显卡


def get_device(device_num):
    cuda_condition = torch.cuda.is_available()
    cuda0 = torch.device('cuda:0' if cuda_condition else 'cpu')
    cuda1 = torch.device('cuda:1' if cuda_condition else 'cpu')
    device = cuda0 if device_num == 0 else cuda1
    print("当前使用的显卡为：", device)
    return device


Device = get_device(device_num)
# 设置初始化随机数种子
seed = 1


# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

IsInitParams = True
def initNetParams(net,w=1):   # 二分类w=0.1,五分类w=1
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.1*w)   # 正态分布
            # nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            nn.init.xavier_uniform_(m.weight_ih_l0)   # 均匀分布
            nn.init.xavier_uniform_(m.weight_hh_l0)
            # nn.init.orthogonal_(m.weight_ih_l0)
            # nn.init.orthogonal_(m.weight_hh_l0)
            m.bias_ih_l0.data.zero_()
            m.bias_hh_l0.data.zero_()
    print("参数初始化完成")
