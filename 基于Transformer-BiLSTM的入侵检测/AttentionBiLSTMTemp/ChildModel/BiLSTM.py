import torch.nn as nn
import torch
from config import *


# 输入维度:(batch_size,seq_len,emb_size)
# 输出维度:(batch_size,seq_len,2*emb_size)

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.bilstm = nn.LSTM(input_size, hidden_size, bidirectional=True)

    def forward(self, x):
        x = x.transpose(0, 1)  # 变成了输入维度:(seq_len,batch_size,emb_size)
        x, (_, _) = self.bilstm(x)
        # x1 = x[-1]
        # 变成了输入维度:(seq_len, batch,num_directions, hidden_size) num_directions=0或者1 分别表示前向结果和反向结果。
        x = x.view(SeqLen, -1, 2, self.hidden_size)
        x1 = x[-1, :, 0]  # 获得前向的结果
        x2 = x[0, :, 1]  # 获得后向的结果
        x = torch.cat((x1, x2), dim=-1)

        return x
