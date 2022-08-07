import torch.nn as nn
import torch

SeqLen = 95

BiLSTMInputSize = 1
BiLSTMHiddenSize = 4


# 输入维度:(batch_size,seq_len,emb_size)
# 输出维度:(batch_size,seq_len,2*emb_size)

class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.hidden_size = BiLSTMHiddenSize
        self.bilstm = nn.LSTM(BiLSTMInputSize, BiLSTMHiddenSize, bidirectional=True)
        self.linear_act = nn.ReLU()
        self.linear = nn.Linear(BiLSTMHiddenSize * 2, 5)  # 由于是五分类，所以最终为5

    def forward(self, x):
        x = x.transpose(0, 1)  # 变成了输入维度:(seq_len,batch_size,emb_size)
        x, (_, _) = self.bilstm(x)
        # x1 = x[-1]
        # 变成了输入维度:(seq_len, batch,num_directions, hidden_size) num_directions=0或者1 分别表示前向结果和反向结果。
        x = x.view(SeqLen, -1, 2, self.hidden_size)
        x1 = x[-1, :, 0]  # 获得前向的结果
        x2 = x[0, :, 1]  # 获得后向的结果
        x = torch.cat((x1, x2), dim=-1)
        x = self.linear_act(x)
        x = self.linear(x)
        return x


LSTMInputSize = 1
LSTMHiddenSize = 8


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.hidden_size = BiLSTMHiddenSize
        self.lstm = nn.LSTM(LSTMInputSize, LSTMHiddenSize, bidirectional=False)
        self.linear_act = nn.ReLU()
        self.linear = nn.Linear(LSTMHiddenSize, 5)  # 由于是五分类，所以最终为5

    def forward(self, x):
        x = x.transpose(0, 1)  # 变成了输入维度:(seq_len,batch_size,emb_size)
        x, (_, _) = self.lstm(x)
        x = x[-1]  # 获取最后一个输出作为最终输出
        x = self.linear_act(x)
        x = self.linear(x)
        return x



