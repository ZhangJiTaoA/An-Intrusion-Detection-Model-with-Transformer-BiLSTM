import torch.nn as nn


# 输入维度:(batch_size,bilstm_hidden_size*2)
# 输出维度:(batch_size,多分类或二分类)

class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(DNN, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size // 2)
        self.linear1_act = nn.ReLU()
        self.linear1_drop = nn.Dropout()
        self.linear2 = nn.Linear(input_size // 2, hidden_size)
        self.linear2_act = nn.ReLU()
        self.linear2_drop = nn.Dropout()
        self.linear3 = nn.Linear(hidden_size, output_size)
        # self.linear3_act = nn.ReLU()
        # self.linear3_drop = nn.Dropout()
        # self.linear4 = nn.Linear(16, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear1_act(x)
        x = self.linear1_drop(x)
        x = self.linear2(x)
        x = self.linear2_act(x)
        x = self.linear2_drop(x)
        x = self.linear3(x)
        # x = self.linear3_act(x)
        # x = self.linear3_drop(x)
        # x = self.linear4(x)
        return x
