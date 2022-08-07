import torch
import torch.nn as nn

# 输入维度：(batch_size,seq_len,1)
# 输出维度：(batch_size,seq_len,emb_size)

class Embedding(nn.Module):
    def __init__(self, emb_size):
        super(Embedding, self).__init__()
        self.fc1 = nn.Linear(1,16)
        # self.lrelu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(16, emb_size)
        # self.layer_norm = nn.LayerNorm(32)

    def forward(self,x):
        x = self.fc1(x)
        # x = self.lrelu(x)
        x = self.drop(x)
        x = self.fc(x)
        # x = self.layer_norm(x)
        return x
