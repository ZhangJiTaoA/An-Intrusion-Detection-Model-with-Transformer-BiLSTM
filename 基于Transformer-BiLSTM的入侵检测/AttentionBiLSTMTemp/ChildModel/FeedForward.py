import torch.nn as nn

# 输入维度：(batch_size,seq_len,emb_size)
# 输出维度：(batch_size,sql_len,emb_size)

class FeedForward(nn.Module):
    def __init__(self, input_size, intermediate_size, dropout):
        super(FeedForward, self).__init__()
        self.dense1 = nn.Linear(input_size, intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, input_size)
        self.feedforward_act = nn.GELU()  # ??
        self.dropout = nn.Dropout(dropout)

    def forward(self, attention_x):
        attention_x = self.dense1(attention_x)
        attention_x = self.feedforward_act(attention_x)
        attention_x = self.dense2(attention_x)
        attention_x = self.dropout(attention_x)
        return attention_x
