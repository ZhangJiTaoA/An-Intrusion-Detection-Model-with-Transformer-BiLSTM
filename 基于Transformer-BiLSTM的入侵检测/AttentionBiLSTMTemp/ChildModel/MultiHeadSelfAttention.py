import torch.nn as nn
import math
from config import *


# 输入维度：(batch_size,seq_len,emb_size)
# 输出维度：(batch_size,sql_len,emb_size)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,
                 attention_head_num,
                 attention_head_size,
                 dropout_prob=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.attention_head_num = attention_head_num
        self.attention_head_size = attention_head_size
        self.out_dim = attention_head_num * attention_head_size  # 一般注意力输入输出前后维度不变

        # 网络
        self.q_dense = nn.Linear(self.out_dim, self.out_dim)
        self.k_dense = nn.Linear(self.out_dim, self.out_dim)
        self.v_dense = nn.Linear(self.out_dim, self.out_dim)
        self.softmax = nn.Softmax(dim=-1)  # 在第几维上进行softmax运算，这里是最后一维。
        self.dropout = nn.Dropout(dropout_prob)
        self.o_dense = nn.Linear(self.out_dim, self.out_dim)

    def forward(self, x):
        qx = x
        kx = x
        vx = x
        q = self.q_dense(qx)
        k = self.k_dense(kx)
        v = self.v_dense(vx)

        # 先将batch_size*seq_len*embedding_size变成batch_size*seq_len*head*head_size
        # 再将batch_size*seq_len*head*head_size转置成batch_size*head*seq_len*head_size
        shape = list(x.size())
        batch_size = shape[0]
        seq_len = shape[1]
        q = q.view([batch_size, seq_len, self.attention_head_num, self.attention_head_size])
        q = q.transpose(1, 2)
        k = k.view([batch_size, seq_len, self.attention_head_num, self.attention_head_size])
        k = k.transpose(1, 2)
        v = v.view([batch_size, seq_len, self.attention_head_num, self.attention_head_size])
        v = v.transpose(1, 2)
        # q和k的转置相乘得到：[batch_size,head,seq_len,seq_len]
        attention_scores = torch.matmul(q, k.transpose(2, 3))
        # 因为q,k相乘，结果变大，对结果除以根号512  ???
        attention_scores = attention_scores / math.sqrt(float(self.attention_head_size))

        # 防止padding补全的0经过softmax后影响结果，对每个0值都加一个很大的负数，这样softmax后也等于0, 在这里不进行遮掩，所以无用

        attention_scores = self.softmax(attention_scores)
        attention_scores = self.dropout(attention_scores)
        attention_scores = torch.matmul(attention_scores, v)
        # permute改变维度，congiguous返回内存中连续的Tensor
        attention_scores = attention_scores.permute(0, 2, 1, 3).contiguous()
        attention_scores = attention_scores.view([batch_size, seq_len, self.out_dim])
        attention_scores = self.o_dense(attention_scores)
        return attention_scores
