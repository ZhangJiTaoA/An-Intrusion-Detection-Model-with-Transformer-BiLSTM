import torch.nn as nn
from ChildModel.MultiHeadSelfAttention import MultiHeadSelfAttention
from ChildModel.FeedForward import FeedForward
from config import *
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.multi_attention = MultiHeadSelfAttention(AttentionHeadNum, AttentionHeadSize, AttentionDropout)
        self.attention_layerNorm = nn.LayerNorm(AttentionHeadSize*AttentionHeadNum)

        self.feed_forward = FeedForward(FFInputSize,FFIntermediateSize,FFDrop)
        self.feedForward_layerNorm = nn.LayerNorm(FFInputSize)

    def forward(self,x):
        attention_x = self.multi_attention(x)
        attention_x = x + attention_x
        attention_x = self.attention_layerNorm(attention_x)

        feedforward_x = self.feed_forward(attention_x)
        feedforward_x = attention_x+feedforward_x
        feedforward_x = self.feedForward_layerNorm(feedforward_x)
        return feedforward_x
