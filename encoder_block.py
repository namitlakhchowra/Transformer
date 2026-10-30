
import torch
import torch.nn as nn

from attention import MultiHeadAttention
from add_norm import AddNorm
from feed_forward import FeedForward


class EncoderBlock(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)

        self.add_norm1 = AddNorm(d_model, dropout)

        self.feed_forward = FeedForward(d_model, d_ff)

        self.add_norm2 = AddNorm(d_model, dropout)

    def forward(self, x, mask=None):

        # Multi-head self-attention
        attention_output = self.attention(x, mask=mask)

        # Residual connection + Layer Normalization
        x = self.add_norm1(x, attention_output)

        # Feed-forward network
        ff_output = self.feed_forward(x)

        # Residual connection + Layer Normalization
        x = self.add_norm2(x, ff_output)

        return x
