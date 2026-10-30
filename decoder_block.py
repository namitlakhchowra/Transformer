
import torch
import torch.nn as nn

from attention import MultiHeadAttention
from feed_forward import FeedForward
from add_norm import AddNorm


class DecoderBlock(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()

        # 1. Masked Self-Attention
        self.masked_attention = MultiHeadAttention(
            d_model, num_heads
        )

        self.add_norm1 = AddNorm(d_model, dropout)

        # 2. Cross-Attention
        self.cross_attention = MultiHeadAttention(
            d_model, num_heads
        )

        self.add_norm2 = AddNorm(d_model, dropout)

        # 3. Feed Forward Network
        self.feed_forward = FeedForward(
            d_model, d_ff
        )

        self.add_norm3 = AddNorm(d_model, dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):

        # Masked Self-Attention
        attention_output = self.masked_attention(
            x, mask=tgt_mask
        )

        x = self.add_norm1(x, attention_output)

        # Cross-Attention
        attention_output = self.cross_attention(
            x,
            encoder_output,
            mask=src_mask
        )

        x = self.add_norm2(x, attention_output)

        # Feed Forward
        ff_output = self.feed_forward(x)

        x = self.add_norm3(x, ff_output)

        return x
