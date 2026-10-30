
import torch
import torch.nn as nn

from encoder_block import EncoderBlock


class Encoder(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout, num_layers):
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):

        for layer in self.layers:
            x = layer(x, mask)

        return x
