
import torch
import torch.nn as nn

from decoder_block import DecoderBlock


class Decoder(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout, num_layers):
        super().__init__()

        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):

        for layer in self.layers:

            x = layer(
                x,
                encoder_output,
                src_mask,
                tgt_mask
            )

        return x
