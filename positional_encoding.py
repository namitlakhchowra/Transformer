import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        for pos in range(max_len):
            for i in range(d_model):

                angle = pos / (10000 ** (2 * (i // 2) / d_model))

                if i % 2 == 0:
                    pe[pos, i] = math.sin(angle)
                else:
                    pe[pos, i] = math.cos(angle)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]