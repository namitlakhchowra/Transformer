
import torch
import torch.nn as nn


class AddNorm(nn.Module):

    def __init__(self, d_model, dropout):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer_output):

        x = x + self.dropout(sublayer_output)

        return self.norm(x)
