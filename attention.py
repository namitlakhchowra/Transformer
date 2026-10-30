
import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key_value=None, mask=None):

        if key_value is None:
            key_value = query

        B, T_q, C = query.shape
        T_kv = key_value.size(1)

        Q = self.W_q(query)
        K = self.W_k(key_value)
        V = self.W_v(key_value)

        Q = Q.view(B, T_q, self.num_heads, self.d_k)
        K = K.view(B, T_kv, self.num_heads, self.d_k)
        V = V.view(B, T_kv, self.num_heads, self.d_k)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        scores = Q @ K.transpose(-2, -1)

        scores = scores / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(
                mask == 0, float('-inf')
            )

        attention = torch.softmax(scores, dim=-1)

        out = attention @ V

        out = out.transpose(1, 2).contiguous()

        out = out.view(B, T_q, C)

        return self.W_o(out)
