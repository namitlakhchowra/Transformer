
import torch
import torch.nn as nn

from embeddings import TokenEmbedding
from positional_encoding import PositionalEncoding
from encoder import Encoder
from decoder import Decoder
from output_layer import OutputLayer


class Transformer(nn.Module):

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        num_heads,
        d_ff,
        dropout,
        num_layers,
        max_len
    ):
        super().__init__()

        self.src_embedding = TokenEmbedding(
            src_vocab_size, d_model
        )

        self.tgt_embedding = TokenEmbedding(
            tgt_vocab_size, d_model
        )

        self.positional_encoding = PositionalEncoding(
            d_model, max_len
        )

        self.encoder = Encoder(
            d_model, num_heads, d_ff, dropout, num_layers
        )

        self.decoder = Decoder(
            d_model, num_heads, d_ff, dropout, num_layers
        )

        self.output_layer = OutputLayer(
            d_model, tgt_vocab_size
        )

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):

        src = self.src_embedding(src)
        src = self.positional_encoding(src)

        encoder_output = self.encoder(src, src_mask)

        tgt = self.tgt_embedding(tgt)
        tgt = self.positional_encoding(tgt)

        decoder_output = self.decoder(
            tgt,
            encoder_output,
            src_mask,
            tgt_mask
        )

        return self.output_layer(decoder_output)
