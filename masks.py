
import torch


def create_padding_mask(seq, pad_idx=0):

    mask = (seq != pad_idx)

    return mask.unsqueeze(1).unsqueeze(2)


def create_causal_mask(seq_len, device):

    mask = torch.tril(
        torch.ones(seq_len, seq_len, device=device)
    )

    return mask.bool()


def create_decoder_mask(seq, pad_idx=0):

    B, T = seq.shape

    padding_mask = create_padding_mask(seq, pad_idx)

    causal_mask = create_causal_mask(T, seq.device)

    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    return padding_mask & causal_mask
