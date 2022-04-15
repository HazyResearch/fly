# Adapted from https://github.com/lucidrains/linformer/blob/master/linformer/linformer.py
# and https://github.com/tatp22/linformer-pytorch

import math
import torch
import torch.nn as nn

from einops import rearrange


class LinformerAttention(nn.Module):
    """
    Arguments
    ---------
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
    """
    def __init__(self, seq_len, k=256, share_kv=False,
                 softmax_temp=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__()
        self.seq_len = seq_len
        self.share_kv = share_kv
        self.proj_k = nn.Parameter(torch.empty(seq_len, k, device=device, dtype=dtype))
        if not share_kv:
            self.proj_v = nn.Parameter(torch.empty(seq_len, k, device=device, dtype=dtype))
        self.softmax_temp = softmax_temp
        self.dropout = nn.Dropout(attention_dropout)
        self.reset_parameters()

    def reset_parameters(self):
        dim = self.proj_k.shape[-1]
        # If we're using the random projection interpretation, then we should initialize as
        # normal with std 1/sqrt(dim) (not 1/dim as in https://github.com/tatp22/linformer-pytorch/blob/master/linformer_pytorch/linformer_pytorch.py)
        std = 1 / math.sqrt(dim)
        nn.init.normal_(self.proj_k, mean=0.0, std=std)
        if not self.share_kv:
            nn.init.normal_(self.proj_v, mean=0.0, std=std)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=False):
        if attn_mask is not None:
            raise NotImplementedError('Linformer does not support attn_mask')
        # Extract some shapes and compute the temperature
        B, T, H, E = query.shape
        _, S_k, _, D = key.shape
        _, S_v, _, D = value.shape
        softmax_temp = self.softmax_temp or 1 / math.sqrt(E)
        assert S_k <= self.seq_len and S_v <= self.seq_len, f'the sequence length of the key / value must be at most {self.seq_len}'

        if key_padding_mask is not None and not key_padding_mask.all_ones:
            key = key.masked_fill(rearrange(~key_padding_mask.bool_matrix, 'b s -> b s 1 1'), 0.0)
            value = value.masked_fill(rearrange(~key_padding_mask.bool_matrix, 'b s -> b s 1 1'), 0.0)

        # Scale the key instead of applying the softmax temperature to the
        # dot products
        key = torch.einsum('bshd,sk->bkhd', key, self.proj_k[:S_k] * softmax_temp)
        value = torch.einsum('bshe,sk->bkhe', value,
                             self.proj_k[:S_v] if self.share_kv else self.proj_v[:S_v])

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("bthe,bkhe->bhtk", query, key)

        # Compute the attention and the weighted average
        attn = torch.softmax(QK, dim=-1)
        A = self.dropout(attn)
        output = torch.einsum("bhtk,bkhd->bthd", A, value)
        return output, attn if need_weights else None
