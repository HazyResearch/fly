import math
import torch
import torch.nn as nn

from einops import rearrange


# Adapted from https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/exact_topk_attention.py
class ExactTopKAttention(nn.Module):
    """Implement the oracle top-k softmax attention.
    Arguments
    ---------
        top-k: The top k keys to attend to  (default: 32)
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
    """
    def __init__(self, topk, softmax_temp=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__()
        self.topk = topk
        self.softmax_temp = softmax_temp
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=False):
        # Extract some shapes and compute the temperature
        B, T, H, E = query.shape
        _, S, _, D = value.shape
        softmax_temp = self.softmax_temp or 1 / math.sqrt(E)
        topk = min(self.topk, S)

        # Scale the query instead of applying the softmax temperature to the
        # dot products
        query = query * softmax_temp

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("bthe,bshe->bhts", query, key)
        if attn_mask is not None and not attn_mask.all_ones:
            QK.masked_fill_(~attn_mask.bool_matrix, float('-inf'))
        if key_padding_mask is not None and not key_padding_mask.all_ones:
            QK.masked_fill_(rearrange(~key_padding_mask.bool_matrix, 'b s -> b 1 1 s'),
                            float('-inf'))

        _, topk_idx = torch.topk(QK, topk, dim=-1, sorted=False)
        non_topk_mask = torch.ones_like(QK, dtype=torch.bool)
        non_topk_mask.scatter_(dim=-1, index=topk_idx, src=torch.zeros_like(non_topk_mask))
        QK.masked_fill_(non_topk_mask, float('-inf'))

        # Compute the attention and the weighted average
        attn = torch.softmax(QK, dim=-1)
        A = self.dropout(attn)
        output = torch.einsum("bhts,bshd->bthd", A, value)
        return output, attn if need_weights else None
