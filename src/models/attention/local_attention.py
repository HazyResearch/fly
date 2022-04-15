import math
import torch
import torch.nn as nn

from einops import rearrange

from fast_transformers.local_product import local_dot_product, local_weighted_average
from src.models.modules.masking import FullMask, LengthMask


# Adapted from https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/local_attention.py
class LocalAttention(nn.Module):
    """Implement fast local attention where a query can only attend to
    neighboring keys.
    In this attention module the query Q_i can only attend to a key K_j if
    |i-j| < local_context/2.
    Arguments
    ---------
        local_context: The neighborhood to consider for local attention.
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, local_context, softmax_temp=None, attention_dropout=0.0, device=None,
                 dtype=None):
        super().__init__()
        self.local_context = local_context
        self.softmax_temp = softmax_temp
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=False):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            query: (B, T, H, E) The tensor containing the query
            key: (B, S, H, E) The tensor containing the key
            value: (B, S, H, D) The tensor containing the value
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            key_padding_mask: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of
        """
        # Extract some shapes and compute the temperature
        B, T, H, E = query.shape
        _, S, _, D = value.shape
        softmax_temp = self.softmax_temp or 1 / math.sqrt(E)

        if attn_mask is None:
            attn_mask_additive_matrix = torch.zeros(T, S, device=query.device)
        else:
            attn_mask_additive_matrix = attn_mask.additive_matrix_finite
        if key_padding_mask is None:
            key_padding_mask_lengths = torch.full(size=(B,), fill_value=S, dtype=torch.long,
                                                  device=key.device)
        else:
            key_padding_mask_lengths = key_padding_mask.lengths

        # Permute the dimensions to BHTE instead of BTHE
        query = rearrange(query, 'b t h e -> b h t e').contiguous()
        key = rearrange(key, 'b s h e -> b h s e').contiguous()
        value = rearrange(value, 'b s h d -> b h s d').contiguous()

        QK = local_dot_product(
            query,
            key,
            attn_mask_additive_matrix,
            key_padding_mask_lengths,
            self.local_context
        )
        attn_local = torch.softmax(softmax_temp * QK, dim=-1)

        # If there are no valid keys for a query (because of local and key_padding_mask),
        # then that row of attn_local will be 1 / S as QK will all be -1e24 on that row.
        # We want that row to actually be zero.
        # If there are no valid keys because of attn_mask, we're not going to set that row to zero
        # because it's too much work to deal with the attn_mask.
        if key_padding_mask is not None and not key_padding_mask.all_ones:
            i = rearrange(torch.arange(T, device=query.device), 't -> 1 1 t 1')
            j = torch.arange(self.local_context, device=query.device)
            local_idx = i - self.local_context // 2 + j
            valid_idx_mask = ((local_idx >= 0)
                              & (local_idx < rearrange(key_padding_mask_lengths, 'b -> b 1 1 1')))
            attn_local = attn_local.masked_fill(~valid_idx_mask, 0.0)

        A = self.dropout(attn_local)
        V_new = local_weighted_average(A, value)

        attn = None
        if need_weights:
            attn = torch.zeros(B, H, T, S, device=query.device)
            i = rearrange(torch.arange(T, device=query.device), 't -> 1 1 t 1')
            j = torch.arange(self.local_context, device=query.device)
            local_idx = i - self.local_context // 2 + j
            valid_idx_mask = ((local_idx >= 0)
                              & (local_idx < rearrange(key_padding_mask_lengths, 'b -> b 1 1 1')))
            k = torch.arange(S, device=key.device)
            idx = k - i
            local_mask = ((idx >= -(self.local_context // 2))
                          & (idx < (self.local_context + 1) // 2)
                          & (k < rearrange(key_padding_mask_lengths, 'b -> b 1 1 1')))
            attn.masked_scatter_(local_mask, attn_local.masked_select(valid_idx_mask))

        return rearrange(V_new, 'b h t d -> b t h d'), attn
