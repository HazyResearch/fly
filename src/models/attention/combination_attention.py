import torch
import torch.nn as nn

import hydra

from einops import rearrange


class CombinationAttention(nn.Module):
    """
    Arguments
    ---------
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
    """
    def __init__(self, d_head, n_heads, attn_cfg_0, attn_cfg_1, gating=True,
                 softmax_temp=None, device=None, dtype=None):
        """
        gating: whether to use sigmoid gating, otherwise we simply average the two attentions.

        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.d_head = d_head
        self.n_heads = n_heads
        self.gating = gating
        self.attn_0 = hydra.utils.instantiate(attn_cfg_0, softmax_temp=softmax_temp, **factory_kwargs)
        self.attn_1 = hydra.utils.instantiate(attn_cfg_1, softmax_temp=softmax_temp, **factory_kwargs)
        if gating:
            self.gate = nn.Conv1d(n_heads, n_heads, kernel_size=d_head, groups=n_heads)

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
        out_0, attn_0 = self.attn_0(query, key, value, attn_mask=attn_mask,
                                    key_padding_mask=key_padding_mask, need_weights=need_weights)
        out_1, attn_1 = self.attn_1(query, key, value, attn_mask=attn_mask,
                                    key_padding_mask=key_padding_mask, need_weights=need_weights)
        if self.gating:
            g = torch.sigmoid(rearrange(self.gate(rearrange(query, 'b t h e -> (b t) h e')),
                                        '(b t) h 1 -> b t h 1', t=query.shape[1]))
        else:
            g = 0.5
        out = torch.lerp(out_0, out_1, g)
        if attn_0 is None or attn_1 is None:
            attn = None
        else:
            attn = torch.lerp(attn_0, attn_1, rearrange(g, 'b t h 1 -> b h t 1'))
        return out, attn
