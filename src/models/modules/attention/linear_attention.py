"""Implement linear attention."""

import torch
import torch.nn as nn

import hydra

from einops import rearrange

from fast_transformers.feature_maps import elu_feature_map
from src.models.modules.masking import TriangularCausalMask

from src.models.modules.attention.performer_utils import causal_linear_attention, linear_attention


# Adapted from https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
class LinearAttention(nn.Module):
    """Implement unmasked attention using dot product of feature maps in
    O(N D^2) complexity.
    Given the query, key and value as Q, K, V instead of computing
        V' = softmax(Q.mm(K.t()), dim=-1).mm(V),
    we make use of a feature map function Φ(.) and perform the following
    computation
        V' = normalize(Φ(Q).mm(Φ(K).t())).mm(V).
    The above can be computed in O(N D^2) complexity where D is the
    dimensionality of Q, K and V and N is the sequence length. Depending on the
    feature map, however, the complexity of the attention might be limited.
    Arguments
    ---------
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor (default: elu(x)+1)
        normalization_eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
    """
    def __init__(self, query_dims, feature_map_cfg=None, normalization_eps=1e-6, softmax_temp=None):
        super().__init__()
        if softmax_temp is not None and feature_map_cfg is not None:
            feature_map_cfg.softmax_temp = softmax_temp
        self.feature_map = (
            hydra.utils.instantiate(feature_map_cfg, query_dims) if feature_map_cfg is not None
            else elu_feature_map(query_dims)
        )
        self.normalization_eps = normalization_eps

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=False):
        # Permute the dimensions to BHTE instead of BTHE
        query = rearrange(query, 'b t h e -> b h t e')
        key = rearrange(key, 'b s h e -> b h s e')
        value = rearrange(value, 'b s h d -> b h s d')

        # Apply the feature map to the query and key
        self.feature_map.new_feature_map(query.device)
        Q = self.feature_map.forward_queries(query)
        K = self.feature_map.forward_keys(key)

        # Apply the key padding mask and make sure that the attn_mask is
        # all_ones or is causal
        causal = attn_mask is not None and attn_mask.lower_triangular
        if not (attn_mask is None or attn_mask.all_ones or causal):
            raise RuntimeError(("LinearAttention does not support arbitrary attention masks"))
        if causal:
            assert query.shape[1] == key.shape[1], 'query and key must have the same sequence length'

        if key_padding_mask is not None:
            K.masked_fill_(~rearrange(key_padding_mask.bool_matrix, 'b s -> b 1 s 1'), 0.0)
        attn_fn = causal_linear_attention if causal else linear_attention
        out, attn = attn_fn(Q, K, value, eps=self.normalization_eps, need_weights=need_weights)
        out = rearrange(out, 'b h s d -> b s h d')
        return out, attn
