# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
Take the standard Transformer as T2T Transformer
"""
import torch.nn as nn

from torchvision.ops import StochasticDepth

from einops import rearrange

import hydra

from src.models.modules.seq_common import Mlp


class T2TAttention(nn.Module):
    def __init__(self, dim, num_heads=8, in_dim=None, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0., attn_cfg=None):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim if in_dim is not None else dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, in_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if attn_cfg is None:
            self.attention_layer = None
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attention_layer = hydra.utils.instantiate(attn_cfg, softmax_temp=self.scale,
                                                           _recursive_=False)

    def forward(self, x):
        B, N, C = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)  # (B, N, D)
        v_og = v
        q, k, v = [rearrange(x, 'b n (n_head head_dim) -> b n n_head head_dim',
                             n_head=self.num_heads) for x in (q, k, v)]

        if self.attention_layer is None:  # Full attention
            q, k, v = [rearrange(x, 'b n n_head head_dim -> b n_head n head_dim') for x in (q, k, v)]
            attn = (q * self.scale) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            attn_output = (attn @ v).transpose(1, 2)
        else:
            attn_output, _ = self.attention_layer(q, k, v)

        x = rearrange(attn_output, 'b n h d -> b n (h d)')
        x = self.proj(x)
        x = self.proj_drop(x)

        # skip connection
        # because the original x has different size with current x, use v to do skip connection
        x = v_og + x
        return x


class Token_transformer(nn.Module):

    def __init__(self, dim, in_dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_cfg=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = T2TAttention(
            dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop,
            attn_cfg=attn_cfg,
        )
        self.drop_path = StochasticDepth(drop_path, mode='row')
        self.norm2 = norm_layer(in_dim)
        self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim * mlp_ratio),
                       out_features=in_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = self.attn(self.norm1(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
