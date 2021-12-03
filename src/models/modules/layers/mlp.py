""" MLP module w/ dropout and configurable activation layer

Hacked together by / Copyright 2020 Ross Wightman
"""
from torch import nn as nn
import torch
from src.models.modules.layers.fastlinear import FastLinear, ButterflyLinear, RandomLinear, SLLinear, \
    SLXLinear, TopkLinear, TopkLrLinear, ButterflyGlobalLinear, NinjaTurtleLinear
from src.models.modules.layers.maskedlinear import MaskLinearWrap
import math
from einops import rearrange

import hydra

from src.ops.butterfly_factor import butterfly_factor_to_matrix


class MlpCustom(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 act_fn=None, drop=0., linear1_cfg=None, linear2_cfg=None):
        """TD [2021-10-27] act_fn takes precedence over act_layer if set.
        This is to support Pytorch 1.10 Transformer interface that construct the activation
        *function*, not the activation *layer*.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if linear1_cfg is None:
            self.fc1 = nn.Linear(in_features, hidden_features)
        else:
            self.fc1 = hydra.utils.instantiate(linear1_cfg, in_features, hidden_features,
                                               _recursive_=False)
        self.act = act_layer() if act_fn is None else act_fn
        if linear2_cfg is None:
            self.fc2 = nn.Linear(hidden_features, out_features)
        else:
            self.fc2 = hydra.utils.instantiate(linear2_cfg, hidden_features, out_features,
                                               _recursive_=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# Copied from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/mlp.py
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.saving = 1.0

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GluMlp(nn.Module):
    """ MLP w/ GLU style gating
    See: https://arxiv.org/abs/1612.08083, https://arxiv.org/abs/2002.05202
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.Sigmoid, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert hidden_features % 2 == 0
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features // 2, out_features)
        self.drop = nn.Dropout(drop)

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        fc1_mid = self.fc1.bias.shape[0] // 2
        nn.init.ones_(self.fc1.bias[fc1_mid:])
        nn.init.normal_(self.fc1.weight[fc1_mid:], std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x, gates = x.chunk(2, dim=-1)
        x = x * self.act(gates)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GatedMlp(nn.Module):
    """ MLP as used in gMLP
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 gate_layer=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        if gate_layer is not None:
            assert hidden_features % 2 == 0
            self.gate = gate_layer(hidden_features)
            hidden_features = hidden_features // 2  # FIXME base reduction on gate property?
        else:
            self.gate = nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.gate(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    """
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, norm_layer=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=True)
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


# Legacy code... delete soon

class ButterflyFactorBanditNewMlp(nn.Module):
    """ ButterflyMlp, similar to Mlp layers in MLP-Mixer
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., factor=0, base_size=3):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop = nn.Dropout(drop)
        self.register_buffer('sparse_mask', torch.eye(in_features))
        butterfly_1d = torch.eye(int(math.sqrt(in_features)))
        window_size = base_size**factor
        for i in range(0, int(math.sqrt(in_features)) - window_size):
            #     for j in range(window_size):
            butterfly_1d[i, i + window_size] = 1.
            butterfly_1d[i + window_size, i] = 1.
        self.sparse_mask = torch.kron(butterfly_1d, butterfly_1d)

    def forward(self, x):
        # sparse
        attn_s = torch.einsum("ds,td->st", self.fc1.weight, self.fc2.weight)
        attn = attn_s*self.sparse_mask
        attn = self.drop(attn)
        x = torch.einsum("bds,st->bdt", x, attn) + self.fc2.bias
        x = self.act(x)
        return x


class ButterflyFactorNewMlp(nn.Module):
    """ ButterflyMlp, similar to Mlp layers in MLP-Mixer
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., factor=0, base_size=3):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop = nn.Dropout(drop)
        self.register_buffer('sparse_mask', torch.zeros([in_features, out_features]))
        b = base_size
        log_b_n = int(math.log(math.sqrt(in_features), b))
        n = b ** log_b_n
        twiddle = torch.arange(1, n * b + 1, dtype=torch.float).reshape(n // b, b, b)
        butterfly_1d = butterfly_factor_to_matrix(twiddle, factor_index=factor)
        self.sparse_mask = torch.kron(butterfly_1d, butterfly_1d)
        self.sparse_mask[self.sparse_mask > 0] = 1.

    def forward(self, x):
        # sparse
        attn_s = torch.einsum("ds,td->st", self.fc1.weight, self.fc2.weight)
        attn = attn_s*self.sparse_mask
        attn = self.drop(attn)
        x = torch.einsum("bds,st->bdt", x, attn) + self.fc2.bias
        x = self.act(x)
        return x


class ButterflyNewMlp(nn.Module):
    """ ButterflyMlp, similar to Mlp layers in MLP-Mixer
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., window_size=6, stripes=3):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop = nn.Dropout(drop)
        self.local_stride = window_size

        self.register_buffer('sparse_mask', torch.zeros([in_features, out_features]))
        stride = int(math.sqrt(in_features))
        for k in range(stripes):
            patch_start = stride * k
            for i in range(0, in_features, window_size):
                self.sparse_mask[patch_start + i:patch_start + i + window_size, i:i + window_size] = 1.
        self.sparse_mask = (self.sparse_mask.bool() | self.sparse_mask.bool().t()).float()

    def forward(self, x):
        # sparse
        attn_s = torch.einsum("ds,td->st", self.fc1.weight, self.fc2.weight)
        attn = attn_s*self.sparse_mask
        attn = self.drop(attn)
        x = torch.einsum("bds,st->bdt", x, attn) + self.fc2.bias
        x = self.act(x)
        return x


class RandomSparseNewMlp(nn.Module):
    """ SparseLrMLP, similar to Mlp layers in MLP-Mixer but with extra gelu act and low-rank
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., sparse_ratio=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop = nn.Dropout(drop)
        self.register_buffer('sparse_mask', torch.zeros([in_features, out_features]))
        nnz = int(in_features*out_features*sparse_ratio)
        ind = torch.randperm(in_features*out_features)[:nnz]
        tmp_mask = torch.zeros([in_features*out_features])
        tmp_mask[ind] = 1.
        self.sparse_mask.data = tmp_mask.view(in_features, out_features)

    def forward(self, x):
        # sparse
        attn_s = torch.einsum("ds,td->st", self.fc1.weight, self.fc2.weight) + self.fc2.bias
        attn = attn_s*self.sparse_mask
        attn = self.drop(attn)
        x = torch.einsum("bds,st->bdt", x, attn)
        x = self.act(x)
        return x


class NewMlp(nn.Module):
    """ newMlp, similar to Mlp layers in MLP-Mixer but with extra gelu act and low-rank
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # dense
        attn = torch.einsum("ds,td->st", self.fc1.weight, self.fc2.weight) + self.fc2.bias
        attn = self.drop(attn)
        x = torch.einsum("bds,st->bdt", x, attn)
        x = self.act(x)
        return x


class ButterflySimpleMlp(nn.Module):
    """ SimpleButterflyMlp, similar to Mlp layers in MLP-Mixer but with extra gelu act and low-rank
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., window_size=3):
        super().__init__()
        out_features = out_features or in_features
        self.fc = nn.Linear(in_features, out_features, bias=True)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.local_stride = window_size

        self.register_buffer('sparse_mask', torch.zeros([in_features, out_features]))
        stride = int(math.sqrt(in_features))
        for k in range(window_size):
            patch_start = stride * k
            for i in range(0, in_features):
                self.sparse_mask[patch_start + i:patch_start + i + window_size, i:i + window_size] = 1
        self.sparse_mask = (self.sparse_mask.bool() | self.sparse_mask.bool().t()).float()

    def forward(self, x):
        attn = x @ (self.fc.weight * self.sparse_mask) + self.fc.bias
        attn = self.act(attn)
        attn = self.drop(attn)
        return attn


class SimpleMlp(nn.Module):
    """ SimpleMlp, similar to Mlp layers in MLP-Mixer but with extra gelu act and low-rank
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        self.fc = nn.Linear(in_features, out_features, bias=True)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        attn = self.fc(x)
        attn = self.act(attn)
        attn = self.drop(attn)
        return attn



class NinjaTurtleMlp(nn.Module):
    """ newMlp, similar to Mlp layers in MLP-Mixer but with extra gelu act and low-rank
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 window_size=3, step=1, stripes_1=3, stripes_2=1, gtoken=1, block_size=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = NinjaTurtleLinear(in_features, hidden_features, bias=True, window_size=window_size,
                                         stripes=stripes_1, step=step, gtoken=gtoken, block_size=block_size)
        self.act = act_layer()
        self.fc2 = NinjaTurtleLinear(hidden_features, out_features, bias=True, window_size=window_size,
                                         stripes=stripes_2, step=step, gtoken=gtoken, block_size=block_size)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ButterflyGlobalMlp(nn.Module):
    """ newMlp, similar to Mlp layers in MLP-Mixer but with extra gelu act and low-rank
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 window_size=3, step=1, stripes_1=3, stripes_2=1, gtoken=1, block_size=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ButterflyGlobalLinear(in_features, hidden_features, bias=True, window_size=window_size,
                                         stripes=stripes_1, step=step, gtoken=gtoken, block_size=block_size)
        self.act = act_layer()
        self.fc2 = ButterflyGlobalLinear(hidden_features, out_features, bias=True, window_size=window_size,
                                         stripes=stripes_2, step=step, gtoken=gtoken, block_size=block_size)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TopkGradMlp(nn.Module):
    """ newMlp, similar to Mlp layers in MLP-Mixer but with extra gelu act and low-rank
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 topk_ratio=0.1, window_size=1, stripes=1, step=1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = MaskLinearWrap(in_features, hidden_features, bias=True, topk_ratio=topk_ratio,
                                  window_size=window_size, stripes=stripes, step=step)
        self.act = act_layer()
        self.fc2 = MaskLinearWrap(hidden_features, out_features, bias=True, topk_ratio=topk_ratio,
                                  window_size=window_size, stripes=stripes, step=step)
        self.drop = nn.Dropout(drop)
        self.saving = topk_ratio

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TopkActMlp(nn.Module):
    """ Sanity check if topk work for activation
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., topk_ratio=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.saving = topk_ratio
        self.topk_ratio = topk_ratio

    def forward(self, x):
        x = self.fc1(x)
        topk, ind = torch.topk(x, int(x.shape[-1]*self.topk_ratio), dim=-1)
        mask = torch.zeros_like(x)
        mask = mask.scatter(-1, ind, 1.)
        x = x*mask
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        topk, ind = torch.topk(x, int(x.shape[-1]*self.topk_ratio), dim=-1)
        mask = torch.zeros_like(x)
        mask = mask.scatter(-1, ind, 1.)
        x = x*mask
        x = self.drop(x)
        return x


class TopkLrMlp(nn.Module):
    """ newMlp, similar to Mlp layers in MLP-Mixer but with extra gelu act and low-rank
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., window_size=3, topk_ratio=0.1, rank_ratio=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = TopkLrLinear(in_features, hidden_features, bias=True, window_size=window_size, topk_ratio=topk_ratio, rank_ratio=rank_ratio)
        self.act = act_layer()
        self.fc2 = TopkLrLinear(hidden_features, out_features, bias=True, window_size=window_size, topk_ratio=topk_ratio, rank_ratio=rank_ratio)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TopkMlp(nn.Module):
    """ newMlp, similar to Mlp layers in MLP-Mixer but with extra gelu act and low-rank
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., window_size=3, topk_ratio=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = TopkLinear(in_features, hidden_features, bias=True, window_size=window_size, topk_ratio=topk_ratio)
        self.act = act_layer()
        self.fc2 = TopkLinear(hidden_features, out_features, bias=True, window_size=window_size, topk_ratio=topk_ratio)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SLXMlp(nn.Module):
    """ newMlp, similar to Mlp layers in MLP-Mixer but with extra gelu act and low-rank
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., window_size=3, step=1, stripes_1=3, stripes_2=1, rank_ratio=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = SLXLinear(in_features, hidden_features, bias=True, window_size=window_size, stripes=stripes_1, step=step, rank_ratio=rank_ratio)
        self.act = act_layer()
        self.fc2 = SLXLinear(hidden_features, out_features, bias=True, window_size=window_size, stripes=stripes_2, step=step, rank_ratio=rank_ratio)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SLMlp(nn.Module):
    """ newMlp, similar to Mlp layers in MLP-Mixer but with extra gelu act and low-rank
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., window_size=3, step=1, stripes_1=3, stripes_2=1, rank_ratio=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = SLLinear(in_features, hidden_features, bias=True, window_size=window_size, stripes=stripes_1, step=step, rank_ratio=rank_ratio)
        self.act = act_layer()
        self.fc2 = SLLinear(hidden_features, out_features, bias=True, window_size=window_size, stripes=stripes_2, step=step, rank_ratio=rank_ratio)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RandomMlp(nn.Module):
    """ newMlp, similar to Mlp layers in MLP-Mixer but with extra gelu act and low-rank
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., window_size=1, sparse_ratio=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = RandomLinear(in_features, hidden_features, bias=True, window_size=window_size, sparse_ratio=sparse_ratio)
        self.act = act_layer()
        self.fc2 = RandomLinear(hidden_features, out_features, bias=True, window_size=window_size, sparse_ratio=sparse_ratio)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ButterflyMlp(nn.Module):
    """ newMlp, similar to Mlp layers in MLP-Mixer but with extra gelu act and low-rank
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., window_size=3, step=1, stripes_1=3, stripes_2=1, block_size=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ButterflyLinear(in_features, hidden_features, bias=True, window_size=window_size, stripes=stripes_1, step=step, block_size=block_size)
        self.act = act_layer()
        self.fc2 = ButterflyLinear(hidden_features, out_features, bias=True, window_size=window_size, stripes=stripes_2, step=step, block_size=block_size)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FastMlp(nn.Module):
    """ FastMlp, two low_rank factors for one linear layer
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, rank_ratio=0.1, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = FastLinear(in_features, hidden_features, rank_ratio=rank_ratio)
        self.act = act_layer()
        self.fc2 = FastLinear(hidden_features, out_features, rank_ratio=rank_ratio)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

