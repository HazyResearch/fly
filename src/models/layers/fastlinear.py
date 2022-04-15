from typing import Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import Linear, init

from einops import rearrange

import hydra

from src.ops.low_rank import low_rank_project
from src.ops.blockdiag_butterfly_einsum import (
    blockdiag_butterfly_multiply_einsum, blockdiag_butterfly_project_einsum
)
from src.utils.utils import get_logger
logger = get_logger()


class BlockdiagButterflyLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, nblocks1: int = 4, nblocks2: int = 4,
                 b1: int = 48, b2: int = 1, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nblocks1 = nblocks1
        self.nblocks2 = nblocks2

        m, n = out_features, in_features
        i = n//nblocks1
        l = m//nblocks2
        assert n == i * nblocks1
        assert m == l * nblocks2
        self.w1_bfly = Parameter(torch.empty((nblocks1, nblocks2*b1, i), **factory_kwargs))
        self.w2_bfly = Parameter(torch.empty((nblocks2, l, nblocks1*b1), **factory_kwargs))
        self.b1 = b1
        self.b2 = b2
        self.saving = ((torch.numel(self.w1_bfly)+torch.numel(self.w2_bfly)))/(self.in_features*self.out_features)

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def init_factors(self, weight):
        self.w1_bfly.data, self.w2_bfly.data = blockdiag_butterfly_project_einsum(weight, nblocks1=self.nblocks1,
                                                nblocks2=self.nblocks2, b1=self.b1, b2=self.b2)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.w1_bfly, a=math.sqrt(5))
        init.kaiming_uniform_(self.w2_bfly, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.w1_bfly)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def preprocess(self, x):
        return x.reshape(-1, x.shape[-1])

    def postprocess(self, output, x_shape):
        batch_shape = x_shape[:-1]
        return output.reshape(batch_shape + (output.shape[-1],))

    def forward(self, input: Tensor) -> Tensor:
        x_shape = input.shape
        output = blockdiag_butterfly_multiply_einsum(self.preprocess(input), self.w1_bfly, self.w2_bfly, self.b2)
        output = self.postprocess(output, x_shape)
        return (output + self.bias) if self.bias is not None else output

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class NinjaTurtleProjLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, repeat: int = 3, window_size: int = 6,
                 stripes: int = 3, step=1, gtoken=1, block_size=None, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        pseudo_mask_size = (min(in_features, out_features//repeat), max(in_features, out_features//repeat))
        tmp_mask = torch.zeros(pseudo_mask_size)
        stride = int(math.sqrt(pseudo_mask_size[0]))
        d = pseudo_mask_size[1] / pseudo_mask_size[0]

        if (math.ceil(d) - d) < 0.1:
            d = math.ceil(d)
        elif (d - math.floor(d)) < 0.1:
            d = math.floor(d)

        for k in range(stripes):
            patch_start = stride * (2 ** k - 1)
            for i in range(0, pseudo_mask_size[0], window_size):
                tmp_mask[patch_start + i:patch_start + i + window_size * step,
                int(i * d): int(i * d + step * d * window_size)] = 1.

        for k in range(stripes):
            patch_start = stride * (2 ** k - 1)
            for i in range(0, pseudo_mask_size[0], window_size):
                tmp_mask[i: i + window_size * step,
                int((i + patch_start) * d): int((patch_start + i) * d + step * d * window_size)] = 1.
        tmp_mask = generate_mask(tmp_mask, block_size)
        tmp_mask[:, :gtoken] = 1.
        tmp_mask[:gtoken, :] = 1.

        if in_features <= out_features//repeat:
            self.register_buffer('sparse_mask', tmp_mask.t().repeat(1, repeat))
        else:
            self.register_buffer('sparse_mask', tmp_mask.repeat(1, repeat))

        self.saving = torch.sum(tmp_mask) / (self.in_features * self.out_features)

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.data = self.weight.data / math.sqrt(
            torch.sum(self.sparse_mask) / (self.in_features * self.out_features))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return input @ ((self.sparse_mask * self.weight).t()) + self.bias

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class NinjaTurtleLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, window_size: int = 6,
                 stripes: int = 3, step=1, gtoken=1, block_size=None, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        pseudo_mask_size = (min(in_features, out_features), max(in_features, out_features))
        tmp_mask = torch.zeros(pseudo_mask_size)
        stride = int(math.sqrt(pseudo_mask_size[0]))
        d = pseudo_mask_size[1] / pseudo_mask_size[0]

        # BC if we want to use 32/16 blocks for speed, modify this
        if (math.ceil(d) - d) < 0.1:
            d = math.ceil(d)
        elif (d - math.floor(d)) < 0.1:
            d = math.floor(d)

        for k in range(stripes):
            patch_start = stride * (2 ** k - 1)
            for i in range(0, pseudo_mask_size[0], window_size):
                tmp_mask[patch_start + i:patch_start + i + window_size * step,
                int(i * d): int(i * d + step * d * window_size)] = 1.

        for k in range(stripes):
            patch_start = stride * (2 ** k - 1)
            for i in range(0, pseudo_mask_size[0], window_size):
                tmp_mask[i: i + window_size * step,
                int((i + patch_start) * d): int((patch_start + i) * d + step * d * window_size)] = 1.
        tmp_mask = generate_mask(tmp_mask, block_size)
        tmp_mask[:, :gtoken] = 1.
        tmp_mask[:gtoken, :] = 1.

        if in_features <= out_features:
            self.register_buffer('sparse_mask', tmp_mask.t())
        else:
            self.register_buffer('sparse_mask', tmp_mask)

        self.saving = torch.sum(tmp_mask) / (self.in_features * self.out_features)

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.data = self.weight.data / math.sqrt(
            torch.sum(self.sparse_mask) / (self.in_features * self.out_features))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return input @ ((self.sparse_mask * self.weight).t()) + self.bias

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class ButterflyGlobalLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, window_size: int = 6,
                 stripes: int = 3, step=1, gtoken=1, block_size=None, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        pseudo_mask_size = (min(in_features, out_features), max(in_features, out_features))
        tmp_mask = torch.zeros(pseudo_mask_size)
        stride = int(math.sqrt(pseudo_mask_size[0]))
        d = math.ceil(pseudo_mask_size[1] / pseudo_mask_size[0])

        for k in range(stripes):
            patch_start = stride * k
            for i in range(0, pseudo_mask_size[0], window_size):
                tmp_mask[patch_start + i:patch_start + i + window_size, i * d: i * d + step * d * window_size] = 1.

        for k in range(stripes):
            patch_start = stride * k
            for i in range(0, pseudo_mask_size[0], window_size):
                tmp_mask[i: i + window_size, (i + patch_start) * d: (patch_start + i) * d + step * d * window_size] = 1.
        tmp_mask = generate_mask(tmp_mask, block_size)
        tmp_mask[:, :gtoken] = 1.
        tmp_mask[:gtoken, :] = 1.

        if in_features <= out_features:
            self.register_buffer('sparse_mask', tmp_mask.t())
        else:
            self.register_buffer('sparse_mask', tmp_mask)

        self.saving = torch.sum(generate_mask(tmp_mask))/(self.in_features*self.out_features)

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.data = self.weight.data/math.sqrt(torch.sum(self.sparse_mask)/(self.in_features*self.out_features))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return input @ ((self.sparse_mask*self.weight).t()) + self.bias

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class NinjaTurtleSparsityConfig:

    linear_cls = NinjaTurtleLinear

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.block = kwargs.pop('block')

    def make_layout(self, out_features, in_features):
        linear = self.linear_cls(in_features, out_features, bias=False, **self.kwargs)
        layout = linear.sparse_mask
        # Convert from (out_features, in_features) mask to
        # (out_features // block_size, in_features // block_size) mask
        layout = rearrange(layout, '(p blksz) (r blksz1) -> p r (blksz blksz1)',
                           blksz=self.block, blksz1=self.block)
        return (layout > 0).any(dim=-1).int()


class ButterflyGlobalSparsityConfig(NinjaTurtleSparsityConfig):
    linear_cls = ButterflyGlobalLinear


class TopkLrLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = False, rank_ratio: float = 0.1,
                 window_size: int = 6, topk_ratio: float = 0.1, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gate = Linear(in_features=in_features, out_features=1)

        self.sparse = TopkLinear(in_features, out_features, window_size=window_size, topk_ratio=topk_ratio)
        self.low_rank = FastLinear(in_features, out_features, rank_ratio=rank_ratio)
        self.saving = self.sparse.saving + self.low_rank.saving

    def forward(self, input: Tensor) -> Tensor:
        g = self.gate(input.detach())
        g = torch.sigmoid(g)
        sparse_comp = self.sparse(input)
        low_rank_comp = self.low_rank(input)
        return g * sparse_comp + (1. - g) * low_rank_comp

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class TopkLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, window_size: int = 1,
                 topk_ratio: float = 0.1, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.window_size = window_size
        self.topk_ratio = topk_ratio
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        tmp_mask = self.reset_parameters()

        if in_features <= out_features:
            self.register_buffer('sparse_mask', tmp_mask.t())
        else:
            self.register_buffer('sparse_mask', tmp_mask)

        self.saving = torch.sum(tmp_mask) / (self.in_features * self.out_features)

    def init_mask(self, weight):
        pseudo_mask_size = (min(self.in_features, self.out_features), max(self.in_features, self.out_features))
        d = math.ceil(pseudo_mask_size[1] / pseudo_mask_size[0])
        x = math.ceil(pseudo_mask_size[0] / self.window_size)
        y = math.ceil(math.ceil(pseudo_mask_size[1] / d) / self.window_size)
        blocks = x * y
        topk_blocks = math.ceil(blocks * self.topk_ratio)

        kernel = torch.nn.AvgPool2d((self.window_size, self.window_size * d),
                                    stride=(self.window_size, self.window_size * d), ceil_mode=True)
        if self.in_features <= self.out_features:
            value, ind = torch.topk(kernel(torch.abs(weight.t()[None, None])).view(-1), k=topk_blocks)
        else:
            value, ind = torch.topk(kernel(torch.abs(weight[None, None])).view(-1), k=topk_blocks)
        base = torch.zeros([blocks, 1], device=weight.device)
        base[ind] = 1.
        base = torch.repeat_interleave(base.view(x, y), self.window_size * d).view(x, y * self.window_size * d)
        tmp_mask = torch.repeat_interleave(base, self.window_size, dim=0)
        return tmp_mask[:pseudo_mask_size[0], :pseudo_mask_size[1]]

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        tmp_mask = self.init_mask(self.weight)
        self.weight.data = self.weight.data / math.sqrt(torch.sum(tmp_mask) / (self.in_features * self.out_features))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
        return tmp_mask

    def forward(self, input: Tensor) -> Tensor:
        tmp_mask = self.init_mask(self.weight)
        if self.in_features <= self.out_features:
            tmp_mask = tmp_mask.t()
        return input @ ((tmp_mask * self.weight).t()) + self.bias

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class SLXLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, rank_ratio: float = 0.1,
                 window_size: int = 6, stripes: int = 3, step=1, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gate = Linear(in_features=in_features, out_features=1)
        self.rank = int(rank_ratio*min(in_features, out_features))

        self.weight1 = Parameter(torch.empty((out_features, min(in_features, out_features)), **factory_kwargs))
        self.weight2 = Parameter(torch.empty((min(in_features, out_features), in_features), **factory_kwargs))

        pseudo_mask_size = (min(in_features, out_features), max(in_features, out_features))
        tmp_mask = torch.zeros(pseudo_mask_size)
        stride = int(math.sqrt(pseudo_mask_size[0]))
        d = math.ceil(pseudo_mask_size[1] / pseudo_mask_size[0])

        for k in range(stripes):
            patch_start = stride * k
            for i in range(0, pseudo_mask_size[0], window_size):
                tmp_mask[patch_start + i:patch_start + i + window_size, i * d: i * d + step * d * window_size] = 1.

        for k in range(stripes):
            patch_start = stride * k
            for i in range(0, pseudo_mask_size[0], window_size):
                tmp_mask[i: i + window_size, (i + patch_start) * d: (patch_start + i) * d + step * d * window_size] = 1.

        if in_features <= out_features:
            self.register_buffer('sparse_mask', tmp_mask.t())
        else:
            self.register_buffer('sparse_mask', tmp_mask)

        self.saving = ((self.in_features+self.out_features)*self.rank)/(self.in_features*self.out_features) \
                      + torch.sum(tmp_mask)/(in_features*out_features)
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def forward(self, input: Tensor) -> Tensor:
        g = self.gate(input)
        g = torch.sigmoid(g)
        attn = torch.einsum("od,di->st", self.weight1, self.weight2)
        attn = attn*self.sparse_mask
        sparse_comp = input @ (attn.t())
        low_rank_comp = input @ (self.weight2.t()[:, :self.rank]) @ (self.weight1.t()[:self.rank, :])
        return g * sparse_comp + (1.-g) * low_rank_comp + self.bias

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class SLLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = False, rank_ratio: float = 0.1,
                 window_size: int = 6, stripes: int = 3, step=1, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gate = Linear(in_features=in_features, out_features=1)

        self.sparse = ButterflyLinear(in_features, out_features, window_size=window_size, stripes=stripes, step=step)
        self.low_rank = FastLinear(in_features, out_features, rank_ratio=rank_ratio)
        self.saving = self.sparse.saving + self.low_rank.saving

    def forward(self, input: Tensor) -> Tensor:
        g = self.gate(input)
        g = torch.sigmoid(g)
        sparse_comp = self.sparse(input)
        low_rank_comp = self.low_rank(input)
        return g * sparse_comp + (1.-g) * low_rank_comp

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


def generate_mask(base, block_size=None):
    if block_size is not None:
        num_r, num_c = base.shape
        b_r, b_c = block_size
        mask = torch.zeros(base.shape)
        for i in range(0, num_r, b_r):
            for j in range(0, num_c, b_c):
                lighten = torch.sum(base[i:(i+b_r), j:(j+b_c)])
                if lighten > 0.0:
                    mask[i:(i+b_r), j:(j+b_c)] = 1.
        return mask
    else:
        return base


class RandomLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, window_size: int = 1,
                 sparse_ratio: float = 0.1, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        pseudo_mask_size = (min(in_features, out_features), max(in_features, out_features))
        tmp_mask = torch.zeros(pseudo_mask_size)
        d = math.ceil(pseudo_mask_size[1] / pseudo_mask_size[0])
        x = math.ceil(pseudo_mask_size[0] / window_size)
        y = math.ceil(pseudo_mask_size[1] // d / window_size)
        blocks = x * y
        nnz_block = math.ceil(blocks * sparse_ratio)
        ind = torch.randperm(blocks)[:nnz_block]

        for k in range(nnz_block):
            block_x = ind[k] // y
            block_y = ind[k] % y
            tmp_mask[block_x * window_size:(block_x + 1) * window_size,
            block_y * window_size * d:(block_y + 1) * window_size * d] = 1.
        tmp_mask = tmp_mask.view(pseudo_mask_size[0], pseudo_mask_size[1])

        if in_features <= out_features:
            self.register_buffer('sparse_mask', tmp_mask.t())
        else:
            self.register_buffer('sparse_mask', tmp_mask)

        self.saving = torch.sum(tmp_mask) / (self.in_features * self.out_features)

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.data = self.weight.data / math.sqrt(
            torch.sum(self.sparse_mask) / (self.in_features * self.out_features))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return input @ ((self.sparse_mask * self.weight).t()) + self.bias

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class ButterflyLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, window_size: int = 6, stripes: int = 3, step = 1,
                 block_size=None, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        pseudo_mask_size = (min(in_features, out_features), max(in_features, out_features))
        tmp_mask = torch.zeros(pseudo_mask_size)
        stride = int(math.sqrt(pseudo_mask_size[0]))
        d = math.ceil(pseudo_mask_size[1] / pseudo_mask_size[0])

        for k in range(stripes):
            patch_start = stride * k
            for i in range(0, pseudo_mask_size[0], window_size):
                tmp_mask[patch_start + i:patch_start + i + window_size, i * d: i * d + step * d * window_size] = 1.

        for k in range(stripes):
            patch_start = stride * k
            for i in range(0, pseudo_mask_size[0], window_size):
                tmp_mask[i: i + window_size, (i + patch_start) * d: (patch_start + i) * d + step * d * window_size] = 1.
        tmp_mask = generate_mask(tmp_mask, block_size)

        if in_features <= out_features:
            self.register_buffer('sparse_mask', tmp_mask.t())
        else:
            self.register_buffer('sparse_mask', tmp_mask)

        self.saving = torch.sum(generate_mask(tmp_mask))/(self.in_features*self.out_features)

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.data = self.weight.data/math.sqrt(torch.sum(self.sparse_mask)/(self.in_features*self.out_features))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return input @ ((self.sparse_mask*self.weight).t()) + self.bias

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class FastLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, rank_ratio: float = 0.1,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = int(rank_ratio*min(in_features, out_features))
        self.low_rank1 = Parameter(torch.empty((in_features, self.rank), **factory_kwargs))
        self.low_rank2 = Parameter(torch.empty((self.rank, out_features), **factory_kwargs))
        self.saving = ((self.in_features+self.out_features)*self.rank)/(self.in_features*self.out_features)

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def init_low_rank(self, weight):
        u, s, v = torch.svd(weight.t())
        self.low_rank1.data = u[:, :self.rank] @ torch.diag(s[:self.rank])
        self.low_rank2.data = v[:, :self.rank].t()
        assert torch.norm(weight.t() - u @ torch.diag(s) @ v.t()) < 0.01

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.low_rank1, a=math.sqrt(5))
        init.kaiming_uniform_(self.low_rank2, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.low_rank1)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return (input @ self.low_rank1 @ self.low_rank2 + self.bias) if self.bias is not None else input @ self.low_rank1 @ self.low_rank2

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class LowRank(nn.Module):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, rank: Union[int, float],
                 bias: bool=True, init='linear', weight_decay: bool = True,
                 device=None, dtype=None) -> None:
        """
        weight_decay: whether to mark the low-rank weights as _no_weight_decay.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if isinstance(rank, float):
            rank = int(rank * min(in_features, out_features))
        self.rank = rank
        self.lr_weight1 = Parameter(torch.empty((self.rank, in_features), **factory_kwargs))
        self.lr_weight2 = Parameter(torch.empty((out_features, self.rank), **factory_kwargs))
        if init not in ['linear', 'svd']:
            raise NotImplementedError(f'init {init} not supported')
        self.init = init

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        if not weight_decay:
            self.lr_weight1._no_weight_decay = True
            self.lr_weight2._no_weight_decay = True

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.init == 'linear':
                # Mimic torch.nn.Linear init
                init.kaiming_uniform_(self.lr_weight1, a=math.sqrt(5))
                init.kaiming_uniform_(self.lr_weight2, a=math.sqrt(5))
            elif self.init == 'svd':
                # Use spectral initialization as described in https://openreview.net/forum?id=KTlJT1nof6d
                dense_init_fn_ = partial(init.kaiming_uniform_, a=math.sqrt(5))
                self.set_weights_from_dense_init(dense_init_fn_)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.lr_weight1)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def set_weights_from_projection(self, weight):
        U, Vt = low_rank_project(weight, rank=self.rank)
        with torch.no_grad():
            self.lr_weight1.copy_(Vt)
            self.lr_weight2.copy_(U)

    def set_weights_from_dense_init(self, dense_init_fn_):
        dense_weight = torch.empty(self.out_features, self.in_features,
                                   device=self.lr_weight1.device, dtype=self.lr_weight1.dtype)
        dense_init_fn_(dense_weight)
        self.set_weights_from_projection(dense_weight)

    @property
    def saving(self):
        return ((self.in_features + self.out_features) * self.rank
                / (self.in_features * self.out_features))

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(F.linear(input, self.lr_weight1), self.lr_weight2, self.bias)


class SparseLRLinear(nn.Module):
    def __init__(self, in_features, out_features, sparse_cfg,
                 bias=True, rank: Union[int, float] = 0.1,
                 gating=True, checkpointing=False):
        """If rank is float (e.g., 0.1), treat it as rank ratio.
        If rank is int (e.g., 32), treat it as rank.
        gating: whether to use sigmoid gating, otherwise we simply average the sparse and low-rank
        components.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparse = hydra.utils.instantiate(sparse_cfg, in_features, out_features, bias=False,
                                              _recursive_=False)
        self.low_rank = LowRank(in_features, out_features, rank=rank, bias=False)
        if gating:
            self.gate = nn.Linear(in_features=in_features, out_features=1)
        else:
            self.register_parameter('gate', None)
        self.checkpointing = checkpointing
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        logger.info(f'Linear class {self.__class__}: saving={self.saving}')

    def reset_parameters(self) -> None:
        if self.bias is not None:
            fan_in = self.bias.shape[0]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    @property
    def saving(self):
        return self.sparse.saving + self.low_rank.saving

    def _multiply(self, x):
        sparse_output = self.sparse(x)
        low_rank_output = self.low_rank(x)
        g = torch.sigmoid(self.gate(x)) if self.gate is not None else 0.5
        # output = (1.0 - g) * sparse_output + g * low_rank_output
        return torch.lerp(sparse_output, low_rank_output, g)

    def forward(self, x):
        if self.checkpointing:
            output = torch.utils.checkpoint.checkpoint(self._multiply, x)
        else:
            output = self._multiply(x)
        # Convert bias to output.dtype in case of AMP, otherwise bias and activation will be in FP32
        return (output + self.bias.to(dtype=output.dtype)) if self.bias is not None else output
