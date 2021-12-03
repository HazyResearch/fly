import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import Linear, init


class MaskLinear(nn.Module):
    r"""
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, topk_ratio: float = 0.1, window_size: int = 6,
                 stripes: int = 3, step = 1, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MaskLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        # Butterfly base
        pseudo_mask_size = (min(in_features, out_features), max(in_features, out_features))
        tmp_mask = torch.zeros(pseudo_mask_size)
        stride = int(math.sqrt(pseudo_mask_size[0]))
        d = math.ceil(pseudo_mask_size[1] / pseudo_mask_size[0])

        for k in range(stripes):
            patch_start = stride * k
            for i in range(0, pseudo_mask_size[0], window_size):
                tmp_mask[patch_start + i:patch_start + i + window_size, i * d: i * d + step * d * window_size] = 1

        for k in range(stripes):
            patch_start = stride * k
            for i in range(0, pseudo_mask_size[0], window_size):
                tmp_mask[i: i + window_size, (i + patch_start) * d: (patch_start + i) * d + step * d * window_size] = 1

        if in_features <= out_features:
            self.register_buffer('sparse_mask', tmp_mask.t())
        else:
            self.register_buffer('sparse_mask', tmp_mask)

        self.register_buffer('sparse_mask_topk', torch.zeros_like(self.weight))

        self.reset_parameters()
        self.input = None
        self.topk = math.ceil(in_features*out_features * topk_ratio)
        self.saving = topk_ratio

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.data = self.weight.data / math.sqrt(
            torch.sum(self.sparse_mask) / (self.in_features * self.out_features))

    def forward(self, input: Tensor) -> Tensor:
        self.input = input
        mask = (self.sparse_mask_topk.bool() | self.sparse_mask.bool()).int()
        y = input @ ((mask * self.weight).t())
        return y

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features is not None
        )


def hook_fn(module, input, output):
    tmp = output[0].float()
    dense_grad = torch.einsum('bsd,bsm->bdm', tmp, module.input)
    dense_grad = torch.sum(dense_grad, dim=0)
    # print(torch.sum(dense_grad==module.weight.grad))
    # BC: first try matrix wise topk
    tmp = torch.abs(dense_grad).flatten()
    _, idx = torch.topk(tmp, module.topk)
    mask = torch.zeros_like(tmp, device=module.sparse_mask.device)
    mask = mask.scatter(-1, idx, 1.)
    mask = mask.reshape(dense_grad.shape)
    module.sparse_mask_topk = mask
    # print("replace mask")


class MaskLinearWrap(nn.Module):
    """ Sanity check if topk work for activation
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, topk_ratio: float = 0.1,
                 window_size: int = 6, stripes: int = 3, step = 1, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MaskLinearWrap, self).__init__()
        self.fc = MaskLinear(in_features, out_features, topk_ratio=topk_ratio, window_size=window_size,
                            stripes=stripes, step=step, device=device, dtype=device)
        self.fc.register_full_backward_hook(hook_fn)
        self.saving = topk_ratio
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.fc.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)


    def forward(self, input: Tensor) -> Tensor:
        self.input = input
        y = self.fc(input)+self.bias
        return y


# Tests:
if __name__ == '__main__':
    x = torch.randn((2, 10, 20), requires_grad=True)
    layer = MaskLinearWrap(20, 40, topk_ratio=0.1, bias=True)
    loss = 1-torch.sum(layer(x))
    loss.backward()

    print(torch.sum(layer.fc.sparse_mask), torch.sum(layer.fc.sparse_mask_topk))
