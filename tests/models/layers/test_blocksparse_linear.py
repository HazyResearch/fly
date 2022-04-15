import math
import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from deepspeed.ops.sparse_attention import FixedSparsityConfig

from src.models.layers.blocksparse_linear import BlockSparseLinear
from src.models.layers.fastlinear import NinjaTurtleLinear, ButterflyGlobalLinear
from src.models.attention.blocksparse_utils import sparsify_tensor, densify_tensor
from src.utils.padding import pad_to_multiple
from src.models.attention.blocksparse_matmul import matmul


def seed_cpu_cuda(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def setup(batch_size, in_features, out_features, block_size):
    x = torch.randn(batch_size, in_features, requires_grad=True, device='cuda')
    kwargs = dict(window_size=block_size, stripes=1, step=2, gtoken=block_size)
    sparsity_config = dict(
        _target_='src.models.layers.fastlinear.NinjaTurtleSparsityConfig',
        block=block_size,
        **kwargs
    )
    bs_linear = BlockSparseLinear(in_features, out_features, sparsity_config, bias=True).to('cuda')
    bs_fake_linear = NinjaTurtleLinear(in_features, out_features, bias=True, **kwargs).to('cuda')
    return x, bs_linear, bs_fake_linear


class TestBlockSparseLinear:

    @pytest.mark.parametrize('block_size', [16, 32])
    @pytest.mark.parametrize('out_features', [698, 3081])
    @pytest.mark.parametrize('in_features', [497, 149])
    def test_init(self, in_features, out_features, block_size):
        batch_size = 128
        x, bs_linear, bs_fake_linear = setup(batch_size, in_features, out_features, block_size)
        assert torch.allclose(bs_linear.weight.mean(), bs_fake_linear.weight.mean(), atol=1e-3)
        assert torch.allclose(bs_linear.weight.std(), bs_fake_linear.weight.std(), atol=1e-2)
        assert torch.allclose(bs_linear.bias.mean(), bs_fake_linear.bias.mean(), atol=1e-2)
        assert torch.allclose(bs_linear.bias.std(), bs_fake_linear.bias.std(), atol=1e-2)
        output = bs_linear.to('cuda')(x) - bs_linear.bias
        assert output.mean().abs().item() < 1e-2
        assert 0.3 < output.std().item() < 3.0

    @pytest.mark.parametrize('out_features', [698, 3081])
    @pytest.mark.parametrize('in_features', [497, 149])
    def test_backends(self, in_features, out_features):
        """The two backends (huggingface and triton) should yield the same output and gradients.
        """
        block_size = 32
        batch_size = 128
        x = torch.randn(batch_size, in_features, requires_grad=True, device='cuda')
        kwargs = dict(window_size=block_size, stripes=1, step=2, gtoken=block_size)
        sparsity_config = dict(
            _target_='src.models.layers.fastlinear.NinjaTurtleSparsityConfig',
            block=block_size,
            **kwargs
        )
        bs_linear_triton = BlockSparseLinear(in_features, out_features, sparsity_config, bias=True,
                                             backend='triton').to('cuda')
        bs_linear_hf = BlockSparseLinear(in_features, out_features, sparsity_config, bias=True,
                                         backend='huggingface').to('cuda')
        with torch.no_grad():
            bs_linear_hf.weight.copy_(rearrange(bs_linear_triton.weight,
                                                '1 nnz blksz blksz1 -> (nnz blksz1) blksz'))
            bs_linear_hf.bias.copy_(bs_linear_triton.bias)
        out_triton = bs_linear_triton(x)
        grad = torch.randn_like(out_triton)
        grad_x_triton, grad_weight_triton = torch.autograd.grad(out_triton,
                                                                (x, bs_linear_triton.weight), grad)
        x = x.clone().detach().requires_grad_(True)
        out_hf = bs_linear_hf(x)
        grad_x_hf, grad_weight_hf = torch.autograd.grad(out_hf, (x, bs_linear_hf.weight), grad)
        assert torch.allclose(out_triton, out_hf, rtol=1e-5, atol=1e-6)
        assert torch.allclose(grad_x_triton, grad_x_hf, rtol=1e-4, atol=1e-5)
        assert torch.allclose(rearrange(grad_weight_triton,
                                        '1 nnz blksz blksz1 -> (nnz blksz1) blksz'),
                              grad_weight_hf, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize('block_size', [16, 32])
    @pytest.mark.parametrize('out_features', [698, 1280, 3081])
    @pytest.mark.parametrize('in_features', [640, 497, 149])
    def test_output(self, in_features, out_features, block_size):
        """With the same weight, the fast implementation (BlockSparseLinear) should yield the same
        output and gradient as the slow implementation.
        """
        batch_size = 128
        x, bs_linear, bs_fake_linear = setup(batch_size, in_features, out_features, block_size)
        with torch.no_grad():
            if in_features % block_size != 0 or out_features % block_size != 0:
                # Make sparse_mask block-aligned in these cases
                sparse_mask = bs_linear.layout
                sparse_mask = repeat(sparse_mask, 'p r -> (p blksz) (r blksz1)',
                                     blksz=block_size, blksz1=block_size)
                sparse_mask = sparse_mask[:out_features, :in_features]
                bs_fake_linear.sparse_mask = sparse_mask
            weight_dense = pad_to_multiple(bs_fake_linear.weight, multiple=block_size, dims=(0, 1))
            weight_sparse = sparsify_tensor(rearrange(weight_dense, 'd2 d1 -> 1 d2 d1'),
                                            bs_linear.layout)
            layout = rearrange(bs_linear.layout, 'd1 d2 -> 1 d1 d2')
            assert torch.allclose(densify_tensor(weight_sparse,
                                                 layout)[:, :, :out_features, :in_features]
                                  * bs_fake_linear.sparse_mask,
                                  bs_fake_linear.weight * bs_fake_linear.sparse_mask)
            if bs_linear.backend == 'triton':
                bs_linear.weight.copy_(weight_sparse)
            elif bs_linear.backend == 'huggingface':
                bs_linear.weight.copy_(rearrange(weight_sparse,
                                                 '1 nnz blksz blksz1 -> (nnz blksz1) blksz'))
            bs_linear.bias.copy_(bs_fake_linear.bias)
        out = bs_linear(x)
        grad = torch.randn_like(out)
        grad_x, grad_weight = torch.autograd.grad(out, (x, bs_linear.weight), grad)
        x = x.clone().detach().requires_grad_(True)
        out_slow = bs_fake_linear(x)
        grad_x_slow, grad_weight_slow = torch.autograd.grad(out_slow, (x, bs_fake_linear.weight),
                                                            grad)
        assert torch.allclose(out, out_slow, rtol=1e-4, atol=1e-5)
        assert torch.allclose(grad_x, grad_x_slow, rtol=1e-4, atol=1e-5)
        if bs_linear.backend == 'huggingface':
            grad_weight = rearrange(grad_weight, '(nnz blksz1) blksz -> 1 nnz blksz blksz1',
                                    blksz=block_size, blksz1=block_size)
        grad_weight_dense = densify_tensor(grad_weight, layout)
        assert torch.allclose(grad_weight_dense,
                              pad_to_multiple(grad_weight_slow, multiple=block_size, dims=(0, 1)),
                              rtol=1e-4, atol=1e-5)
