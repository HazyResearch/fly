import math

import torch
import pytest

from einops import rearrange

from src.ops.blockdiag_multiply import blockdiag_multiply_reference, blockdiag_multiply


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_blockdiag_multiply(device, dtype):
    # set seed
    torch.random.manual_seed(0)
    n = 768
    batch_size = 3
    x = torch.randn(batch_size, n, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(8, 96 * 2, 96, device=x.device, dtype=x.dtype, requires_grad=True)
    out = blockdiag_multiply(x, weight)
    grad = torch.randn_like(out)
    dx, dweight = torch.autograd.grad(out, (x, weight), grad, retain_graph=True)
    assert out.shape == (batch_size, 8 * 96 * 2)
    out_ref = blockdiag_multiply_reference(x, weight)
    dx_ref, dweight_bfly = torch.autograd.grad(out_ref, (x, weight), grad, retain_graph=True)
    assert torch.allclose(out, out_ref, rtol=1e-4, atol=1e-4)
    assert torch.allclose(dx, dx_ref, rtol=1e-4, atol=1e-4)
    assert torch.allclose(dweight, dweight_bfly, rtol=1e-4, atol=1e-4)
