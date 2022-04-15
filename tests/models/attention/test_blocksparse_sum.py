import torch
import triton
import pytest

from src.models.attention.blocksparse_sum import blocksparse_sum
from src.models.attention.blocksparse_utils import sparsify_tensor, mask_tensor


@pytest.mark.parametrize(
    "BLOCK, WIDTH",
    [(block, width) for block in [16, 32] for width in [256, 576, 1024, 1792]],
)
def test_logsumexp(BLOCK, WIDTH, DTYPE=torch.float32):
    # set seed
    torch.random.manual_seed(0)
    Z, H, M, N = 2, 4, WIDTH, WIDTH
    scale = 0.4
    # create inputs
    layout = torch.randint(2, (H, M // BLOCK, N // BLOCK))
    x = torch.randn((Z, H, M, N), dtype=DTYPE, requires_grad=True, device="cuda")
    # triton result
    op = blocksparse_sum(layout, BLOCK)
    tx = sparsify_tensor(x, layout)
    ty = op(tx * scale)
    grad = torch.randn_like(ty)
    tdx, = torch.autograd.grad(ty, x, grad)
    # torch result
    rx = mask_tensor(x, layout, value=0)
    ry = torch.sum(rx * scale, -1)
    rdx, = torch.autograd.grad(ry, x, grad)
    # compare
    assert torch.allclose(ry, ty, rtol=1e-4, atol=1e-5)
    assert torch.allclose(rdx, tdx)
