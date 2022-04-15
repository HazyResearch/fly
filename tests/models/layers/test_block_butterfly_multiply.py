import math

import torch
import pytest

from src.models.layers.block_butterfly_multiply import block_butterfly_multiply
from src.models.layers.block_butterfly_multiply import block_butterfly_factor_multiply


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('nstacks', [1, 2, 3])
@pytest.mark.parametrize('nblocks', [1, 2, 3])
@pytest.mark.parametrize('increasing_stride', [True, False])
@pytest.mark.parametrize('log_n', [3, 4, 5])
@pytest.mark.parametrize('block_size', [1, 2, 4, 6])
def test_block_butterfly_multiply(block_size, log_n, increasing_stride, nblocks, nstacks, device):
    # set seed
    torch.random.manual_seed(0)
    n = 1 << log_n
    batch_size = 3
    twiddle = torch.randn(nstacks, nblocks, log_n, n // 2, 2, 2, block_size, block_size,
                          device=device)
    input = torch.randn(batch_size, nstacks, block_size * n, device=device)
    increasing_stride = True
    output_size = None
    output = block_butterfly_multiply(twiddle, input, increasing_stride=increasing_stride,
                                      output_size=output_size)
    assert output.shape == (batch_size, nstacks, block_size * n)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('nstacks', [1, 2, 3])
@pytest.mark.parametrize('increasing_stride', [True, False])
@pytest.mark.parametrize('log_n', [3, 4, 5])
@pytest.mark.parametrize('block_size', [1, 2, 4, 6])
def test_block_butterfly_factor_multiply(block_size, log_n, increasing_stride, nstacks, device):
    # set seed
    torch.random.manual_seed(0)
    nblocks = 1
    n = 8
    batch_size = 3
    log_n = int(math.log2(n))
    twiddle = torch.randn(nstacks, nblocks, log_n, n // 2, 2, 2, block_size, block_size,
                          device=device)
    input = torch.randn(batch_size, nstacks, block_size * n, device=device)
    increasing_stride = True
    output = block_butterfly_multiply(twiddle, input, increasing_stride=increasing_stride)
    assert output.shape == (batch_size, nstacks, block_size * n)
    output_factor = input
    for idx in range(log_n):
        output_factor = block_butterfly_factor_multiply(twiddle[:, 0], output_factor, idx,
                                                        increasing_stride=increasing_stride)
    assert torch.allclose(output, output_factor)
