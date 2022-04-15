import math

import torch
import pytest

from einops import rearrange

from src.models.layers.blockdiag_butterfly_multiply import blockdiag_butterfly_multiply
from src.ops.blockdiag_butterfly_projection import blockdiag_butterfly_project, factors
from src.ops.blockdiag_butterfly_projection import ButterflyFFT, ButterflyFFT2
# from src.ops.permutation import bitreversal_permutation


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('log_n', [2, 4, 10, 12])
def test_block_diag_butterfly_project_sqrtn(log_n, device):
    # set seed
    torch.random.manual_seed(0)
    n = 1 << log_n
    sqrtn = 1 << (log_n // 2)
    x = torch.eye(n, device=device)
    w1_bfly = torch.randn(sqrtn, sqrtn, sqrtn, device=x.device, dtype=x.dtype, requires_grad=True)
    w2_bfly = torch.randn(sqrtn, sqrtn, sqrtn, device=x.device, dtype=x.dtype, requires_grad=True)
    bfly = blockdiag_butterfly_multiply(x, w1_bfly, w2_bfly).t()
    w1_bfly_projected, w2_bfly_projected = blockdiag_butterfly_project(bfly)
    bfly_projected = blockdiag_butterfly_multiply(x, w1_bfly_projected, w2_bfly_projected).t()
    print((bfly_projected - bfly).abs().max())
    assert torch.allclose(bfly_projected, bfly, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize('direction', ['fft', 'ifft'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('log_n', [2, 4, 10])
def test_block_diag_butterfly_project_fft_sqrtn(log_n, device, direction):
    # set seed
    torch.random.manual_seed(0)
    n = 1 << log_n
    sqrtn = 1 << (log_n // 2)
    batch_size = 3
    eye = torch.eye(n, dtype=torch.complex64, device=device)
    transform = torch.fft.fft if direction == 'fft' else torch.fft.ifft
    dft = transform(eye, norm='ortho').t()
    # perm = bitreversal_permutation(n)
    # We don't actually need the bitreversal permutation, any permutation that swap
    # the axes of the sqrtn x sqrtn input will work.
    perm = rearrange(torch.arange(n, device=device), '(i j) -> (j i)', i=sqrtn)
    # The BP (butterfly - permutation) decomposition of FFT / iFFT
    # Converting to complex128 makes the approximation an order of magnitude more accurate
    w1_fft_projected, w2_fft_projected = blockdiag_butterfly_project(dft[:, perm].cdouble())
    w1_fft_projected, w2_fft_projected = w1_fft_projected.cfloat(), w2_fft_projected.cfloat()
    fft_projected = blockdiag_butterfly_multiply(eye, w1_fft_projected, w2_fft_projected).t()
    print((fft_projected - dft[:, perm]).abs().max())
    assert torch.allclose(fft_projected, dft[:, perm], rtol=1e-4, atol=1e-4)
    x = torch.randn(batch_size, n, dtype=torch.complex64, device=device)
    out_fft = transform(x, norm='ortho')
    out = blockdiag_butterfly_multiply(x[:, perm], w1_fft_projected, w2_fft_projected)
    assert torch.allclose(out, out_fft, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize('direction', ['fft', 'ifft'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('norm', ['ortho', None])
@pytest.mark.parametrize('n', [15, 36, 196, 27, 48, 42, 85, 168, 512])
def test_block_diag_butterfly_project_fft_rectangular(n, norm, device, direction):
    # set seed
    torch.random.manual_seed(0)
    batch_size = 3
    eye = torch.eye(n, dtype=torch.complex64, device=device)
    transform = torch.fft.fft if direction == 'fft' else torch.fft.ifft
    dft = transform(eye, norm=norm).t()
    sizes = factors(n)[-1]
    sizes = (sizes[1], sizes[0])
    perm = rearrange(torch.arange(n, device=device), '(i j) -> (j i)', j=sizes[0])
    # The BP (butterfly - permutation) decomposition of FFT / iFFT
    # Converting to complex128 makes the approximation an order of magnitude more accurate
    w1_fft_projected, w2_fft_projected = blockdiag_butterfly_project(dft[:, perm].cdouble(),
                                                                     sizes=sizes)
    w1_fft_projected, w2_fft_projected = w1_fft_projected.cfloat(), w2_fft_projected.cfloat()
    fft_projected = blockdiag_butterfly_multiply(eye, w1_fft_projected, w2_fft_projected).t()
    print((fft_projected - dft[:, perm]).abs().max())
    assert torch.allclose(fft_projected, dft[:, perm], rtol=1e-4, atol=1e-4)
    x = torch.randn(batch_size, n, dtype=torch.complex64, device=device)
    out_fft = transform(x, norm=norm)
    out = blockdiag_butterfly_multiply(x[:, perm], w1_fft_projected, w2_fft_projected)
    assert torch.allclose(out, out_fft, rtol=1e-4, atol=1e-4)

    bfly_fft = ButterflyFFT(n, direction=direction, norm=norm).to(device=device)
    out_module = bfly_fft(x)
    assert torch.allclose(out_module, out_fft, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize('direction', ['fft', 'ifft'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('norm', ['ortho', None])
@pytest.mark.parametrize('n2', [85, 512])
@pytest.mark.parametrize('n1', [42, 160, 161])
def test_butterflyfft2(n1, n2, norm, device, direction):
    # set seed
    torch.random.manual_seed(0)
    batch_size = 3
    x = torch.randn(batch_size, n1, n2, dtype=torch.complex64, device=device)
    transform = torch.fft.fft2 if direction == 'fft' else torch.fft.ifft2
    out_fft = transform(x, norm=norm)
    bfly_fft = ButterflyFFT2(n1, n2, direction=direction, norm=norm).to(device=device)
    out = bfly_fft(x)
    assert torch.allclose(out, out_fft, rtol=1e-4, atol=1e-4)
