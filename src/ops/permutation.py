import math
import torch


def bitreversal_permutation(n, device=None, dtype=None):
    """Return the bit reversal permutation used in FFT.
    By default, the permutation is stored in numpy array.
    Parameter:
        n: integer, must be a power of 2.
    Return:
        perm: bit reversal permutation, pytorch tensor of size n
    """
    log_n = int(math.log2(n))
    assert n == 1 << log_n, 'n must be a power of 2'
    perm = torch.arange(n, device=device, dtype=dtype).reshape(1, n)
    for i in range(log_n):
        perm = torch.vstack(perm.chunk(2, dim=-1))
    perm = perm.squeeze(-1)
    return perm


def invert_permutation(perm: torch.Tensor) -> torch.Tensor:
    """
    Params:
        perm: (..., n)
    Return:
        inverse_perm: (..., n)
    """
    # This is simpler but has complexity O(n log n)
    # return torch.argsort(perm, dim=-1)
    # This is more complicated but has complexity O(n)
    arange = torch.arange(perm.shape[-1], device=perm.device).expand_as(perm)
    return torch.empty_like(perm).scatter_(-1, perm, arange)
