import math

import torch
from torch.nn import functional as F

from einops import rearrange


def block_butterfly_multiply(twiddle, input, increasing_stride=True,
                             output_size=None):
    """
    twiddle: (nstacks, nblocks, log_n, n // 2, 2, 2, block_size, block_size)
    input: (batch_size, nstacks, block_size * n)
    """
    batch_size, nstacks, input_size = input.shape
    nblocks = twiddle.shape[1]
    log_n = twiddle.shape[2]
    block_size = twiddle.shape[-1]
    n = 1 << log_n
    assert twiddle.shape == (nstacks, nblocks, log_n, n // 2, 2, 2, block_size, block_size)
    # Pad or trim input to size block_size * n
    input = (F.pad(input, (0, block_size * n - input_size)) if input_size < block_size * n
             else input[:, :, :block_size * n])
    output_size = block_size * n if output_size is None else output_size
    assert output_size <= block_size * n
    output = input.contiguous()
    cur_increasing_stride = increasing_stride
    for block in range(nblocks):
        for idx in range(log_n):
            log_stride = idx if cur_increasing_stride else log_n - 1 - idx
            stride = 1 << log_stride
            # shape (nstacks, n // (2 * stride), 2, 2, stride, block_size, block_size)
            t = rearrange(twiddle[:, block, idx],
                          's (diagblk stride) i j k l -> s diagblk i j stride k l', stride=stride)
            output_reshape = rearrange(output,
                                       'b s (diagblk j stride l) -> b s diagblk j stride l',
                                       stride=stride, j=2, l=block_size)
            output = torch.einsum('s d i j t k l, b s d j t l -> b s d i t k',
                                  t, output_reshape)
            output = rearrange(output, 'b s diagblk i stride k -> b s (diagblk i stride k)')
            # output_reshape = output.view(
                # batch_size, nstacks, n // (2 * stride), 1, 2, stride, block_size, 1)
            # output = (t @ output_reshape).sum(dim=4).reshape(batch_size, nstacks, block_size * n)
        cur_increasing_stride = not cur_increasing_stride
    return output.view(batch_size, nstacks, block_size * n)[:, :, :output_size]


def block_butterfly_factor_multiply(twiddle, input, factor_idx, increasing_stride=True, output_size=None):
    """
    twiddle: (nstacks, log_n, n // 2, 2, 2, block_size, block_size)
    input: (batch_size, nstacks, block_size * n)
    """
    batch_size, nstacks, input_size = input.shape
    block_size = twiddle.shape[-1]
    log_n = twiddle.shape[1]
    n = 1 << log_n
    assert twiddle.shape == (nstacks, log_n, n // 2, 2, 2, block_size, block_size)
    # Pad or trim input to size block_size * n
    input = (F.pad(input, (0, block_size * n - input_size)) if input_size < block_size * n
             else input[:, :, :block_size * n])
    output_size = block_size * n if output_size is None else output_size
    assert output_size <= block_size * n
    output = input.contiguous()
    cur_increasing_stride = increasing_stride
    idx = factor_idx
    log_stride = idx if cur_increasing_stride else log_n - 1 - idx
    stride = 1 << log_stride
    # shape (nstacks, n // (2 * stride), 2, 2, stride, block_size, block_size)
    t = rearrange(twiddle[:, idx],
                  's (diagblk stride) i j k l -> s diagblk i j stride k l', stride=stride)
    output_reshape = rearrange(output,
                               'b s (diagblk j stride l) -> b s diagblk j stride l',
                               stride=stride, j=2, l=block_size)
    output = torch.einsum('s d i j t k l, b s d j t l -> b s d i t k',
                          t, output_reshape)
    output = rearrange(output, 'b s diagblk i stride k -> b s (diagblk i stride k)')
    return output.view(batch_size, nstacks, block_size * n)[:, :, :output_size]
