# Copied from https://github.com/facebookresearch/xformers/blob/main/xformers/triton/k_softmax.py
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import torch
import triton
import triton.language as tl

# CREDITS: This is adapted from the vanilla Triton example. See https://openai.com/blog/triton/
# and https://triton-lang.org/getting-started/tutorials/02-fused-softmax.html


def get_depth(K):
    return triton.next_power_of_2(K)


# fmt: off
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["K"],
)
@triton.heuristics({'DEPTH': lambda nargs: get_depth(nargs['K'])})
@triton.heuristics({'IS_FP16': lambda nargs: nargs['GradIn'].dtype == torch.float16})
@triton.jit
def _softmax_dropout_backward(
    GradIn, GradOut, Out, DropoutMask, dropout_prob,
    stride_bm, stride_bn,
    stride_gm, stride_gn,
    stride_om, stride_on,
    stride_mm, stride_mn,
    K,
    CAUSAL: tl.constexpr,
    DEPTH: tl.constexpr,
    IS_FP16: tl.constexpr,
):
    # fmt: on

    """
    Compute the softmax gradients.
    ..Note: Not autotuning for now because this would lead to broken accumulated gradients
    """

    m = tl.program_id(0)
    n = tl.program_id(1)

    # col indices
    k = tl.arange(0, DEPTH)

    # the memory address of all the elements that we want to load can be computed as follows
    grad_out_ptrs = GradOut + m * stride_gm + n * stride_gn + k
    out_ptrs = Out + m * stride_om + n * stride_on + k
    dropout_mask_ptrs = DropoutMask + m * stride_mm + n * stride_mn + k

    # load input data; pad out-of-bounds elements with 0
    io_mask = k < K

    # Causal - 1: skip on the loads directly
    if CAUSAL:
        io_mask = io_mask & (k <= n)

    g = tl.load(grad_out_ptrs, mask=io_mask, other=float(0))
    o = tl.load(out_ptrs, mask=io_mask, other=float(0))

    zero = float(0)
    zero = zero.to(g.dtype)
    # Causal - 2: enforce correctness over a couple of misloaded values
    if CAUSAL:
        g = tl.where(k > n, zero, g)
        o = tl.where(k > n, zero, o)

    dropout_mask = tl.load(dropout_mask_ptrs, mask=io_mask, other=float(0))
    g = tl.where(dropout_mask != 0, g / (1 - dropout_prob), zero)

    # Step 1: Compute the intermediate sum used for the gradient
    s = tl.sum(g * o, 0)

    # Step 2: Compute the gradients
    grad_in = o * (g - s)

    # write back to the input gradients
    # technically we could write only the lower triangular matrix in the causal case
    # but this is deemed to error prone
    grad_in_ptrs = GradIn + m * stride_bm + n * stride_bn + k
    tl.store(grad_in_ptrs, grad_in, mask=k < K)
