from typing import Optional
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd

from einops import rearrange, repeat

import triton
import triton.language as tl

from src.ops.triton.k_softmax import _softmax, _softmax_backward
from src.ops.triton.k_softmax_dropout import _softmax_dropout_backward
from src.ops.triton.softmax import softmax

FAST_MHA_AVAILABLE = True
try:
    from fast_multihead_attn import additive_mask_softmax_dropout_backward
except ImportError:
    from src.utils.utils import get_logger
    logger = get_logger()
    logger.info('fast_multihead_attn from apex is not installed.')
    FAST_MHA_AVAILABLE = False


_triton_registered_overflow = False
_triton_softmax_fp16_enabled = False  # NOTE: PyTorch keeps softmax as fp32


# Helper to handle the SPMD launch grid and error cases
class _softmax_dropout_triton(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16 if _triton_softmax_fp16_enabled else None)
    def forward(ctx, x, p, mask, causal, mask_type):
        """
        Fused softmax implementation, using the Triton programming model.
        This only supports a reduction over the last dimension for now
        Argument:
            x: (bs, nheads, q_seqlen, k_seqlen)
            mask: (bs, 1, 1, k_seqlen)
        """
        assert x.ndim == 4
        # Handle 2D/3D tensors
        x_ = x.unsqueeze(0) if x.ndim == 2 else x
        x_ = x_.flatten(0, -3)

        if not x_.is_contiguous():
            x_ = x_.contiguous()

        y = torch.empty_like(x_)
        assert (
            y.stride(2) == 1 and x_.stride(2) == 1
        ), f"{x.shape} - {x_.shape} - {x_.stride()}"

        # SPMD launch grid
        grid_2d = (
            x_.shape[0],
            x_.shape[1],
        )

        # enqueue GPU kernel
        if mask is None:
            #  placeholder, will not be used
            mask = x_
            mask_type = None
        else:
            assert mask.dtype == x.dtype, "An additive mask is requested"
            if mask_type == 'bk':
                mask = repeat(mask, 'b 1 1 s -> b h 1 s', h=x_.shape[0] // mask.shape[0])
            mask = mask.flatten(0, -2).contiguous()

        _softmax[grid_2d](
            y,
            x_,
            mask,
            y.stride(0),
            y.stride(1),
            x_.stride(0),
            x_.stride(1),
            mask.stride(0),
            x_.shape[2],
            LOG=False,
            MASK_TYPE=mask_type,
            CAUSAL=causal,
        )

        # torch._fused_dropout takes 1 - p
        dropout_results, dropout_mask = torch._fused_dropout(y, p=1.0 - p)

        ctx.save_for_backward(y, dropout_mask)
        ctx.dropout_prob = p
        ctx.causal = causal
        ctx.mask_type = mask_type
        return dropout_results.reshape_as(x)

    # @staticmethod
    # @custom_bwd
    # def backward(ctx, grad_out):
    #     (y, dropout_mask) = ctx.saved_tensors

    #     # triton can't read from bool, uint8, or int8. Converting to int16 negatives the speed
    #     # benefits.
    #     # dropout_mask_triton = triton.code_gen.reinterpret(dropout_mask, tl.uint8)
    #     dropout_mask_triton = dropout_mask.to(dtype=torch.int16)

    #     # Handle 2D/3D tensors
    #     grad_out_ = grad_out.unsqueeze(0) if grad_out.ndim == 2 else grad_out
    #     grad_out_ = grad_out_.flatten(0, -3)

    #     # SPMD launch grid
    #     grid_2d = (
    #         grad_out_.shape[0],
    #         grad_out_.shape[1],
    #     )

    #     grad_in = torch.empty_like(
    #         y
    #     )  # torch.zeros is measurably slower, we'll zero y in the kernel

    #     # Make sure that the tensor are contiguous
    #     grad_in, grad_out, y = map(lambda x: x.contiguous(), [grad_in, grad_out, y])

    #     # fmt: off
    #     _softmax_dropout_backward[grid_2d](
    #         grad_in, grad_out_, y, dropout_mask_triton, ctx.dropout_prob,
    #         grad_in.stride(0), grad_in.stride(1),
    #         grad_out_.stride(0), grad_out_.stride(1),
    #         y.stride(0), y.stride(1),
    #         dropout_mask.stride(0), dropout_mask.stride(1),
    #         y.shape[2],
    #         CAUSAL=ctx.causal
    #     )
    #     # fmt: on
    #     return grad_in.reshape_as(grad_out), None, None, None, None

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        (y, dropout_mask) = ctx.saved_tensors

        # Handle 2D/3D tensors
        grad_out_ = grad_out.unsqueeze(0) if grad_out.ndim == 2 else grad_out
        grad_out_ = grad_out_.flatten(0, -3)

        # SPMD launch grid
        grid_2d = (
            grad_out_.shape[0],
            grad_out_.shape[1],
        )

        # Make sure that the tensor are contiguous
        # grad_in, grad_out_, y = map(lambda x: x.contiguous(), [grad_in, grad_out_, y])
        grad_out_, y = map(lambda x: x.contiguous(), [grad_out_, y])

        if (FAST_MHA_AVAILABLE and grad_out.dtype == torch.float16 and not ctx.causal
            and ctx.mask_type == 'bk'):
            # fast_multihead_attn from apex only works for fp16 for now.
            # Apex overwrites grad_output, i.e. in-place.
            # The first two flags (use_mask, heads) aren't used at all, can be set to whatever.
            grad_in = additive_mask_softmax_dropout_backward(True, 1, grad_out_, y, dropout_mask,
                                                             ctx.dropout_prob)
        else:
            dropout_grads = torch._masked_scale(grad_out_, dropout_mask,
                                                1.0 / (1.0 - ctx.dropout_prob))
            grad_in = torch.empty_like(
                y
            )  # torch.zeros is measurably slower, we'll zero y in the kernel

            # fmt: off
            _softmax_backward[grid_2d](
                grad_in, dropout_grads, y,
                grad_in.stride(0), grad_in.stride(1),
                grad_out_.stride(0), grad_out_.stride(1),
                y.stride(0), y.stride(1),
                y.shape[2],
                LOG=False,
                CAUSAL=ctx.causal
            )

        # fmt: on
        return grad_in.reshape_as(grad_out), None, None, None, None


def softmax_dropout(
        x: torch.Tensor, p: float, mask: Optional[torch.Tensor] = None, causal: bool = False,
        mask_type: str = 'qk'
) -> torch.Tensor:
    if p == 0.0:
        return softmax(x, mask=mask, mask_type=mask_type)
    else:
        return _softmax_dropout_dispatch(x, p, mask, causal, mask_type=mask_type)


def _softmax_dropout_dispatch(
        x: torch.Tensor, p: float, mask: Optional[torch.Tensor], causal: bool = False,
        mask_type: str = 'qk'
) -> torch.Tensor:
    # Triton is used if
    # - CUDA
    # - there's enough data to make it faster than pytorch. This could change over time, Triton is improving
    # - there was no previous failure

    global _triton_registered_overflow

    try:
        if torch.cuda.is_available() and x.is_cuda and not _triton_registered_overflow:
            return _softmax_dropout_triton.apply(x, p, mask, causal, mask_type)
    except (triton.code_gen.OutOfResources, RuntimeError) as e:
        # Catch cases where the current GPU does not have enough registers to hold a full tensor line
        # fallback to PyTorch's implementation, which streams the tensor in and out
        _triton_registered_overflow = True
        logging.warning(
            "Triton softmax kernel register spillover or invalid image caught."
            "Deactivating this kernel, please file an issue int the xFormers repository"
        )
        logging.warning(e)

    if mask is not None:
        mask = mask.to(dtype=x.dtype)
        if mask_type == 'qk':
            x = x + mask
        elif mask_type == 'bk':
            x = x + rearrange(mask, '... k -> ... 1 k')

    if causal:
        x = x + torch.triu(torch.full_like(x, float("-inf")), diagonal=1)

    return F.dropout(F.softmax(x, dim=-1, dtype=x.dtype), p)


class SoftmaxDropout(nn.Module):
    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, causal: bool = False,
                mask_type: str = 'qk') -> torch.Tensor:
        p = self.p if self.training else 0.0
        if not x.is_cuda:
            if mask is not None:
                mask = mask.to(dtype=x.dtype)
                if mask_type == 'qk':
                    x = x + mask
                elif mask_type == 'bk':
                    x = x + rearrange(mask, '... k -> ... 1 k')
            if causal:
                x = x + torch.triu(torch.full_like(x, float("-inf")), diagonal=1)
            return F.dropout(F.softmax(x, dim=-1, dtype=x.dtype), self.p)
        else:
            return softmax_dropout(x, p, mask=mask, causal=causal, mask_type=mask_type)
