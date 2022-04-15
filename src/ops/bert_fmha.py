# Adapt from https://github.com/mlcommons/training_results_v1.1/blob/main/NVIDIA/benchmarks/bert/implementations/pytorch/fmha.py
import torch
import torch.nn as nn

# import fmhalib
# import fmhalibmine as fmhalib
import fmhalibtd as fmhalib

from einops import rearrange


def _fmha_forward(qkv, cu_seqlens, p_dropout, max_s, is_training, return_softmax):
    context, softmax_lse, *rest = fmhalib.fwd(qkv, cu_seqlens, p_dropout, max_s, is_training,
                                              False, return_softmax, None)
    S_dmask = rest[0] if return_softmax else None
    return context, softmax_lse, S_dmask


def _fmha_backward(dout, qkv, out, S_dmask, softmax_lse, cu_seqlens, p_dropout, max_s):
    dqkv, dp, softmax_d = fmhalib.bwd(dout, qkv, out, S_dmask, softmax_lse, cu_seqlens, p_dropout, max_s, False)
    return dqkv


class FMHAFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, qkv, cu_seqlens, p_dropout, max_s, is_training):
        context, softmax_lse, S_dmask = _fmha_forward(qkv, cu_seqlens, p_dropout, max_s, is_training,
                                                      return_softmax=False)
        ctx.save_for_backward(qkv, context, S_dmask, softmax_lse, cu_seqlens)
        ctx.p_dropout = p_dropout
        ctx.max_s = max_s
        return context

    @staticmethod
    def backward(ctx, dout):
        qkv, context, S_dmask, softmax_lse, cu_seqlens = ctx.saved_tensors
        # S_dmask is None, temporarily use another tensor just to get it running
        dqkv = _fmha_backward(dout, qkv, context, context, softmax_lse, cu_seqlens, ctx.p_dropout, ctx.max_s)
        return dqkv, None, None, None, None, None


# We duplicate code to return both the output and the softmax for testing
# Returning both makes backward a bit slower, so we want to keep using the other version for speed.
class FMHAFunWithS(torch.autograd.Function):

    @staticmethod
    def forward(ctx, qkv, cu_seqlens, p_dropout, max_s, is_training):
        context, softmax_lse, S_dmask = _fmha_forward(qkv, cu_seqlens, p_dropout, max_s, is_training,
                                                      return_softmax=True)
        ctx.save_for_backward(qkv, context, S_dmask, softmax_lse, cu_seqlens)
        ctx.p_dropout = p_dropout
        ctx.max_s = max_s
        return context, S_dmask, softmax_lse

    @staticmethod
    def backward(ctx, dout, _dS_dmask_ignored, _dsoftmax_sum_ignored):
        qkv, context, S_dmask, softmax_lse, cu_seqlens = ctx.saved_tensors
        dqkv = _fmha_backward(dout, qkv, context, S_dmask, softmax_lse, cu_seqlens, ctx.p_dropout, ctx.max_s)
        return dqkv, None, None, None, None, None


def fmha_func(qkv, cu_seqlens, p_dropout, max_s, is_training, return_attn_probs=False):
    func = FMHAFun if not return_attn_probs else FMHAFunWithS
    return func.apply(qkv, cu_seqlens, p_dropout, max_s, is_training)
