import math

import torch
import torch.nn.functional as F

import pytest

from einops import rearrange, repeat

from src.ops.fused_softmax_dropout import _fused_softmax_dropout


@pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.parametrize('seqlen', [128, 512, 1024])
def test_softmax_dropout(seqlen, dtype):
    device = 'cuda'
    dropout_prob = 0.37
    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-3, 3e-4)
    # set seed
    torch.random.manual_seed(0)
    batch_size = 16
    nheads = 2
    x = torch.randn(batch_size, nheads, seqlen, seqlen, device=device, dtype=dtype,
                    requires_grad=True)

    lengths = torch.randint(seqlen - 20, seqlen, (batch_size, 1), device=device)
    attention_mask_bool = repeat(torch.arange(seqlen, device=device),
                                 's -> b s', b=batch_size) < lengths
    attention_mask = torch.zeros(batch_size, seqlen, device=device, dtype=dtype)
    attention_mask[~attention_mask_bool] = -10000.0
    attention_mask = rearrange(attention_mask, 'b s -> b 1 1 s')

    torch.random.manual_seed(0)
    out, dropout_mask = _fused_softmax_dropout.apply(x, dropout_prob, attention_mask,
                                                     True)
    out_pt = F.softmax(x + attention_mask, dim=-1, dtype=x.dtype) * dropout_mask / (1 - dropout_prob)
    assert torch.allclose(out, out_pt, rtol=rtol, atol=atol)

    g = torch.randn_like(out)
    dx, = torch.autograd.grad(out, x, g)
    dx_pt, = torch.autograd.grad(out_pt, x, g)
    assert torch.allclose(dx, dx_pt, rtol=rtol, atol=atol)
