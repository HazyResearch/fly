import pytest

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, rearrange

from fast_transformers.masking import LengthMask, TriangularCausalMask

from src.models.modules.attention.full_attention import FullAttention
from src.models.modules.attention.sblocal_attention import SBLocalAttention


def seed_cpu_cuda(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class TestSBLocalAttention:

    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    @pytest.mark.parametrize('softmax_temp', [None, 1.0, 0.235])
    @pytest.mark.parametrize('nb_features', [73, 26, 30000])
    @pytest.mark.parametrize('local_context', [3, 4, 28, 33])
    # @pytest.mark.parametrize('causal', [False, True])
    @pytest.mark.parametrize('causal', [False])
    def test_output(self, causal, local_context, nb_features, softmax_temp, device):
        if nb_features > 10000 and device == 'cpu':  # Would be too slow on CPU
            return
        # TODO: test causal masks
        seed = 2357
        embed_dim = 21
        v_dim = 17
        num_heads = 7
        batch_size = 18
        q_seqlen = 47
        # [2021-08-08] local_dot_product_cuda has a bug when q_seqlen != k_seqlen
        # https://github.com/idiap/fast-transformers/issues/98
        k_seqlen = 39 if device == 'cpu' or not causal else q_seqlen

        seed_cpu_cuda(seed)
        attn_mask = None if not causal else TriangularCausalMask(q_seqlen, device=device)
        key_padding_mask = LengthMask(torch.randint(low=k_seqlen // 4, high=k_seqlen,
                                                    size=(batch_size,), device=device),
                                      max_len=k_seqlen)
        sblocal_attn = SBLocalAttention(local_context, embed_dim, nb_features,
                                        softmax_temp=softmax_temp, attention_dropout=0.0,
                                        causal=causal).to(device)
        q = torch.randn(batch_size, q_seqlen, num_heads, embed_dim, device=device)
        k = torch.randn(batch_size, k_seqlen, num_heads, embed_dim, device=device)
        v = torch.randn(batch_size, k_seqlen, num_heads, v_dim, device=device)
        # key_padding_mask = LengthMask(k.new_full((k.shape[0],), k.shape[1], dtype=torch.long))
        out_sblocal, (A_sblocal, A_sblocal_unnormalized, A_lr_unnormalized) = sblocal_attn(
            q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=True,
            return_attn_unnormalized=True
        )
        assert out_sblocal.shape == (batch_size, q_seqlen, num_heads, v_dim)
        assert A_sblocal.shape == (batch_size, num_heads, q_seqlen, k_seqlen)
        assert torch.all(A_sblocal >= 0)
        # Sum of each row should be either 0.0 or 1.0
        A_sblocal_sum = A_sblocal.sum(dim=-1)
        assert torch.all(torch.isclose(A_sblocal_sum, torch.ones_like(A_sblocal_sum), atol=1e-2)
                         | torch.isclose(A_sblocal_sum, torch.zeros_like(A_sblocal_sum), atol=1e-2))
        assert torch.allclose(out_sblocal, torch.einsum('bhts,bshd->bthd', A_sblocal, v),
                              rtol=1e-3, atol=1e-3)

        temperature = 1 / math.sqrt(embed_dim) if softmax_temp is None else softmax_temp
        A_full_unnormalized = torch.exp(torch.einsum('bthe,bshe->bhts', q * temperature, k))
        if attn_mask is not None:
            A_full_unnormalized.masked_fill_(~attn_mask.bool_matrix, 0.0)
        A_full_unnormalized.masked_fill_(rearrange(~key_padding_mask.bool_matrix, 'b s -> b 1 1 s'),
                                         0.0)
        # Test that A_sblocal_unnormalized matches A_full_unnormalized on the local indices
        # and A_lr_unnormalized on the non-local indices.
        i = rearrange(torch.arange(q_seqlen, device=q.device), 't -> 1 1 t 1')
        j = torch.arange(k_seqlen, device=k.device)
        idx = j - i
        local_mask = ((idx >= -(local_context // 2))
                      & (idx < (local_context + 1) // 2)
                      & (j < rearrange(key_padding_mask.lengths, 'b -> b 1 1 1')))
        assert torch.allclose(A_sblocal_unnormalized.masked_select(local_mask),
                              A_full_unnormalized.masked_select(local_mask),
                              rtol=1e-3, atol=1e-4)
        assert torch.allclose(A_sblocal_unnormalized.masked_select(~local_mask),
                              A_lr_unnormalized.masked_select(~local_mask),
                              rtol=1e-3, atol=1e-4)

        rel_error = ((A_sblocal_unnormalized - A_full_unnormalized)
                        / A_full_unnormalized.clamp_min_(1e-6)).abs().mean()
        if device == 'cuda':
            print(f'Relative error of attention matrix: {rel_error}')
        # If nb_features is large, test that A_sblocal_unnormalized is close to A_full_unnormalized
        if nb_features > 10000 and local_context > 10 and (softmax_temp is None
                                                       or softmax_temp < 1.0):
            assert torch.allclose(rel_error, torch.zeros(1, device=device), atol=0.2)
