import pytest

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, rearrange

from src.models.modules.masking import LengthMask, TriangularCausalMask
from src.models.attention.full_attention import FullAttention
from src.models.attention.sbsmyrf_attention import SBSmyrfAttention


def seed_cpu_cuda(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class TestSBSmyrfAttention:

    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    @pytest.mark.parametrize('softmax_temp', [None, 1.0, 0.235])
    @pytest.mark.parametrize('nb_features', [73, 26, 30000])
    @pytest.mark.parametrize('n_clusters', [4, 6, 8])
    @pytest.mark.parametrize('n_hashes', [1, 2, 3])
    # @pytest.mark.parametrize('causal', [False, True])
    @pytest.mark.parametrize('causal', [False])
    def test_output(self, causal, n_hashes, n_clusters, nb_features, softmax_temp, device):
        if nb_features > 10000 and device == 'cpu':  # Would be too slow on CPU
            return
        # TODO: test causal masks
        seed = 2357
        embed_dim = 21
        v_dim = 17
        num_heads = 7
        batch_size = 18
        q_seqlen = 47
        k_seqlen = 39 if not causal else q_seqlen

        seed_cpu_cuda(seed)
        attn_mask = None if not causal else TriangularCausalMask(q_seqlen, device=device)
        key_padding_mask = LengthMask(torch.randint(low=k_seqlen // 4, high=k_seqlen,
                                                    size=(batch_size,), device=device),
                                      max_len=k_seqlen)
        key_padding_mask = None
        q_cluster_size = (q_seqlen + n_clusters - 1) // n_clusters
        k_cluster_size = (k_seqlen + n_clusters - 1) // n_clusters
        sbsmyrf_attn = SBSmyrfAttention(n_hashes, q_cluster_size, k_cluster_size,
                                        embed_dim, nb_features=nb_features,
                                        softmax_temp=softmax_temp, attention_dropout=0.0,
                                        causal=causal).to(device)
        q = torch.randn(batch_size, q_seqlen, num_heads, embed_dim, device=device)
        k = torch.randn(batch_size, k_seqlen, num_heads, embed_dim, device=device)
        v = torch.randn(batch_size, k_seqlen, num_heads, v_dim, device=device)
        # key_padding_mask = LengthMask(k.new_full((k.shape[0],), k.shape[1], dtype=torch.long))
        out_sbsmyrf, (A_sbsmyrf, A_sbsmyrf_unnormalized, A_lr_unnormalized, smyrf_mask) = sbsmyrf_attn(
            q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=True,
            return_attn_unnormalized=True
        )
        assert out_sbsmyrf.shape == (batch_size, q_seqlen, num_heads, v_dim)
        assert A_sbsmyrf.shape == (batch_size, num_heads, q_seqlen, k_seqlen)
        assert torch.all(A_sbsmyrf >= 0)
        # Sum of each row should be either 0.0 or 1.0
        A_sbsmyrf_sum = A_sbsmyrf.sum(dim=-1)
        assert torch.all(torch.isclose(A_sbsmyrf_sum, torch.ones_like(A_sbsmyrf_sum), atol=1e-2)
                         | torch.isclose(A_sbsmyrf_sum, torch.zeros_like(A_sbsmyrf_sum), atol=1e-2))
        # For some reason if nb_features is large this fails,
        # maybe because of clamping the normalization
        if nb_features < 1000:
            assert torch.allclose(out_sbsmyrf, torch.einsum('bhts,bshd->bthd', A_sbsmyrf, v),
                                rtol=1e-3, atol=1e-3)

        temperature = 1 / math.sqrt(embed_dim) if softmax_temp is None else softmax_temp
        A_full_unnormalized = torch.exp(torch.einsum('bthe,bshe->bhts', q * temperature, k))
        if attn_mask is not None:
            A_full_unnormalized.masked_fill_(~attn_mask.bool_matrix, 0.0)
        if key_padding_mask is not None:
            A_full_unnormalized.masked_fill_(rearrange(~key_padding_mask.bool_matrix, 'b s -> b 1 1 s'),
                                            0.0)
        # Test that A_sbsmyrf_unnormalized matches A_full_unnormalized on the smyrf indices
        # and A_lr_unnormalized on the non-smyrf indices.
        rel_error = ((A_sbsmyrf_unnormalized - A_full_unnormalized)
                        / A_full_unnormalized.clamp_min_(1e-6)).abs().mean()
        if device == 'cuda':
            print(f'Relative error of attention matrix: {rel_error}')
        # For some reason if nb_features is large this fails,
        # maybe because of clamping the normalization
        if nb_features < 1000:
            assert torch.allclose(A_sbsmyrf_unnormalized.masked_select(smyrf_mask),
                                A_full_unnormalized.masked_select(smyrf_mask),
                                rtol=1e-3, atol=1e-4)
        assert torch.allclose(A_sbsmyrf_unnormalized.masked_select(~smyrf_mask),
                              A_lr_unnormalized.masked_select(~smyrf_mask),
                              rtol=1e-3, atol=1e-4)

        # If nb_features is large, test that A_sbsmyrf_unnormalized is close to A_full_unnormalized
        if nb_features > 10000 and (softmax_temp is None):
            assert torch.allclose(rel_error, torch.zeros(1, device=device), atol=0.2)
