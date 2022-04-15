import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, rearrange

from src.models.modules.masking import FullMask, LengthMask

from src.models.attention.full_attention import FullAttention
from src.models.attention.smyrf_attention import SmyrfAttention


def seed_cpu_cuda(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class TestSmyrfAttention:

    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    @pytest.mark.parametrize('softmax_temp', [None, 1.0, 0.235])
    @pytest.mark.parametrize('n_clusters', [4, 6, 8])
    @pytest.mark.parametrize('n_hashes', [1, 2, 3])
    def test_output(self, n_hashes, n_clusters, softmax_temp, device):
        seed = 2357
        embed_dim = 21
        v_dim = 17
        num_heads = 7
        batch_size = 18
        q_seqlen = 257
        k_seqlen = 173

        seed_cpu_cuda(seed)
        attn_mask = None
        key_padding_mask = LengthMask(torch.randint(low=0, high=k_seqlen, size=(batch_size,),
                                                    device=device), max_len=k_seqlen)
        q_cluster_size = (q_seqlen + n_clusters - 1) // n_clusters
        k_cluster_size = (k_seqlen + n_clusters - 1) // n_clusters
        smyrf_attn = SmyrfAttention(n_hashes, q_cluster_size, k_cluster_size,
                                    softmax_temp=softmax_temp, attention_dropout=0.0).to(device)
        full_attn = FullAttention(softmax_temp=softmax_temp, attention_dropout=0.0).to(device)
        q = torch.randn(batch_size, q_seqlen, num_heads, embed_dim, device=device)
        k = torch.randn(batch_size, k_seqlen, num_heads, embed_dim, device=device)
        v = torch.randn(batch_size, k_seqlen, num_heads, v_dim, device=device)

        out_smyrf, A_smyrf = smyrf_attn(q, k, v, attn_mask, key_padding_mask, need_weights=True)
        assert out_smyrf.shape == (batch_size, q_seqlen, num_heads, v_dim)
        assert A_smyrf.shape == (batch_size, num_heads, q_seqlen, k_seqlen)
        assert torch.all(A_smyrf >= 0)
        # Sum of each row should be either 0.0 or 1.0
        A_smyrf_sum = A_smyrf.sum(dim=-1)
        assert torch.all(torch.isclose(A_smyrf_sum, torch.ones_like(A_smyrf_sum))
                         | torch.isclose(A_smyrf_sum, torch.zeros_like(A_smyrf_sum)))
        assert torch.allclose(out_smyrf, torch.einsum('bhts,bshd->bthd', A_smyrf, v),
                              rtol=1e-5, atol=1e-6)

        # Test that A_smyrf is equivalent to zero-ing out some elements A_full and then
        # re-normalize so each row sums to 1
        if n_hashes == 1:
            _, A_full = full_attn(q, k, v, attn_mask, key_padding_mask, need_weights=True)
            smyrf_mask = A_smyrf > 0.0
            A_full_smyrf = F.normalize(A_full.masked_fill(~smyrf_mask, 0.0), p=1, dim=-1)
            assert torch.allclose(A_smyrf, A_full_smyrf, rtol=1e-5, atol=1e-6)
