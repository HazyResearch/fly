import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, rearrange

from src.models.modules.masking import FullMask, LengthMask

from src.models.attention.reformer_attention import ReformerAttention


def seed_cpu_cuda(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class TestReformerAttention:

    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    @pytest.mark.parametrize('softmax_temp', [None, 1.0, 0.235])
    @pytest.mark.parametrize('bucket_size', [16, 32, 64])
    @pytest.mark.parametrize('n_hashes', [1, 2, 3])
    def test_output(self, n_hashes, bucket_size, softmax_temp, device):
        seed = 2357
        embed_dim = 21
        v_dim = 17
        # num_heads = 7
        # batch_size = 18
        num_heads = 1
        # batch_size = 18
        batch_size = 1
        # seqlen = 213
        seqlen = 64

        seed_cpu_cuda(seed)
        # attn_mask = None
        attn_mask = FullMask(torch.randint(low=0, high=2, size=(seqlen, seqlen),
                                           dtype=torch.bool, device=device))
        # key_padding_mask = None
        key_padding_mask = LengthMask(torch.randint(low=0, high=seqlen, size=(batch_size,),
                                                    device=device), max_len=seqlen)
        reformer_attn = ReformerAttention(n_hashes=n_hashes, bucket_size=bucket_size,
                                          allow_duplicate_attention=False,
                                          softmax_temp=softmax_temp, attention_dropout=0.0).to(device)
        q = torch.randn(batch_size, seqlen, num_heads, embed_dim, device=device)
        k = q
        v = torch.randn(batch_size, seqlen, num_heads, v_dim, device=device)

        out_reformer, A_reformer = reformer_attn(q, k, v, attn_mask, key_padding_mask, need_weights=True)
        assert out_reformer.shape == (batch_size, seqlen, num_heads, v_dim)
        assert A_reformer.shape == (batch_size, num_heads, seqlen, seqlen)
        assert torch.all(A_reformer >= 0)
        # Sum of each row should be either 0.0 or 1.0
        # A_smyrf_sum = A_reformer.sum(dim=-1)
        # assert torch.all(torch.isclose(A_smyrf_sum, torch.ones_like(A_smyrf_sum))
        #                  | torch.isclose(A_smyrf_sum, torch.zeros_like(A_smyrf_sum)))
        assert torch.allclose(out_reformer, torch.einsum('bhts,bshd->bthd', A_reformer, v),
                              rtol=1e-5, atol=1e-6)
