import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, rearrange

from src.models.modules.masking import FullMask, LengthMask

from src.models.modules.attention.full_attention import FullAttention
from src.models.modules.attention.local_attention import LocalAttention


def seed_cpu_cuda(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class TestLocalAttention:

    def test_local_dot_product(self):
        """Test if we can import and run local_dot_product at all
        """
        from fast_transformers.local_product import local_dot_product, local_weighted_average
        batch_size = 1
        num_heads = 3
        seqlen = 10
        embed_dim = 7
        local_context = 5
        q = torch.randn(batch_size, num_heads, seqlen, embed_dim, requires_grad=True)
        k = torch.randn(batch_size, num_heads, seqlen, embed_dim, requires_grad=True)
        v = torch.randn(batch_size, num_heads, seqlen, embed_dim, requires_grad=True)
        attn_additive_mask = torch.zeros(10, 10)
        lengths = torch.full((1,), 10)
        qk = local_dot_product(q, k, attn_additive_mask, lengths, local_context)
        qk.sum().backward()
        assert qk.shape == (batch_size, num_heads, seqlen, local_context)
        qk = local_dot_product(q.cuda(), k.cuda(), attn_additive_mask.cuda(), lengths.cuda(), local_context)
        assert qk.shape == (batch_size, num_heads, seqlen, local_context)
        A = torch.softmax(1.5 * qk, dim=-1)
        out = local_weighted_average(A, v.cuda())
        out.sum().backward()  # Test that backward also runs
        assert out.shape == (batch_size, num_heads, seqlen, embed_dim)

    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    @pytest.mark.parametrize('softmax_temp', [None, 1.0, 0.235])
    @pytest.mark.parametrize('local_context', [3, 4, 28, 33])
    def test_output(self, local_context, softmax_temp, device):
        seed = 2357
        embed_dim = 21
        v_dim = 17
        num_heads = 7
        batch_size = 18
        q_seqlen = 47
        # [2021-08-08] local_dot_product_cuda has a bug when q_seqlen != k_seqlen
        # https://github.com/idiap/fast-transformers/issues/98
        k_seqlen = 39 if device == 'cpu' else q_seqlen

        seed_cpu_cuda(seed)
        # For arbitrary attn_mask, we could have a row with no valid keys. That row should actually
        # be zero but for implementation simplicity we don't correct it.
        # Therefore we're only testing with full attn_mask.
        # attn_mask = FullMask(torch.randint(low=0, high=2, size=(q_seqlen, k_seqlen),
        #                                    dtype=torch.bool, device=device))
        attn_mask = None
        key_padding_mask = LengthMask(torch.randint(low=0, high=k_seqlen, size=(batch_size,),
                                                    device=device), max_len=k_seqlen)
        # key_padding_mask = LengthMask(torch.ones(batch_size, dtype=torch.long) * 3, max_len=k_seqlen)
        local_attn = LocalAttention(local_context, softmax_temp=softmax_temp,
                                    attention_dropout=0.0).to(device)
        full_attn = FullAttention(softmax_temp=softmax_temp, attention_dropout=0.0).to(device)
        q = torch.randn(batch_size, q_seqlen, num_heads, embed_dim, device=device)
        k = torch.randn(batch_size, k_seqlen, num_heads, embed_dim, device=device)
        v = torch.randn(batch_size, k_seqlen, num_heads, v_dim, device=device)

        out_local, A_local = local_attn(q, k, v, attn_mask, key_padding_mask, need_weights=True)
        assert out_local.shape == (batch_size, q_seqlen, num_heads, v_dim)
        assert A_local.shape == (batch_size, num_heads, q_seqlen, k_seqlen)
        assert torch.all(A_local >= 0)
        # Sum of each row should be either 0.0 or 1.0
        A_local_sum = A_local.sum(dim=-1)
        assert torch.all(torch.isclose(A_local_sum, torch.ones_like(A_local_sum))
                         | torch.isclose(A_local_sum, torch.zeros_like(A_local_sum)))
        assert torch.all((A_local >= 1e-6).sum(dim=-1) <= local_context)
        assert torch.allclose(out_local, torch.einsum('bhts,bshd->bthd', A_local, v),
                              rtol=1e-6, atol=1e-7)

        _, A_full = full_attn(q, k, v, attn_mask, key_padding_mask, need_weights=True)
        # Test that A_local is equivalent to zero-ing out non-local elements A_full and then
        # re-normalize so each row sums to 1
        i = rearrange(torch.arange(q_seqlen, device=q.device), 't -> 1 1 t 1')
        j = torch.arange(k_seqlen, device=k.device)
        idx = j - i
        local_mask = ((idx >= -(local_context // 2))
                      & (idx < (local_context + 1) // 2)
                      & (j < rearrange(key_padding_mask.lengths, 'b -> b 1 1 1')))
        A_full_local = F.normalize(A_full.masked_fill(~local_mask, 0.0), p=1, dim=-1)
        assert torch.allclose(A_local, A_full_local, rtol=1e-5, atol=1e-7)
