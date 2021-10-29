import pytest

import torch
import torch.nn as nn

from einops import rearrange, reduce

from fast_transformers.masking import FullMask, LengthMask

from src.models.modules.multihead_attention import MultiheadAttention
from src.models.modules.attention.full_attention import FullAttention


def seed_cpu_cuda(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class TestMultiheadAttention:

    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    @pytest.mark.parametrize('batch_first', [True, False])
    @pytest.mark.parametrize('add_zero_attn', [True, False])
    @pytest.mark.parametrize('add_bias_kv', [True, False])
    @pytest.mark.parametrize('bias', [True, False])
    @pytest.mark.parametrize('dropout', [0.0, 0.3])
    @pytest.mark.parametrize('same_kv', [True, False])
    @pytest.mark.parametrize('same_qk', [True, False])
    def test_output(self, same_qk, same_kv, dropout, bias, add_bias_kv, add_zero_attn, batch_first,
                    device):
        seed = 2357
        embed_dim = 21
        num_heads = 3
        batch_size = 5
        q_seqlen = 47
        k_seqlen = 37 if not same_qk else q_seqlen
        seed_cpu_cuda(seed)
        attn_mask = FullMask(torch.randint(low=0, high=2, size=(q_seqlen, k_seqlen),
                                           dtype=torch.bool, device=device))
        key_padding_mask = LengthMask(torch.randint(low=0, high=k_seqlen, size=(batch_size,),
                                                    device=device), max_len=k_seqlen)
        seed_cpu_cuda(seed)
        mha_pt = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=bias,
                                       add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn,
                                       batch_first=batch_first).to(device)
        seed_cpu_cuda(seed)
        mha = MultiheadAttention(embed_dim, num_heads, bias=bias,
                                 add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn,
                                 batch_first=batch_first).to(device)
        full_attn = FullAttention(attention_dropout=dropout)
        q = torch.randn(batch_size, q_seqlen, embed_dim, device=device)
        k = q if same_qk else torch.randn(batch_size, k_seqlen, embed_dim, device=device)
        v = k if same_kv else torch.randn(batch_size, k_seqlen, embed_dim, device=device)
        if not batch_first:
             q, k, v = [rearrange(x, 'b s d -> s b d').contiguous() for x in [q, k, v]]
        for training in [True, False]:
            seed_cpu_cuda(seed + 1)
            mha_pt.train(training)
            # Pytorch uses a different convention for the mask: 1 means ignore and 0 means keep
            output_pt, attn_pt = mha_pt(q, k, v,
                                        attn_mask=~attn_mask.bool_matrix,
                                        key_padding_mask=~key_padding_mask.bool_matrix
                                        )
            seed_cpu_cuda(seed + 1)
            mha.train(training)
            full_attn.train(training)
            output, attn = mha(full_attn, q, k, v,
                               attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask,
                               need_weights=(dropout == 0.0))
            assert output.shape == ((batch_size, q_seqlen, embed_dim) if batch_first
                                    else (q_seqlen, batch_size, embed_dim))
            if attn is not None:
                assert attn.shape == (batch_size, num_heads, q_seqlen,
                                      k_seqlen + int(add_bias_kv) + int(add_zero_attn))
            # We put the bias_k/v and the zero_k/v in a different position compared to Pytorch's
            # implementation, so the dropout mask would be different.
            if not (dropout != 0.0 and (add_bias_kv or add_zero_attn)):
                assert torch.allclose(output, output_pt, rtol=1e-5, atol=1e-6, equal_nan=True)
            # Our FullAttention returns the attention weight *before* dropout, while PyTorch
            # returns the attention weight *after* dropout, so we don't expect them to be equal.
            # Also we return the attention weights for all heads, while Pytorch takes the mean.
            if dropout == 0.0 and not add_bias_kv and not add_zero_attn:
                attn_mean = reduce(attn, 'b h t s -> b t s', 'mean')
                assert torch.allclose(attn_mean, attn_pt, rtol=1e-5, atol=1e-6, equal_nan=True)

    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    @pytest.mark.parametrize('batch_first', [True, False])
    @pytest.mark.parametrize('add_zero_attn', [True, False])
    @pytest.mark.parametrize('add_bias_kv', [True, False])
    @pytest.mark.parametrize('bias', [True, False])
    @pytest.mark.parametrize('dropout', [0.0, 0.3])
    @pytest.mark.parametrize('same_kv', [True, False])
    @pytest.mark.parametrize('same_qk', [True, False])
    def test_kdim_vdim(self, same_qk, same_kv, dropout, bias, add_bias_kv, add_zero_attn,
                       batch_first, device):
        """ Because of our different interpretation of kdim and vdim compared to Pytorch,
        we only test the output shape and don't compare to Pytorch's output.
        """
        seed = 2357
        embed_dim = 21
        kdim = 33
        vdim = 18
        num_heads = 3
        batch_size = 5
        q_seqlen = 47
        k_seqlen = 37 if not same_qk else q_seqlen
        seed_cpu_cuda(seed)
        attn_mask = FullMask(torch.randint(low=0, high=2, size=(q_seqlen, k_seqlen),
                                           dtype=torch.bool, device=device))
        key_padding_mask = LengthMask(torch.randint(low=0, high=k_seqlen, size=(batch_size,),
                                                    device=device), max_len=k_seqlen)
        seed_cpu_cuda(seed)
        mha = MultiheadAttention(embed_dim, num_heads, bias=bias,
                                 add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn,
                                 kdim=kdim, vdim=vdim, batch_first=batch_first).to(device)
        full_attn = FullAttention(attention_dropout=dropout)
        q = torch.randn(batch_size, q_seqlen, embed_dim, device=device)
        k = q if same_qk else torch.randn(batch_size, k_seqlen, embed_dim, device=device)
        v = k if same_kv else torch.randn(batch_size, k_seqlen, embed_dim, device=device)
        if not batch_first:
             q, k, v = [rearrange(x, 'b s d -> s b d').contiguous() for x in [q, k, v]]
        for training in [True, False]:
            mha.train(training)
            full_attn.train(training)
            output, attn = mha(full_attn, q, k, v,
                               attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask,
                               need_weights=(dropout == 0.0))
            assert output.shape == ((batch_size, q_seqlen, embed_dim) if batch_first
                                    else (q_seqlen, batch_size, embed_dim))
            if attn is not None:
                assert attn.shape == (batch_size, num_heads, q_seqlen,
                                      k_seqlen + int(add_bias_kv) + int(add_zero_attn))
