import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, rearrange

from src.models.modules.masking import FullMask, LengthMask

from src.models.attention.linformer_attention import LinformerAttention


def seed_cpu_cuda(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class TestLinformerAttention:

    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    @pytest.mark.parametrize('softmax_temp', [None, 1.0, 0.235])
    @pytest.mark.parametrize('share_kv', [False, True])
    @pytest.mark.parametrize('proj_dim_k', [13, 47, 88])
    @pytest.mark.parametrize('seq_len', [127, 28, 468])
    def test_output(self, seq_len, proj_dim_k, share_kv, softmax_temp, device):
        seed = 2357
        embed_dim = 21
        v_dim = 17
        num_heads = 7
        batch_size = 18
        q_seqlen = 47
        # [2021-08-08] local_dot_product_cuda has a bug when q_seqlen != k_seqlen
        # https://github.com/idiap/fast-transformers/issues/98
        k_seqlen = seq_len

        seed_cpu_cuda(seed)
        key_padding_mask = LengthMask(torch.randint(low=0, high=k_seqlen, size=(batch_size,),
                                                    device=device), max_len=k_seqlen)
        lin_attn = LinformerAttention(seq_len, k=proj_dim_k, share_kv=share_kv,
                                      softmax_temp=softmax_temp, attention_dropout=0.0).to(device)
        q = torch.randn(batch_size, q_seqlen, num_heads, embed_dim, device=device)
        k = torch.randn(batch_size, k_seqlen, num_heads, embed_dim, device=device)
        v = torch.randn(batch_size, k_seqlen, num_heads, v_dim, device=device)

        out_lin, A_lin = lin_attn(q, k, v, key_padding_mask=key_padding_mask, need_weights=True)
        assert out_lin.shape == (batch_size, q_seqlen, num_heads, v_dim)
        assert A_lin.shape == (batch_size, num_heads, q_seqlen, proj_dim_k)
        assert torch.all(A_lin >= 0)
        # Sum of each row should be either 0.0 or 1.0
        A_local_sum = A_lin.sum(dim=-1)
        assert torch.all(torch.isclose(A_local_sum, torch.ones_like(A_local_sum))
                         | torch.isclose(A_local_sum, torch.zeros_like(A_local_sum)))
