import pytest

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, rearrange

from fast_transformers.masking import LengthMask, TriangularCausalMask

from src.models.attention.full_attention import FullAttention
from src.models.attention.sbblocksparse_attention import SBBlockSparseAttention
from src.models.attention.blocksparse_utils import mask_tensor


def seed_cpu_cuda(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class TestSBBlockSparseAttention:

    @pytest.mark.parametrize('device', ['cuda'])
    @pytest.mark.parametrize('softmax_temp', [None, 1.0, 0.235])
    @pytest.mark.parametrize('nb_features', [80, 32, 1024])
    @pytest.mark.parametrize('num_local_blocks', [1, 3, 4])
    @pytest.mark.parametrize('block_size', [16])
    @pytest.mark.parametrize('causal', [False])
    def test_output(self, causal, block_size, num_local_blocks, nb_features, softmax_temp, device):
        if nb_features > 10000 and device == 'cpu':  # Would be too slow on CPU
            return
        # TODO: test causal masks
        seed = 2357
        embed_dim = 32
        v_dim = 48
        num_heads = 7
        batch_size = 18
        q_seqlen = 240
        # [2021-08-08] local_dot_product_cuda has a bug when q_seqlen != k_seqlen
        # https://github.com/idiap/fast-transformers/issues/98
        k_seqlen = 64 if device == 'cpu' or not causal else q_seqlen
        q_seqlen_padded = int(math.ceil(q_seqlen / block_size) * block_size)
        k_seqlen_padded = int(math.ceil(k_seqlen / block_size) * block_size)

        seed_cpu_cuda(seed)
        attn_mask = None if not causal else TriangularCausalMask(q_seqlen, device=device)
        # key_padding_mask = LengthMask(torch.randint(low=k_seqlen // 4, high=k_seqlen,
        #                                             size=(batch_size,), device=device),
        #                               max_len=k_seqlen)
        key_padding_mask = None
        sparsity_config = dict(
            _target_='deepspeed.ops.sparse_attention.FixedSparsityConfig',
            num_heads=num_heads,
            block=block_size,
            num_local_blocks=num_local_blocks
        )
        sbblocksparse_attn = SBBlockSparseAttention(sparsity_config, embed_dim, nb_features,
                                                    softmax_temp=softmax_temp, attention_dropout=0.0,
                                                    causal=causal,
                                                    max_seq_length=max(q_seqlen, k_seqlen)).to(device)
        q = torch.randn(batch_size, q_seqlen, num_heads, embed_dim, device=device)
        k = torch.randn(batch_size, k_seqlen, num_heads, embed_dim, device=device)
        v = torch.randn(batch_size, k_seqlen, num_heads, v_dim, device=device)
        layout = sbblocksparse_attn.get_layout(q_seqlen_padded, k_seqlen_padded)
        # key_padding_mask = LengthMask(k.new_full((k.shape[0],), k.shape[1], dtype=torch.long))
        out_sblocal, (A_sbblocksparse, A_sbblocksparse_unnormalized, A_lr_unnormalized) = sbblocksparse_attn(
            q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=True,
            return_attn_unnormalized=True
        )
        assert out_sblocal.shape == (batch_size, q_seqlen, num_heads, v_dim)
        assert A_sbblocksparse.shape == (batch_size, num_heads, q_seqlen, k_seqlen)
        assert torch.all(A_sbblocksparse >= 0)
        # Sum of each row should be either 0.0 or 1.0
        A_sblocal_sum = A_sbblocksparse.sum(dim=-1)
        assert torch.all(torch.isclose(A_sblocal_sum, torch.ones_like(A_sblocal_sum), atol=1e-2)
                         | torch.isclose(A_sblocal_sum, torch.zeros_like(A_sblocal_sum), atol=1e-2))
        assert torch.allclose(out_sblocal, torch.einsum('bhts,bshd->bthd', A_sbblocksparse, v),
                              rtol=1e-3, atol=1e-3)

        temperature = 1 / math.sqrt(embed_dim) if softmax_temp is None else softmax_temp
        A_full_unnormalized = torch.exp(torch.einsum('bthe,bshe->bhts', q * temperature, k))
        if attn_mask is not None:
            A_full_unnormalized.masked_fill_(~attn_mask.bool_matrix, 0.0)
        if key_padding_mask is not None:
            A_full_unnormalized.masked_fill_(rearrange(~key_padding_mask.bool_matrix,
                                                       'b s -> b 1 1 s'),
                                             0.0)
        # Test that A_sbblocksparse_unnormalized matches A_full_unnormalized on the block sparse
        # indices and A_lr_unnormalized on the non-block sparse indices.
        blocksparse_mask = mask_tensor(torch.ones_like(A_full_unnormalized), layout).bool()
        assert torch.allclose(A_sbblocksparse_unnormalized.masked_select(blocksparse_mask),
                              A_full_unnormalized.masked_select(blocksparse_mask),
                              rtol=1e-3, atol=1e-4)
        assert torch.allclose(A_sbblocksparse_unnormalized.masked_select(~blocksparse_mask),
                              A_lr_unnormalized.masked_select(~blocksparse_mask),
                              rtol=1e-3, atol=1e-4)

        rel_error = ((A_sbblocksparse_unnormalized - A_full_unnormalized)
                        / A_full_unnormalized.clamp_min_(1e-6)).abs().mean()
        if device == 'cuda':
            print(f'Relative error of attention matrix: {rel_error}')
