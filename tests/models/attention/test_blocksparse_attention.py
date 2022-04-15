import math
import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat

# from triton.ops.blocksparse import matmul, softmax
from triton.ops.blocksparse import softmax
from deepspeed.ops.sparse_attention import FixedSparsityConfig

from src.models.modules.masking import FullMask, LengthMask

from src.models.attention.full_attention import FullAttention
from src.models.attention.blocksparse_attention import BlockSparseAttention
from src.models.attention.blocksparse_utils import sparsify_tensor, densify_tensor
from src.models.attention.blocksparse_utils import mask_tensor
from src.utils.padding import pad_to_multiple
from src.models.attention.blocksparse_matmul import matmul


def seed_cpu_cuda(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class TestBlockSparseAttention:

    # Adapted from https://github.com/openai/triton/blob/8bedcce9befbbe95d8fe0a082718edc4050e2831/python/test/operators/test_blocksparse.py#L13
    def test_blocksparse_matmul(self):
        seed = 2357
        block_size = 16
        batch_size = 2
        n_head = 3
        seqlen_q = 64
        seqlen_k = 48
        dim = 32
        softmax_temp = 0.23

        # create inputs
        seed_cpu_cuda(seed)
        q = torch.randn((batch_size, n_head, seqlen_q, dim), device="cuda", requires_grad=True)
        k = torch.randn((batch_size, n_head, seqlen_k, dim), device="cuda", requires_grad=True)
        v = torch.randn((batch_size, n_head, seqlen_k, dim), device="cuda", requires_grad=True)
        shape = (seqlen_q, seqlen_k)
        p, r = shape[0] // block_size, shape[1] // block_size
        layout = torch.randint(2, (n_head, p, r))  # On cpu
        nnz = layout.bool().sum()

        # triton result
        matmul_sdd_op = matmul(layout, block_size, 'sdd', trans_a=False, trans_b=True)
        qk_sparse = matmul_sdd_op(q, k)
        qk_dense = densify_tensor(qk_sparse, layout)
        assert qk_sparse.shape == (batch_size, nnz, block_size, block_size)
        assert qk_dense.shape == (batch_size, n_head, seqlen_q, seqlen_k)
        softmax_op = softmax(layout, block_size)
        attn = softmax_op(qk_sparse.clone(), scale=softmax_temp)
        matmul_dsd_op = matmul(layout, block_size, 'dsd', trans_a=False, trans_b=False)
        out = matmul_dsd_op(attn, v)

        # torch result
        qk_dense_pt = q @ k.transpose(-1, -2)
        qk_dense_masked_pt = mask_tensor(qk_dense_pt, layout)
        qk_sparse_pt = sparsify_tensor(qk_dense_pt, layout)
        assert qk_sparse_pt.shape == (batch_size, nnz, block_size, block_size)
        assert qk_dense_masked_pt.shape == (batch_size, n_head, seqlen_q, seqlen_k)
        qk_dense_inf_filled_pt = mask_tensor(qk_dense_pt, layout, value=float('-inf'))
        attn_pt = torch.softmax(qk_dense_inf_filled_pt * softmax_temp, dim=-1)
        # Some rows could be all zero, and torch.softmax will fill those with NaNs
        zero_rows = (~(layout.to(attn_pt.device).bool())).all(dim=-1, keepdims=True)
        zero_rows = repeat(zero_rows, 'h p 1 -> h (p blk_sz) 1', blk_sz=block_size)
        attn_pt.masked_fill_(zero_rows, 0.0)
        out_pt = attn_pt @ v

        # Compare results
        assert torch.allclose(qk_dense, qk_dense_masked_pt)
        assert torch.allclose(qk_sparse, qk_sparse_pt)
        assert torch.allclose(densify_tensor(attn, layout), attn_pt)
        assert torch.allclose(out, out_pt, rtol=1e-5, atol=1e-7)

    @pytest.mark.parametrize('device', ['cuda'])
    @pytest.mark.parametrize('softmax_temp', [None, 0.1, 0.235])
    @pytest.mark.parametrize('dim', [15, 33, 51])
    @pytest.mark.parametrize('num_local_blocks', [1, 3, 4])
    @pytest.mark.parametrize('block_size', [16, 32])
    def test_output(self, block_size, num_local_blocks, dim, softmax_temp, device):
        seed = 2357
        embed_dim = dim
        v_dim = 17
        num_heads = 7
        batch_size = 18
        q_seqlen = 235
        k_seqlen = 179
        q_seqlen_padded = int(math.ceil(q_seqlen / block_size) * block_size)
        k_seqlen_padded = int(math.ceil(k_seqlen / block_size) * block_size)

        seed_cpu_cuda(seed)
        attn_mask = FullMask(torch.randint(low=0, high=2, size=(q_seqlen, k_seqlen),
                                           dtype=torch.bool, device=device))
        # attn_mask = None
        key_padding_mask = LengthMask(torch.randint(low=0, high=k_seqlen, size=(batch_size,),
                                                    device=device), max_len=k_seqlen)
        # key_padding_mask = None
        sparsity_config = dict(
            _target_='deepspeed.ops.sparse_attention.FixedSparsityConfig',
            num_heads=num_heads,
            block=block_size,
            num_local_blocks=num_local_blocks
        )
        # sparsity_config = FixedSparsityConfig(num_heads = num_heads, block=block_size,
        #                                       num_local_blocks=num_local_blocks)
        blocksparse_attn = BlockSparseAttention(
            sparsity_config, softmax_temp=softmax_temp, attention_dropout=0.0,
            max_seq_length=max(q_seqlen_padded, k_seqlen_padded)
        ).to(device)
        full_attn = FullAttention(softmax_temp=softmax_temp, attention_dropout=0.0).to(device)
        q = torch.randn(batch_size, q_seqlen, num_heads, embed_dim, device=device)
        k = torch.randn(batch_size, k_seqlen, num_heads, embed_dim, device=device)
        v = torch.randn(batch_size, k_seqlen, num_heads, v_dim, device=device)
        layout = blocksparse_attn.get_layout(q_seqlen_padded, k_seqlen_padded)

        out_bs, A_bs = blocksparse_attn(q, k, v, attn_mask, key_padding_mask, need_weights=True)
        assert out_bs.shape == (batch_size, q_seqlen, num_heads, v_dim)
        assert A_bs.shape == (batch_size, num_heads, q_seqlen, k_seqlen)
        assert torch.all(A_bs >= 0)
        # Sum of each row should be either 0.0 or 1.0
        A_bs_sum = A_bs.sum(dim=-1)
        assert torch.all(torch.isclose(A_bs_sum, torch.ones_like(A_bs_sum))
                         | torch.isclose(A_bs_sum, torch.zeros_like(A_bs_sum)))
        assert torch.allclose(out_bs, torch.einsum('bhts,bshd->bthd', A_bs, v),
                              rtol=1e-6, atol=1e-7)

        _, A_full = full_attn(q, k, v, attn_mask, key_padding_mask, need_weights=True)
        A_full.nan_to_num_()
        # Test that A_bs is equivalent to zero-ing out some blocks in A_full and then
        # re-normalize so each row sums to 1
        A_full_padded = pad_to_multiple(A_full, block_size, dims=(-1, -2))
        A_full_padded_masked = mask_tensor(A_full_padded, layout)
        A_full_bs = F.normalize(A_full_padded_masked, p=1, dim=-1, eps=1e-8)[:, :, :q_seqlen, :k_seqlen]
        assert torch.allclose(A_bs, A_full_bs, rtol=1e-4, atol=1e-6)
