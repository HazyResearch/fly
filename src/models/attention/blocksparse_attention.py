import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist

from einops import rearrange

import hydra

from deepspeed.ops.sparse_attention import SparsityConfig
# from triton.ops.blocksparse import matmul, softmax
from triton.ops.blocksparse import softmax
from src.models.modules.masking import FullMask, LengthMask

from src.utils.padding import pad_to_multiple
from src.models.attention.blocksparse_utils import sparsify_tensor, densify_tensor
from src.models.attention.mask_utils import pad_mask
from src.models.attention.blocksparse_matmul import matmul


class BlockSparseAttention(nn.Module):
    """Implements an efficient Sparse Self Attention of Transformer layer based on `Generative Modeling with Sparse Transformers`: https://arxiv.org/abs/1904.10509
    For more information please see, TODO DeepSpeed Sparse Transformer.
    For usage example please see, TODO DeepSpeed Sparse Transformer Tutorial.
    Arguments
    ---------
        sparsity_config: optional: this parameter determins sparsity pattern configuration; it is based on SparsityConfig class.
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
        max_seq_length: optional: the maximum sequence length this sparse attention module will be applied to; it controls the size of the master_layout.
    """
    def __init__(self, sparsity_config, softmax_temp=None, attention_dropout=0.1, max_seq_length=2048):
        super().__init__()
        self.sparsity_config = hydra.utils.instantiate(sparsity_config)
        self.softmax_temp = softmax_temp
        self.dropout = nn.Dropout(attention_dropout)
        # initialize sparse layout and register as buffer
        master_layout = self.sparsity_config.make_layout(max_seq_length)
        self.register_buffer("master_layout", master_layout)
        self._need_layout_synchronization = True
        self.ops_cache = dict()

    def get_layout(self, T, S=None):
        if S is None:
            S = T
        # if layout is never synchronized across GPUs, broadcast the layout from global rank 0
        if self._need_layout_synchronization and dist.is_initialized():
            dist.broadcast(self.master_layout, src=0)
            self._need_layout_synchronization = False

        block = self.sparsity_config.block
        if (T % block != 0) or (S % block != 0):
            raise ValueError(
                f'Sequence lengths, {T} and {S}, needs to be divisible by block size {block}!'
            )

        num_blocksT, num_blocksS = T // self.sparsity_config.block, S // self.sparsity_config.block
        return self.master_layout[..., :num_blocksT, :num_blocksS].cpu()  # layout needs to be a CPU tensor

    # add to cache
    def get_ops(self, T, S=None):
        if S is None:
            S = T
        if (T, S) not in self.ops_cache:
            sparsity_layout = self.get_layout(T, S)
            sparse_dot_sdd_nt = matmul(sparsity_layout,
                                       self.sparsity_config.block,
                                       'sdd',
                                       trans_a=False,
                                       trans_b=True)

            sparse_dot_dsd_nn = matmul(sparsity_layout,
                                       self.sparsity_config.block,
                                       'dsd',
                                       trans_a=False,
                                       trans_b=False)

            sparse_softmax = softmax(sparsity_layout, self.sparsity_config.block)

            self.ops_cache[T, S] = (sparse_dot_sdd_nt, sparse_dot_dsd_nn, sparse_softmax)
        return self.ops_cache[T, S]

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=False):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            query: (B, T, H, E) The tensor containing the query
            key: (B, S, H, E) The tensor containing the key
            value: (B, S, H, D) The tensor containing the value
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to  (seqlen, seqlen)
            key_padding_mask: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of. (bs, seqlen)
        """
        # if attn_mask is None:
        #     attn_mask = FullMask(N=query.size(1), M=key.size(1), device=query.device)
        block = self.sparsity_config.block
        # Pad to multiples of @block
        _, q_seqlen_og, n_head, q_dim_og = query.shape
        _, k_seqlen_og, _, _ = key.shape
        _, _, _, v_dim_og = value.shape
        query = pad_to_multiple(query, block, dims=(1, 3))
        key = pad_to_multiple(key, block, dims=(1, 3))
        value = pad_to_multiple(value, block, dims=(1, 3))

        # Extract some shapes and compute the temperature
        B, T, H, E = query.shape
        _, S, _, D = value.shape
        softmax_temp = self.softmax_temp or 1 / math.sqrt(q_dim_og)

        # pad the masks
        if S > k_seqlen_og:
            if key_padding_mask is None:
                key_padding_mask = LengthMask(key.new_full((key.shape[0],), k_seqlen_og,
                                                           dtype=torch.long), max_len=S)
            else:
                key_padding_mask = pad_mask(key_padding_mask, pad_length=S - k_seqlen_og,
                                            left=False, value=False)
        # TODO: this doesn't work for causal right now
        if attn_mask is not None and (S > k_seqlen_og or T > q_seqlen_og):
            attn_mask = FullMask(F.pad(attn_mask._mask, (0, S - k_seqlen_og, 0, T - q_seqlen_og),
                                       value=False))

        # Permute the dimensions to BHTE instead of BTHE
        query = rearrange(query, 'b t h e -> b h t e').contiguous()
        key = rearrange(key, 'b s h e -> b h s e').contiguous()
        value = rearrange(value, 'b s h d -> b h s d').contiguous()

        # cache look-up table computations etc
        layout = self.get_layout(T, S)
        sparse_dot_sdd_nt, sparse_dot_dsd_nn, sparse_softmax = self.get_ops(T, S)
        # [2021-09-05] TD: There's a bug in triton that gives wrong result when block size is 16
        # and dim is an odd multiple of 16. https://github.com/openai/triton/issues/266
        # [2021-09-06] I've fixed it in my version of blocksparse matmul.
        # if block == 16 and (E // 16) % 2 == 1:
        #     # print('Using dense matmul instead of block sparse matmul due to bug in triton')
        #     sparse_dot_sdd_nt = lambda q, k: sparsify_tensor(q @ k.transpose(-1, -2), layout)

        # attention scores
        QK = sparse_dot_sdd_nt(query, key)
        attn_blocksparse = sparse_softmax(  # This is done in-place
            QK,
            scale=softmax_temp,
            key_padding_mask=None if key_padding_mask is None else key_padding_mask.additive_matrix,
            attn_mask=None if attn_mask is None else attn_mask.additive_matrix,
            key_padding_mask_mode='add',
            attn_mask_mode='add')
        # If there are no valid keys for a query (because of sparse mask, key_padding_mask, and
        # attention mask), then that row of attn_local will be NaN.
        # We want that row to actually be zero.
        if ((key_padding_mask is not None and not key_padding_mask.all_ones)
            or (attn_mask is not None and not attn_mask.all_ones)):
            attn_blocksparse.nan_to_num_()

        A = self.dropout(attn_blocksparse)
        # outputs
        output = sparse_dot_dsd_nn(A, value)
        attn = None
        if need_weights:
            attn = densify_tensor(attn_blocksparse, layout)[:, :, :q_seqlen_og, :k_seqlen_og]
        return rearrange(output, 'b h t d -> b t h d')[:, :q_seqlen_og, :, :v_dim_og], attn
