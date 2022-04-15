import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist

from einops import rearrange

import hydra

from triton.ops.blocksparse import softmax

from deepspeed.ops.sparse_attention import FixedSparsityConfig
from fast_transformers.local_product import local_dot_product, local_weighted_average
from src.models.modules.masking import FullMask, LengthMask

from src.models.attention.feature_maps_sb import SBPerformerFeatures
from src.models.attention.scatterbrain_utils import linear_attention_normalization
from src.models.attention.scatterbrain_utils import causal_linear_attention, linear_attention

from src.utils.padding import pad_to_multiple
from src.models.attention.blocksparse_utils import densify_tensor
from src.models.attention.blocksparse_utils import sparsify_broadcast_tensor
from src.models.attention.mask_utils import pad_mask
from src.models.attention.blocksparse_matmul import matmul
from src.models.attention.blocksparse_logsumexp import logsumexp
from src.models.attention.blocksparse_sum import blocksparse_sum


class SBBlockSparseAttention(nn.Module):
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
    def __init__(self, sparsity_config, dim_heads, nb_features=None, ortho_scaling=0,
                 causal=False, softmax_temp=None, attention_dropout=0.0, softmax_eps=0.0,
                 max_seq_length=2048):
        super().__init__()
        self.sparsity_config = hydra.utils.instantiate(sparsity_config)
        # TODO: nb_features should be a multiple of block
        self.feature_map = SBPerformerFeatures(dim_heads, nb_features, ortho_scaling=ortho_scaling,
                                               softmax_temp=softmax_temp, eps=softmax_eps)
        self.causal = causal
        self.softmax_temp = softmax_temp
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax_eps = softmax_eps

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

            sparse_logsumexp = logsumexp(sparsity_layout, self.sparsity_config.block)
            sparse_sum = blocksparse_sum(sparsity_layout, self.sparsity_config.block)
            sparse_softmax = softmax(sparsity_layout, self.sparsity_config.block)

            self.ops_cache[T, S] = (sparse_dot_sdd_nt, sparse_dot_dsd_nn, sparse_softmax,
                                    sparse_logsumexp, sparse_sum)
        return self.ops_cache[T, S]

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=False,
                return_attn_unnormalized=False):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            query: (B, T, H, E) The tensor containing the query
            key: (B, S, H, E) The tensor containing the key
            value: (B, S, H, D) The tensor containing the value
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            key_padding_mask: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of
        """
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
        softmax_temp = self.softmax_temp or 1 / math.sqrt(E)

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

        # TODO: check causal
        if attn_mask is None:
            attn_mask_additive_matrix = torch.zeros(T, S, device=query.device)
        else:
            attn_mask_additive_matrix = attn_mask.additive_matrix_finite
        if key_padding_mask is None:
            key_padding_mask_lengths = torch.full(size=(B,), fill_value=S, dtype=torch.long,
                                                  device=key.device)
        else:
            key_padding_mask_lengths = key_padding_mask.lengths

        # Permute the dimensions to BHTE instead of BTHE
        query = rearrange(query, 'b t h e -> b h t e').contiguous()
        key = rearrange(key, 'b s h e -> b h s e').contiguous()
        value = rearrange(value, 'b s h d -> b h s d').contiguous()

        self.feature_map.new_feature_map(query.device)
        q_prime, q_prime_log_scale = self.feature_map.forward_queries(query)
        k_prime, k_prime_log_scale = self.feature_map.forward_keys(key)

        prime_log_scale = q_prime_log_scale + k_prime_log_scale
        m = q_prime.shape[-1]
        if key_padding_mask is not None:
            k_prime = k_prime.masked_fill(~rearrange(key_padding_mask.bool_matrix,
                                                     'b s -> b 1 s 1'), 0.0)
        attn_fn = linear_attention if not self.causal else causal_linear_attention
        q_prime_k_prime_1 = linear_attention_normalization(q_prime, k_prime, causal=self.causal)
        q_prime_k_prime_v, attn_prime = attn_fn(q_prime, k_prime, value, need_weights=need_weights)

        # cache look-up table computations etc
        layout = self.get_layout(T, S)
        (sparse_dot_sdd_nt, sparse_dot_dsd_nn, sparse_softmax, sparse_logsumexp,
         sparse_sum) = self.get_ops(T, S)

        QK = softmax_temp * sparse_dot_sdd_nt(query, key)
        dots_prime = sparse_dot_sdd_nt(q_prime, k_prime)
        assert attn_mask is None and key_padding_mask is None
        # TODO: sparse masked_fill_

        # Compute the normalization first
        QK_lse = rearrange(sparse_logsumexp(QK), 'b h t -> b h t 1')
        dots_prime_sum = rearrange(sparse_sum(dots_prime), 'b h t -> b h t 1')
        lr_log_normalization = torch.log((rearrange(q_prime_k_prime_1, 'b h s -> b h s 1')
                                          - dots_prime_sum).clamp_min_(1e-24)) + prime_log_scale
        log_normalization = torch.logaddexp(QK_lse, lr_log_normalization)

        prime_scale = torch.exp(prime_log_scale - log_normalization)
        # When we drop out, we want that location in the attn matrix to be zero.
        # So here we dropout just torch.exp(QK) and leave -dots_prime, so that when we add it back
        # to attn_prime it's equivalent to setting that location to zero.
        log_normalization_sparse = sparsify_broadcast_tensor(log_normalization.squeeze(-1), layout,
                                                             block)
        prime_scale_sparse = sparsify_broadcast_tensor(prime_scale.squeeze(-1), layout, block)
        dots = self.dropout(torch.exp(QK - log_normalization_sparse)) - dots_prime * prime_scale_sparse

        # outputs
        out_blocksparse = sparse_dot_dsd_nn(dots, value)
        out = out_blocksparse + q_prime_k_prime_v * prime_scale

        attn = None
        if need_weights:
            attn_blocksparse = densify_tensor(dots, layout)
            attn = attn_blocksparse + attn_prime * prime_scale
            if return_attn_unnormalized:  # For testing purpose
                attn = (attn, attn * torch.exp(log_normalization),
                        attn_prime * torch.exp(prime_log_scale))
        return rearrange(out, 'b h t d -> b t h d'), attn
