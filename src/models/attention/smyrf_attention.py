# Adapted from https://github.com/giannisdaras/smyrf/blob/master/smyrf/torch/attn.py
import math
import torch
import torch.nn as nn

from einops import rearrange, repeat

from src.utils.padding import pad_to_multiple
from src.ops.permutation import invert_permutation
from src.models.attention.hash_utils import XBOXPLUS, lsh_clustering
from src.models.attention.batching_utils import batched_index_select
from src.models.attention.reformer_attention import max_neg_value
from src.models.modules.masking import LengthMask
from src.models.attention.mask_utils import pad_mask


class SmyrfAttention(nn.Module):
    def __init__(self, n_hashes, q_cluster_size, k_cluster_size,
                 r=1, # LSH clustering
                 softmax_temp=None, attention_dropout=0., device=None, dtype=None):
        super().__init__()
        self.n_hashes = n_hashes
        self.q_cluster_size = q_cluster_size
        self.k_cluster_size = k_cluster_size
        self.softmax_temp = softmax_temp
        self.dropout = nn.Dropout(attention_dropout)
        self.hash_fn = XBOXPLUS()
        self.clustering_params = {'r': r, 'n_hashes': self.n_hashes}

    def hash_vectors(self, query, key, key_padding_mask=None):
        # XBOX+ transform
        self.hash_fn.set_norms(query, key)
        query_t = self.hash_fn.Q(query)
        key_t = self.hash_fn.K(key)

        num_clusters = query_t.shape[-2] // self.q_cluster_size
        assert num_clusters == (key_t.shape[-2] // self.k_cluster_size), 'Unequal number of clusters for query and key.'
        q_positions, k_positions = lsh_clustering(query_t, key_t, **self.clustering_params,
                                                  key_padding_mask=key_padding_mask)
        return q_positions, k_positions

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=False):
        if attn_mask is not None:
            raise NotImplementedError('Support for attn_mask is not implemented')
        _, q_seqlen_og, _, _ = query.shape
        _, k_seqlen_og, _, _ = key.shape
        query = pad_to_multiple(query, self.q_cluster_size, dims=1)
        key = pad_to_multiple(key, self.k_cluster_size, dims=1)
        value = pad_to_multiple(value, self.k_cluster_size, dims=1)

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

        query = rearrange(query, 'b t h e -> (b h) t e')
        key = rearrange(key, 'b t h e -> (b h) t e')
        value = rearrange(value, 'b s h d -> (b h) s d')
        bs = query.shape[0]

        if key_padding_mask is not None and not key_padding_mask.all_ones:
            # Repeat for all heads
            key_padding_mask_bool = repeat(key_padding_mask.bool_matrix, 'b s -> (b h) s', h=H)
        else:
            key_padding_mask_bool = None

        with torch.no_grad():
            q_positions, k_positions = self.hash_vectors(query, key,
                                                         rearrange(key_padding_mask_bool,
                                                                   'b s -> 1 b s')
                                                         if key_padding_mask_bool is not None else None)

        # sort queries, keys, values
        def sort_to_buckets(x, perm, bucketsz):
            return rearrange(batched_index_select(rearrange(x, 'b s d -> 1 b s d'), perm),
                             'h b (nbuckets bucketsz) d -> h b nbuckets bucketsz d',
                             bucketsz=bucketsz)

        s_query = sort_to_buckets(query, q_positions, self.q_cluster_size)
        s_key = sort_to_buckets(key, k_positions, self.k_cluster_size)
        s_value = sort_to_buckets(value, k_positions, self.k_cluster_size)

        inner = torch.einsum('...id,...jd->...ij', s_query, s_key) * softmax_temp

        masked_value = max_neg_value(inner)
        # mask out attention to padded tokens
        if key_padding_mask is not None and not key_padding_mask.all_ones:
            s_key_padding_mask = sort_to_buckets(rearrange(key_padding_mask_bool,
                                                           'b s -> b s 1'),
                                                 k_positions, self.k_cluster_size)
            s_key_padding_mask = rearrange(s_key_padding_mask,
                                           '... bucketsz 1 -> ... 1 bucketsz')
            inner.masked_fill_(~s_key_padding_mask, masked_value)

        q_rev_positions = invert_permutation(q_positions)
        # free memory
        if not need_weights:
            del q_positions, k_positions

        # softmax denominator
        dots_logsumexp = torch.logsumexp(inner, dim=-1, keepdim=True)
        # softmax
        dots = torch.exp(inner - dots_logsumexp)
        # If the whole row within this bucket is masked out, then inner is the uniform distribution.
        # We actually want it to be zero.
        if key_padding_mask is not None and not key_padding_mask.all_ones:
            full_row_mask = (inner <= masked_value).all(dim=-1, keepdim=True)
            dots = dots.masked_fill(full_row_mask, 0.0)

        # dropout
        dropped_dots = self.dropout(dots)

        # n_hashes outs
        so = torch.einsum('...ij,...jd->...id', dropped_dots, s_value)

        # undo sort
        def unsort_from_buckets(s_x, perm_inverse):
            b_x = rearrange(s_x, 'h b nbuckets bucketsz d -> h b (nbuckets bucketsz) d')
            return batched_index_select(b_x, perm_inverse)

        o = unsort_from_buckets(so, q_rev_positions)
        logits = unsort_from_buckets(dots_logsumexp, q_rev_positions)

        # free memory
        del q_rev_positions

        probs = torch.exp(logits - torch.logsumexp(logits, dim=0, keepdim=True))
        out = torch.sum(o * probs, dim=0)
        out = rearrange(out, '(b h) t d -> b t h d', h=H)
        out = out[:, :q_seqlen_og]

        attn = None
        if need_weights:
            q_pos_2d = rearrange(q_positions, 'h b (nbuckets bucketsz) -> h b nbuckets bucketsz 1',
                                 bucketsz=self.q_cluster_size)
            k_pos_2d = rearrange(k_positions, 'h b (nbuckets bucketsz) -> h b nbuckets 1 bucketsz',
                                 bucketsz=self.k_cluster_size)
            pos_2d = rearrange(q_pos_2d * S + k_pos_2d,
                               'h b nbuckets qbucketsz kbucketsz -> h b (nbuckets qbucketsz kbucketsz)')
            unsorted_dots = torch.zeros(self.n_hashes, bs, T * S, device=query.device)
            unsorted_dots.scatter_(-1, pos_2d, dots.view_as(pos_2d))
            del pos_2d
            unsorted_dots = rearrange(unsorted_dots,
                                      'h b (q_seqlen k_seqlen) -> h b q_seqlen k_seqlen',
                                      q_seqlen=T)
            attn = torch.sum(unsorted_dots * probs, dim=0)
            attn = rearrange(attn, '(b h) t s -> b h t s', h=H)[:, :, :q_seqlen_og, :k_seqlen_og]

        return out, attn
