# Adapted from https://github.com/giannisdaras/smyrf/blob/master/smyrf/torch/attn.py
import math
import torch
import torch.nn as nn

from einops import rearrange, repeat

from src.utils.padding import pad_to_multiple
from src.ops.permutation import invert_permutation
from src.models.attention.hash_utils import XBOXPLUS, lsh_clustering
from src.models.attention.batching_utils import batched_index_select
from src.models.attention.reformer_attention import max_neg_value, chunked_sum
from src.models.modules.masking import LengthMask
from src.models.attention.mask_utils import pad_mask
from src.models.attention.feature_maps_sb import SBPerformerFeatures
from src.models.attention.scatterbrain_utils import linear_attention_normalization
from src.models.attention.scatterbrain_utils import causal_linear_attention, linear_attention


class SBSmyrfAttention(nn.Module):
    def __init__(self, n_hashes, q_cluster_size, k_cluster_size, d_head,
                 r=1, # LSH clustering
                 nb_features=None, ortho_scaling=0,  softmax_eps=0.0, # Performer
                 causal=False, softmax_temp=None, attention_dropout=0., device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.feature_map = SBPerformerFeatures(d_head, nb_features, ortho_scaling=ortho_scaling,
                                               softmax_temp=softmax_temp, eps=softmax_eps,
                                               **factory_kwargs)
        self.n_hashes = n_hashes
        self.q_cluster_size = q_cluster_size
        self.k_cluster_size = k_cluster_size
        self.causal = causal
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

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=False,
                return_attn_unnormalized=False):
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


        self.feature_map.new_feature_map(query.device)
        q_prime, q_prime_log_scale = self.feature_map.forward_queries(query)
        k_prime, k_prime_log_scale = self.feature_map.forward_keys(key)

        prime_log_scale = q_prime_log_scale + k_prime_log_scale
        m = q_prime.shape[-1]
        if key_padding_mask_bool is not None:
            k_prime.masked_fill_(~rearrange(key_padding_mask_bool, 'b s -> b s 1'), 0.0)
        if self.causal:
            try:
                import fast_transformers.causal_product.causal_product_cuda
                causal_linear_fn = causal_linear_attention
            except ImportError:
                print('unable to import cuda code for auto-regressive Performer. will default to the memory inefficient non-cuda version')
                # self.causal_linear_fn = causal_linear_attention_noncuda
        attn_fn = linear_attention if not self.causal else causal_linear_attention
        q_prime_k_prime_1 = linear_attention_normalization(q_prime, k_prime, causal=self.causal)
        q_prime_k_prime_v, attn_prime = attn_fn(q_prime, k_prime, value, need_weights=need_weights)

        # sort queries, keys, values
        def sort_to_buckets(x, perm, bucketsz):
            return rearrange(batched_index_select(rearrange(x, 'b s d -> 1 b s d'), perm),
                             'h b (nbuckets bucketsz) d -> h b nbuckets bucketsz d',
                             bucketsz=bucketsz)

        s_query = sort_to_buckets(query, q_positions, self.q_cluster_size)
        s_key = sort_to_buckets(key, k_positions, self.k_cluster_size)
        s_value = sort_to_buckets(value, k_positions, self.k_cluster_size)
        sq_prime = sort_to_buckets(q_prime, q_positions, self.q_cluster_size)
        sk_prime = sort_to_buckets(k_prime, k_positions, self.k_cluster_size)
        # sq_prime, sq_prime_log_scale = kernel_fn(s_queries, is_query=True)
        # sk_prime, sk_prime_log_scale = kernel_fn(s_keys, is_query=False)
        # k_prime_log_scale doesn't depend on the index of the token
        sprime_log_scale = sort_to_buckets(prime_log_scale, q_positions, self.q_cluster_size)
        # sprime_log_scale = sq_prime_log_scale + sk_prime_log_scale

        inner = torch.einsum('...id,...jd->...ij', s_query, s_key) * softmax_temp
        dots_prime = torch.einsum('...im,...jm->...ij', sq_prime, sk_prime)

        masked_value = max_neg_value(inner)
        # mask out attention to padded tokens
        if key_padding_mask is not None and not key_padding_mask.all_ones:
            s_key_padding_mask = sort_to_buckets(rearrange(key_padding_mask_bool,
                                                           'b s -> b s 1'),
                                                 k_positions, self.k_cluster_size)
            s_key_padding_mask = rearrange(s_key_padding_mask,
                                           '... bucketsz 1 -> ... 1 bucketsz')
            inner.masked_fill_(~s_key_padding_mask, masked_value)
            dots_prime.masked_fill_(~s_key_padding_mask, 0.0)

        # Causal masking
        if self.causal:
            s_q_positions = rearrange(q_positions,
                                      'h b (nbuckets bucketsz) -> h b nbuckets bucketsz 1',
                                      bucketsz=self.q_cluster_size)
            s_k_positions = rearrange(k_positions,
                                      'h b (nbuckets bucketsz) -> h b nbuckets 1 bucketsz',
                                      bucketsz=self.k_cluster_size)
            causal_mask = s_q_positions < s_k_positions
            inner.masked_fill_(causal_mask, masked_value)
            dots_prime.masked_fill_(causal_mask, 0.0)
            del causal_mask

        q_rev_positions = invert_permutation(q_positions)

        # Don't double-count query-key pairs across multiple rounds of hashing.
        # Count how many times a query-key pair is repeated, and to lower its log-prob
        # correspondingly at each repetition.
        if self.n_hashes > 1:
            k_rev_positions = invert_permutation(k_positions)
            q_bucket_idx = rearrange(q_rev_positions // self.q_cluster_size,
                                     'h b seqlen -> b seqlen h')
            k_bucket_idx = rearrange(k_rev_positions // self.k_cluster_size,
                                     'h b seqlen -> b seqlen h')
            s_q_bucket_idx = sort_to_buckets(q_bucket_idx, q_positions, self.q_cluster_size)
            s_k_bucket_idx = sort_to_buckets(k_bucket_idx, k_positions, self.k_cluster_size)
            dup_counts = (rearrange(s_q_bucket_idx, '... bk_size h -> ... bk_size 1 h') ==
                          rearrange(s_k_bucket_idx, '... bk_size h -> ... 1 bk_size h'))
            # for memory considerations, chunk summation of last dimension for counting duplicates
            dup_counts = chunked_sum(dup_counts, chunks=(self.n_hashes * bs))
            dup_counts = dup_counts.detach()
            assert dup_counts.shape == inner.shape
            inner = inner - torch.log(dup_counts.float())
            dots_prime = dots_prime / dup_counts

        # free memory
        if not need_weights:
            del q_positions, k_positions

        # softmax denominator
        # TD: Even though we call this dots_logsumexp, it can be of arbitrary value and the
        # computation would still be correct (assuming infinite precision), since it's just an
        # arbitrary scaling of @dots.
        # Here we choose it for numerical stability: we want torch.exp(inner - dots_logsumexp) <= 1.0
        # and torch.exp(spring_log_scale - dots_logsumexp) <= 1.0
        # dots_logsumexp = torch.logsumexp(inner, dim=-1, keepdim=True)
        dots_logsumexp = torch.maximum(torch.amax(inner, dim=-1, keepdim=True), sprime_log_scale)
        # TD: dots and dots_sum has log scale dots_logsumexp
        # TD: No longer need this because we pick dots_logsumexp to not be -inf
        # dots_prime_scale = torch.exp(sprime_log_scale - dots_logsumexp)
        # nan_q_indices = dots_prime_scale.isinf()
        # # dots_logsumexp[nan_q_indices] = 0.0
        # dots_logsumexp = torch.where(nan_q_indices, torch.tensor(0.0, device=dots_logsumexp.device),
        #                              dots_logsumexp)
        dots_prime_scale = torch.exp(sprime_log_scale - dots_logsumexp)
        dots = torch.exp(inner - dots_logsumexp) - dots_prime * dots_prime_scale
        # TD: No longer need this because we pick dots_logsumexp to not be -inf
        # If the whole row within this bucket is masked out, then inner is the uniform distribution.
        # We actually want it to be zero.
        # if key_padding_mask is not None and not key_padding_mask.all_ones:
        #     full_row_mask = (inner <= masked_value).all(dim=-1, keepdim=True)
        #     dots = dots.masked_fill(full_row_mask, 0.0)
        dots_sum = dots.sum(dim=-1, keepdim=True)

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
        dots_sum_unsort = unsort_from_buckets(dots_sum, q_rev_positions)

        # free memory
        del q_rev_positions

        normalization_log_scale = torch.logsumexp(logits, dim=0)
        probs = torch.exp(logits - rearrange(normalization_log_scale, '... -> 1 ...'))
        out_lsh = torch.sum(o * probs, dim=0)

        prime_scale = torch.exp(prime_log_scale - normalization_log_scale)
        out = out_lsh + q_prime_k_prime_v * prime_scale

        normalization = (dots_sum_unsort * probs).sum(dim=0) + q_prime_k_prime_1.unsqueeze(-1) * prime_scale
        out_normalized = out / normalization.clamp_min(1e-6)
        out_normalized = (rearrange(out_normalized, '(b h) t d -> b t h d', h=H))[:, :q_seqlen_og]

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
            attn_lsh = torch.sum(unsorted_dots * probs, dim=0)
            attn_unnormalized = attn_lsh + attn_prime * prime_scale
            attn = attn_unnormalized / normalization.clamp_min(1e-6)
            attn = rearrange(attn, '(b h) t s -> b h t s', h=H)[:, :, :q_seqlen_og, :k_seqlen_og]
            if return_attn_unnormalized:  # For testing purpose
                attn_unnormalized = rearrange(
                    attn_unnormalized, '(b h) t s -> b h t s', h=H
                )[:, :, :q_seqlen_og, :k_seqlen_og]
                normalization_log_scale = rearrange(normalization_log_scale,
                                                    '(b h) s 1 -> b h s 1', h=H)[:, :, :q_seqlen_og]
                attn_prime = rearrange(attn_prime,
                                       '(b h) s d -> b h s d', h=H)[:, :, :q_seqlen_og, :k_seqlen_og]
                prime_log_scale = rearrange(prime_log_scale,
                                            '(b h) s 1 -> b h s 1', h=H)[:, :, :q_seqlen_og]
                smyrf_mask = rearrange(attn_lsh != 0.0,
                                       '(b h) t s -> b h t s', h=H)[:, :, :q_seqlen_og, :k_seqlen_og]
                attn = (attn, attn_unnormalized * torch.exp(normalization_log_scale),
                        attn_prime * torch.exp(prime_log_scale), smyrf_mask)

        return out_normalized, attn
