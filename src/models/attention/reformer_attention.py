# Adapted from https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/reformer_pytorch.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import wraps

from einops import rearrange, repeat

from src.utils.padding import pad_to_multiple
from src.ops.permutation import invert_permutation
from src.models.attention.hash_utils import sort_key_val
from src.models.attention.batching_utils import batched_index_select
from src.models.modules.masking import FullMask, LengthMask
from src.models.attention.mask_utils import pad_mask

#constants

TOKEN_SELF_ATTN_VALUE = -5e4 # carefully set for half precision to work

# helper fns

def chunked_sum(tensor, chunks=1):
    *orig_size, last_dim = tensor.shape
    tensor = tensor.reshape(-1, last_dim)
    summed_tensors = [c.sum(dim=-1) for c in tensor.chunk(chunks, dim=0)]
    return torch.cat(summed_tensors, dim=0).reshape(orig_size)

def default(val, default_val):
    return default_val if val is None else val

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def cache_method_decorator(cache_attr, cache_namespace, reexecute=False):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, key_namespace=None, fetch=False, set_cache=True, **kwargs):
            namespace_str = str(default(key_namespace, ''))
            _cache = getattr(self, cache_attr)
            _keyname = f'{cache_namespace}:{namespace_str}'

            if fetch:
                val = _cache[_keyname]
                if reexecute:
                    fn(self, *args, **kwargs)
            else:
                val = fn(self, *args, **kwargs)
                if set_cache:
                    setattr(self, cache_attr, {**_cache, **{_keyname: val}})
            return val
        return wrapper
    return inner_fn

# LSH attention as described in https://openreview.net/pdf?id=rkgNKkHtvB
# adapted from trax, stripped to what paper said needed to work
# namely that buckets need to be at least 64 with 8 rounds of hashing
# https://github.com/google/trax/blob/master/trax/layers/research/efficient_attention.py#L442

class ReformerAttention(nn.Module):
    def __init__(self,
                 softmax_temp=None,
                 attention_dropout=0.,
                 bucket_size=64,
                 n_hashes=8,
                 causal=False,
                 allow_duplicate_attention=True,
                 attend_across_buckets=True,
                 rehash_each_round=True,
                 drop_for_hash_rate=0.0,
                 random_rotations_per_head=False,
                 device=None, dtype=None):
        super().__init__()
        self.softmax_temp = softmax_temp
        self.dropout = nn.Dropout(attention_dropout)
        self.dropout_for_hash = nn.Dropout(drop_for_hash_rate)

        assert rehash_each_round or allow_duplicate_attention, (
            'The setting {allow_duplicate_attention=False, rehash_each_round=False}'
            ' is not implemented.')

        self.causal = causal
        self.bucket_size = bucket_size

        self.n_hashes = n_hashes

        self._allow_duplicate_attention = allow_duplicate_attention
        self._attend_across_buckets = attend_across_buckets
        self._rehash_each_round = rehash_each_round
        self._random_rotations_per_head = random_rotations_per_head

        # cache buckets for reversible network, reported by authors to make Reformer work at depth
        self._cache = {}

    @cache_method_decorator('_cache', 'buckets', reexecute=True)
    def hash_vectors(self, n_buckets, vecs):
        batch_size = vecs.shape[0]
        device = vecs.device

        # See https://arxiv.org/pdf/1509.02897.pdf
        # We sample a different random rotation for each round of hashing to
        # decrease the probability of hash misses.
        assert n_buckets % 2 == 0

        rot_size = n_buckets

        rotations_shape = (
            batch_size if self._random_rotations_per_head else 1,
            vecs.shape[-1],
            self.n_hashes if self._rehash_each_round else 1,
            rot_size // 2)

        random_rotations = torch.randn(rotations_shape, dtype=vecs.dtype, device=device).expand(batch_size, -1, -1, -1)

        dropped_vecs = self.dropout_for_hash(vecs)
        rotated_vecs = torch.einsum('btf,bfhi->bhti', dropped_vecs, random_rotations)

        if self._rehash_each_round:
            # rotated_vectors size [batch,n_hash,seq_len,buckets]
            rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
            buckets = torch.argmax(rotated_vecs, dim=-1)
        else:
            rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
            # In this configuration, we map each item to the top self.n_hashes buckets
            rotated_vecs = torch.squeeze(rotated_vecs, 1)
            bucket_range = torch.arange(rotated_vecs.shape[-1], device=device)
            bucket_range = torch.reshape(bucket_range, (1, -1))
            bucket_range = bucket_range.expand_as(rotated_vecs)

            _, buckets = sort_key_val(rotated_vecs, bucket_range, dim=-1)
            # buckets size [batch size, seq_len, buckets]
            buckets = buckets[... , -self.n_hashes:].transpose(1, 2)

        # buckets is now (batch_size, self.n_hashes, seq_len).
        return buckets

    def forward(self, qk, k, v, attn_mask=None, key_padding_mask=None, need_weights=False):
        # Ignoring k, assuming that q = k = qk
        _, seqlen_og, n_head, _ = qk.shape
        qk = pad_to_multiple(qk, self.bucket_size * 2, dims=1)
        v = pad_to_multiple(v, self.bucket_size * 2, dims=1)

        # Extract some shapes and compute the temperature
        B, T, H, E = qk.shape
        _, S, _, D = v.shape
        softmax_temp = self.softmax_temp or 1 / math.sqrt(E)

        # pad the masks
        if S > seqlen_og:
            if key_padding_mask is None:
                key_padding_mask = LengthMask(qk.new_full((qk.shape[0],), seqlen_og,
                                                           dtype=torch.long), max_len=S)
            else:
                key_padding_mask = pad_mask(key_padding_mask, pad_length=S - seqlen_og,
                                            left=False, value=False)
        if key_padding_mask is not None and not key_padding_mask.all_ones:
            # Repeat for all heads and all hash functions
            key_padding_mask_bool = repeat(key_padding_mask.bool_matrix, 'b s -> (b head) s',
                                           head=H)
        else:
            key_padding_mask_bool = None
        if attn_mask is not None and (S > seqlen_og or T > seqlen_og):
            attn_mask = FullMask(F.pad(attn_mask._mask, (0, S - seqlen_og, 0, T - seqlen_og),
                                       value=False))
        if attn_mask is not None and not attn_mask.all_ones:
            attn_mask_bool = attn_mask.bool_matrix  # (T, S)
        else:
            attn_mask_bool = None

        qk = rearrange(qk, 'b t h e -> (b h) t e')
        v = rearrange(v, 'b s h d -> (b h) s d')
        batch_size, seqlen, dim, device = *qk.shape, qk.device

        assert seqlen % (self.bucket_size * 2) == 0, f'Sequence length ({seqlen}) needs to be divisible by target bucket size  x 2 - {self.bucket_size * 2}'

        n_buckets = seqlen // self.bucket_size
        buckets = self.hash_vectors(n_buckets, qk, set_cache=self.training)

        assert buckets.shape[1] == self.n_hashes
        assert buckets.shape[2] == seqlen

        total_hashes = self.n_hashes

        buckets = rearrange(buckets, 'b nhashes seqlen -> nhashes b seqlen')
        s_buckets, perm = torch.sort(buckets, dim=-1, stable=True)

        perm_inv = invert_permutation(perm)

        # Allow each chunk to attend within itself, and also one chunk back. Chunk
        # boundaries might occur in the middle of a sequence of items from the
        # same bucket, so this increases the chances of attending to relevant items.
        def look_one_back(x):
            x_extra = torch.cat([x[:, -1:, ...], x[:, :-1, ...]], dim=1)
            return torch.cat([x, x_extra], dim=2)

        # We differ here from https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/reformer_pytorch.py
        # the look_one_back only looks back at the bucket from the same hash function, while
        # lucidrains's implementation could look back at the bucket from the previous hash function.
        perm_oneback = look_one_back(rearrange(perm, 'h b (nbuckets bucketsz) '
                                                     '-> (h b) nbuckets bucketsz',
                                               nbuckets=n_buckets))
        perm_oneback = rearrange(perm_oneback, '(h b) nbuckets2 bucketsz -> h b (nbuckets2 bucketsz)',
                                 h=self.n_hashes)

        # sort queries, keys, values
        def sort_to_buckets(x, perm, bucketsz, unsqueeze=True):
            if unsqueeze:
                x = rearrange(x, 'b s d -> 1 b s d')
            return rearrange(batched_index_select(x, perm),
                             'h b (nbuckets bucketsz) d -> h b nbuckets bucketsz d',
                             bucketsz=bucketsz)

        qk_norm = F.normalize(qk, p=2, dim=-1).type_as(qk)
        tq = sort_to_buckets(qk, perm, self.bucket_size)
        tk = sort_to_buckets(qk_norm, perm_oneback, self.bucket_size * 2)
        tv = sort_to_buckets(v, perm_oneback, self.bucket_size * 2)

        # Dot-product attention.
        inner = torch.einsum('zbhie,zbhje->zbhij', tq, tk) * softmax_temp
        masked_value = max_neg_value(inner)

        bq_idx = rearrange(perm, 'h b (nbuckets bucketsz) -> h b nbuckets bucketsz 1',
                           bucketsz=self.bucket_size)
        bkv_idx = rearrange(perm_oneback, 'h b (nbuckets bucketsz2) -> h b nbuckets 1 bucketsz2',
                            bucketsz2=self.bucket_size * 2)

        # Mask for post qk attention logits of the input sequence
        if attn_mask_bool is not None:
            dot_attn_indices = bq_idx * seqlen + bkv_idx
            mask = attn_mask_bool.flatten()[dot_attn_indices]
            inner.masked_fill_(~mask, masked_value)
            del mask

        # mask out attention to padded tokens
        if key_padding_mask is not None and not key_padding_mask.all_ones:
            s_key_padding_mask = sort_to_buckets(rearrange(key_padding_mask_bool,
                                                           'b s -> b s 1'),
                                                 perm_oneback, self.bucket_size * 2)
            s_key_padding_mask = rearrange(s_key_padding_mask,
                                           '... bucketsz 1 -> ... 1 bucketsz')
            inner.masked_fill_(~s_key_padding_mask, masked_value)

        # Causal masking
        if self.causal:
            mask = bq_idx < bkv_idx
            inner.masked_fill_(mask, masked_value)
            del mask

        # Mask out attention to self except when no other targets are available.
        self_mask = bq_idx == bkv_idx
        inner.masked_fill_(self_mask, TOKEN_SELF_ATTN_VALUE)
        del self_mask

        # Mask out attention to other hash buckets.
        if not self._attend_across_buckets:
            bq_buckets = sort_to_buckets(rearrange(buckets, 'h b s -> h b s 1'), perm,
                                         self.bucket_size, unsqueeze=False)
            bkv_buckets = sort_to_buckets(rearrange(buckets, 'h b s -> h b s 1'), perm_oneback,
                                          self.bucket_size * 2, unsqueeze=False)
            bkv_buckets = rearrange(bkv_buckets, 'h b nbuckets bucketsz2 1 -> h b nbuckets 1 bucketsz2')
            bucket_mask = bq_buckets != bkv_buckets
            inner.masked_fill_(bucket_mask, masked_value)
            del bucket_mask

        # Don't double-count query-key pairs across multiple rounds of hashing.
        # There are two possible strategies here. (1) The default is to count how
        # many times a query-key pair is repeated, and to lower its log-prob
        # correspondingly at each repetition. (2) When hard_k is set, the code
        # instead masks all but the first occurence of each query-key pair.
        if not self._allow_duplicate_attention:
            locs1 = rearrange(perm_inv // self.bucket_size, 'h b seqlen -> b seqlen h')
            locs2 = (locs1 + 1) % n_buckets
            if not self._attend_across_buckets:
                locs1 = buckets * n_buckets + locs1
                locs2 = buckets * n_buckets + locs2
            locs = torch.cat([locs1, locs2], dim=-1)

            slocs = sort_to_buckets(locs, perm, self.bucket_size)  # (h b nbuckets bucketsz h*2)
            bq_locs = repeat(slocs[..., :total_hashes],
                             'h b nbuckets bucketsz nh -> h b nbuckets bucketsz 1 (2 nh)')
            bkv_locs = look_one_back(rearrange(slocs, 'h b nbuckets bucketsz nh2'
                                                      '-> (h b) nbuckets bucketsz nh2'))
            bkv_locs = rearrange(bkv_locs,
                                 '(h b) nbuckets bucketsz2 nh2 -> h b nbuckets 1 bucketsz2 nh2',
                                 h=self.n_hashes)
            dup_counts = bq_locs == bkv_locs
            # for memory considerations, chunk summation of last dimension for counting duplicates
            dup_counts = chunked_sum(dup_counts, chunks=(total_hashes * batch_size))
            dup_counts = dup_counts.detach()
            assert dup_counts.shape == inner.shape
            inner = inner - torch.log(dup_counts + 1e-9)
            del dup_counts

        # Softmax.
        dots_logsumexp = torch.logsumexp(inner, dim=-1, keepdim=True)
        dots = torch.exp(inner - dots_logsumexp).type_as(inner)
        dropped_dots = self.dropout(dots)

        so = torch.einsum('...ij,...jd->...id', dropped_dots, tv)

        # undo sort
        def unsort_from_buckets(s_x, perm_inverse):
            b_x = rearrange(s_x, 'h b nbuckets bucketsz d -> h b (nbuckets bucketsz) d')
            return batched_index_select(b_x, perm_inverse)

        o = unsort_from_buckets(so, perm_inv)
        logits = unsort_from_buckets(dots_logsumexp, perm_inv)

        probs = torch.exp(logits - torch.logsumexp(logits, dim=0, keepdim=True))
        out = torch.sum(o * probs, dim=0)
        out = rearrange(out, '(b h) t d -> b t h d', h=H)
        out = out[:, :seqlen_og]

        attn = None
        if need_weights:
            dot_attn_indices = rearrange(bq_idx * seqlen + bkv_idx,
                                         'h b nbuckets qbucketsz kbucketsz -> h b (nbuckets qbucketsz kbucketsz)')
            unsorted_dots = torch.zeros(self.n_hashes, batch_size, seqlen * seqlen, device=device)
            unsorted_dots.scatter_(-1, dot_attn_indices, dots.view_as(dot_attn_indices))
            del dot_attn_indices
            unsorted_dots = rearrange(unsorted_dots,
                                      'h b (q_seqlen k_seqlen) -> h b q_seqlen k_seqlen',
                                      q_seqlen = seqlen)
            attn = torch.sum(unsorted_dots * probs, dim=0)
            attn = rearrange(attn, '(b h) t s -> b h t s', h=n_head)[:, :, :seqlen_og, :seqlen_og]

        return out, attn
