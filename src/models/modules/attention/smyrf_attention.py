# Adapted from https://github.com/giannisdaras/smyrf/blob/master/smyrf/torch/attn.py
import math
import torch
import torch.nn as nn

from einops import rearrange

from src.models.modules.attention.hash_utils import XBOXPLUS, lsh_clustering
from src.models.modules.attention.reformer_attention import pad_to_multiple


class SmyrfAttention(nn.Module):
    def __init__(self, n_hashes, q_cluster_size, k_cluster_size,
                 q_attn_size=None, k_attn_size=None,
                 clustering_algo='lsh',
                 attention_dropout=0.,
                 # LSH clustering
                 r=1,
                 # kmeans clustering
                 max_iters=50):
        super(SmyrfAttention, self).__init__()
        self.n_hashes = n_hashes

        if q_attn_size is None:
            self.q_attn_size = q_cluster_size
        else:
            self.q_attn_size = q_attn_size

        if k_attn_size is None:
            self.k_attn_size = k_cluster_size
        else:
            self.k_attn_size = k_attn_size

        self.dropout = nn.Dropout(attention_dropout)
        self.xbox_plus = XBOXPLUS()

        self.clustering_algo = clustering_algo
        if clustering_algo == 'lsh':
            self.clustering_params = {
                'r': r,
                'n_hashes': self.n_hashes
            }
        else:
            raise NotImplementedError('Uknown clustering algorithm')


    def forward(self, query, key, value, attn_mask=None, progress=False,
                norm_factor=1):
        key_padding_mask = attn_mask  # TODO: I think this is wrong
        _, q_seqlen_og, n_head, _ = query.shape
        _, k_seqlen_og, _, _ = key.shape
        if q_seqlen_og % (self.q_attn_size) != 0:
            # TODO: pad the masks
            query = pad_to_multiple(query, self.q_attn_size, dims=1)
        if k_seqlen_og % (self.k_attn_size) != 0:
            # TODO: pad the masks
            key = pad_to_multiple(key, self.k_attn_size, dims=1)
            value = pad_to_multiple(value, self.k_attn_size, dims=1)
        query = rearrange(query, 'b t h e -> (b h) t e')
        key = rearrange(key, 'b t h e -> (b h) t e')
        value = rearrange(value, 'b s h d -> (b h) s d')
        bs, q_seqlen, dim = query.shape
        bs, k_seqlen, dim = key.shape
        v_dim = value.shape[-1]
        assert query.device == key.device, 'Queries, key in different devices'
        device = query.device


        # prepare mask if not None
        if attn_mask is not None:
            # We expect first dimension to be batch_size and second dimension seq. length
            if len(attn_mask.shape) == 1:
                attn_mask = attn_mask.unsqueeze(0)
            # repeat for n_hashes, heads
            attn_mask = attn_mask.unsqueeze(0).repeat(self.n_hashes, query.shape[0] // attn_mask.shape[0], 1)

        with torch.no_grad():
            # XBOX+ transform
            self.xbox_plus.set_norms(query, key)
            Queries = self.xbox_plus.Q(query)
            Keys = self.xbox_plus.K(key)

            num_clusters = Queries.shape[1] // self.q_attn_size
            assert num_clusters == (Keys.shape[1] // self.k_attn_size), 'Unequal number of clusters for query and key.'


            if self.clustering_algo == 'lsh':
                q_positions, k_positions = lsh_clustering(Queries, Keys, **self.clustering_params,
                                                          key_padding_mask=key_padding_mask)
            else:
                raise NotImplementedError('This algorithm is not supported')

            q_positions = q_positions.reshape(self.n_hashes, bs, -1)
            k_positions = k_positions.reshape(self.n_hashes, bs, -1)

        # free memory
        del Queries
        del Keys


        q_rev_positions = torch.argsort(q_positions, dim=-1)
        q_offset = torch.arange(bs, device=query.device).unsqueeze(-1) * q_seqlen
        k_offset = torch.arange(bs, device=query.device).unsqueeze(-1) * k_seqlen


        q_flat = (q_positions + q_offset).reshape(-1)
        k_flat = (k_positions + k_offset).reshape(-1)

        # sorted query, key, value
        s_queries = query.reshape(-1, dim).index_select(0, q_flat).reshape(-1, self.q_attn_size, dim)
        s_keys = key.reshape(-1, dim).index_select(0, k_flat).reshape(-1, self.k_attn_size, dim)
        s_values = value.reshape(-1, v_dim).index_select(0, k_flat).reshape(-1, self.k_attn_size, v_dim)

        inner = s_queries @ s_keys.transpose(2, 1)
        inner = inner / norm_factor

        # mask out attention to padded tokens
        if attn_mask is not None:
            inner = (attn_mask.reshape(-1)[k_flat].reshape(-1, self.k_attn_size).unsqueeze(1) + inner)

        # free memory
        del q_positions, k_positions

        # softmax denominator
        dots_logsumexp = torch.logsumexp(inner, dim=-1, keepdim=True)
        # softmax
        dots = torch.exp(inner - dots_logsumexp)
        # dropout
        dots = self.dropout(dots)

        # n_hashes outs
        bo = (dots @ s_values).reshape(self.n_hashes, bs, q_seqlen, -1)

        # undo sort
        q_offset = torch.arange(bs * self.n_hashes, device=query.device).unsqueeze(-1) * q_seqlen
        q_rev_flat = (q_rev_positions.reshape(-1, q_seqlen) + q_offset).reshape(-1)
        o = bo.reshape(-1, v_dim).index_select(0, q_rev_flat).reshape(self.n_hashes, bs, q_seqlen, -1)

        slogits = dots_logsumexp.reshape(self.n_hashes, bs, -1)
        logits = torch.gather(slogits, 2, q_rev_positions)

        # free memory
        del q_rev_positions

        probs = torch.exp(logits - torch.logsumexp(logits, dim=0, keepdim=True))
        out = torch.sum(o * probs.unsqueeze(-1), dim=0)
        out = rearrange(out, '(b h) t d -> b t h d', h=n_head)
        if q_seqlen_og % (self.q_attn_size) != 0:
            out = out[:, :q_seqlen_og]

        return out, None
