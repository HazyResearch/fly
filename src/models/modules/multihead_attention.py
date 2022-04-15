import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn.init import constant_
from torch.nn.parameter import Parameter
from torch import Tensor

from typing import Tuple, List, Optional

from einops import rearrange

from src.models.modules.masking import FullMask, LengthMask

from src.models.attention.mask_utils import pad_mask


# Adapted from https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#MultiheadAttention
# https://github.com/pytorch/pytorch/blob/release/1.9/torch/nn/modules/activation.py
class MultiheadAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """

    """TD: We have a different interpretation of kdim and vdim compared to Pytorch.
    To be fair the Pytorch's interpretation is very confusing and the docs is unclear as well.
    https://github.com/pytorch/pytorch/issues/60831
    https://github.com/pytorch/pytorch/pull/61977/files

    Here we use the interpretation from the original "Attention is all you need" paper.
    query, key, value all have last dimension embed_dim.
    They are then projected to dimension kdim, kdim, vdim.
    """
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, shared_qk=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_dim = self.kdim == self.vdim

        self.num_heads = num_heads
        self.batch_first = batch_first
        assert self.kdim % num_heads == 0, "self.kdim must be divisible by num_heads"
        assert self.vdim % num_heads == 0, "self.vdim must be divisible by num_heads"

        self.shared_qk = shared_qk
        if self._qkv_same_dim is False:
            self.q_proj_weight = Parameter(torch.empty((self.kdim, embed_dim), **factory_kwargs))
            if not shared_qk:
                self.k_proj_weight = Parameter(torch.empty((self.kdim, embed_dim), **factory_kwargs))
            else:
                self.register_parameter('k_proj_weight', None)
            self.v_proj_weight = Parameter(torch.empty((self.vdim, embed_dim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty(((3 if not shared_qk else 2) * self.kdim,
                                                         embed_dim), **factory_kwargs))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty((2 if not shared_qk else 1) * self.kdim
                                                      + self.vdim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.in_proj_container = InProjContainer(self.q_proj_weight, self.k_proj_weight,
                                                 self.v_proj_weight, self.in_proj_weight,
                                                 self.in_proj_bias, shared_qk=self.shared_qk)

        self.out_proj = nn.Linear(self.vdim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, self.kdim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, self.vdim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            if self.k_proj_weight is not None:
                xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def forward(self, attention_layer: nn.Module, query: Tensor, key: Tensor, value: Tensor,
                key_padding_mask: Optional[Tensor] = None, need_weights: bool = False,
                attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
    Shapes for inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension. :math:`(N, S, E)` if ``batch_first`` is ``True``.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension. :math:`(N, S, E)` if ``batch_first`` is ``True``.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: if a 2D mask: :math:`(L, S)` where L is the target sequence length, S is the
          source sequence length.
          If a 3D mask: :math:`(N\cdot\text{num\_heads}, L, S)` where N is the batch size, L is the target sequence
          length, S is the source sequence length. ``attn_mask`` ensure that position i is allowed to attend
          the unmasked positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
    Shapes for outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        use_separate_proj_weight = not self._qkv_same_dim
        num_heads = self.num_heads
        # set up shape vars
        if self.batch_first:
            bsz, tgt_len, embed_dim = query.shape
            _, src_len, _ = key.shape
        else:
            tgt_len, bsz, embed_dim = query.shape
            src_len, _, _ = key.shape
        assert embed_dim == self.embed_dim, \
            f"was expecting embedding dimension of {self.embed_dim}, but got {embed_dim}"
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

        #
        # compute in-projection
        #
        q, k, v = self.in_proj_container(query, key, value)

        # TD: We want to do this transposition after the in_proj_container, because that projection
        # checks if q is k and k is v, and if so it can group some matmuls together for speed.
        if self.batch_first:
            q, k, v = [rearrange(x, 'b s ... -> s b ...') for x in (q, k, v)]

        # prep attention mask
        if attn_mask is not None:
            assert isinstance(attn_mask, (FullMask, LengthMask))
            if isinstance(attn_mask, FullMask):
                correct_shape = (tgt_len, src_len)
                if attn_mask.bool_matrix.shape != correct_shape:
                    raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.bool_matrix.shape}, but should be {correct_shape}.")
            elif isinstance(attn_mask, LengthMask):
                correct_shape = (tgt_len,)
                if attn_mask._lengths.shape != (tgt_len,):
                    raise RuntimeError(f"The length of the 2D attn_mask is {attn_mask._lengths.shape}, but should be {correct_shape}.")

        # add bias along batch dimension (currently second)
        if self.bias_k is not None and self.bias_v is not None:
            # Pytorch's implementation puts k first and the bias after.
            # We put the bias first because our key_padding_mask needs to be consecutive.
            # We don't want True True False ... False True
            k = torch.cat([self.bias_k.repeat(1, bsz, 1), k])
            v = torch.cat([self.bias_v.repeat(1, bsz, 1), v])
            if attn_mask is not None:
                attn_mask = pad_mask(attn_mask, 1, left=True)
            if key_padding_mask is not None:
                key_padding_mask = pad_mask(key_padding_mask, 1, left=True)
        else:
            assert self.bias_k is None
            assert self.bias_v is None

        # add zero attention along batch dimension
        if self.add_zero_attn:
            zero_attn_shape_k = (1, bsz, self.kdim)
            zero_attn_shape_v = (1, bsz, self.vdim)
            # Pytorch's implementation puts k first and the zeros after.
            # We put the zeros first because our key_padding_mask needs to be consecutive.
            # We don't want True True False ... False True
            k = torch.cat([torch.zeros(zero_attn_shape_k, dtype=k.dtype, device=k.device), k],
                          dim=0)
            v = torch.cat([torch.zeros(zero_attn_shape_v, dtype=v.dtype, device=v.device), v],
                          dim=0)
            if attn_mask is not None:
                attn_mask = pad_mask(attn_mask, 1, left=True)
            if key_padding_mask is not None:
                key_padding_mask = pad_mask(key_padding_mask, 1, left=True)

        #
        # reshape q, k, v for multihead attention and make em batch first
        #
        q, k, v = [rearrange(x, 't b (n_head head_dim) -> b t n_head head_dim',
                             n_head=self.num_heads) for x in (q, k, v)]

        #
        # (deep breath) calculate attention and out projection
        #
        attn_output, attn_output_weights = attention_layer(q, k, v,
                                                           attn_mask=attn_mask,
                                                           key_padding_mask=key_padding_mask,
                                                           need_weights=need_weights)
        attn_output = rearrange(attn_output, 'b t h d -> b t (h d)')
        attn_output = self.out_proj(attn_output)

        if not self.batch_first:
            attn_output = rearrange(attn_output, 'b t e -> t b e')

        return attn_output, attn_output_weights if need_weights else None


class InProjContainer(torch.nn.Module):

    def __init__(self, q_proj_weight, k_proj_weight, v_proj_weight, in_proj_weight, in_proj_bias,
                 shared_qk=False):
        r"""A in-proj container to project query/key/value in MultiheadAttention. This module happens before reshaping
        the projected query/key/value into multiple heads. See the linear layers (bottom) of Multi-head Attention in
        Fig 2 of Attention Is All You Need paper. Also check the usage example
        in torchtext.nn.MultiheadAttentionContainer.
        Args:
            q_proj_weight: a proj layer for query. A typical projection layer is torch.nn.Linear.
            k_proj_weight: a proj layer for key. A typical projection layer is torch.nn.Linear.
            v_proj_weight: a proj layer for value. A typical projection layer is torch.nn.Linear.
        """

        super().__init__()
        self.q_proj_weight = q_proj_weight
        self.k_proj_weight = k_proj_weight
        self.v_proj_weight = v_proj_weight
        self.in_proj_weight = in_proj_weight
        self.in_proj_bias = in_proj_bias
        self.packed_weight = in_proj_weight is not None
        self.shared_qk = shared_qk

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Projects the input sequences using in-proj layers. q/k/v are simply passed to
        the forward func of q/k/value_proj, respectively.
        Args:
            q (Tensor): The q to be projected.
            k (Tensor): The keys to be projected.
            v (Tensor): The values to be projected.
        Examples::
            >>> import torch
            >>> from torchtext.nn import InProjContainer
            >>> embed_dim, bsz = 10, 64
            >>> in_proj_container = InProjContainer(torch.nn.Linear(embed_dim, embed_dim),
                                                    torch.nn.Linear(embed_dim, embed_dim),
                                                    torch.nn.Linear(embed_dim, embed_dim))
            >>> q = torch.rand((5, bsz, embed_dim))
            >>> k = v = torch.rand((6, bsz, embed_dim))
            >>> q, k, v = in_proj_container(q, k, v)
        """
        if self.packed_weight:
            if not self.shared_qk:
                return _in_projection_packed(q, k, v, self.in_proj_weight, self.in_proj_bias)
            else:
                E = self.in_proj_weight.shape[0] // 2
                w, b = self.in_proj_weight, self.in_proj_bias
                if k is v:
                    if q is k:
                        # self-attention
                        qk_projected, v_projected = F.linear(q, w, b).chunk(2, dim=-1)
                        return qk_projected, qk_projected, v_projected
                    else:
                        # encoder-decoder attention
                        w_q, _ = w.chunk(2)
                        w_kv = w
                        if b is None:
                            b_q = b_kv = None
                        else:
                            b_q, _ = b.chunk(2)
                            b_kv = b
                        return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).chunk(2, dim=-1)
                else:
                    w_qk, w_v = w.chunk(2)
                    if b is None:
                        b_qk = b_v = None
                    else:
                        b_qk, b_v = b.chunk(2)
                    return F.linear(q, w_qk, b_qk), F.linear(k, w_qk, b_qk), F.linear(v, w_v, b_v)

        else:
            w_q = self.q_proj_weight
            w_k = self.k_proj_weight if not self.shared_qk else self.q_proj_weight
            w_v = self.v_proj_weight
            assert w_q is not None, "use_separate_proj_weight is False but q_proj_weight is None"
            assert w_k is not None, "use_separate_proj_weight is False but k_proj_weight is None"
            assert w_v is not None, "use_separate_proj_weight is False but v_proj_weight is None"
            if self.in_proj_bias is None:
                b_q = b_k = b_v = None
            else:
                kdim, vdim = self.q_proj_weight.shape[0], self.v_proj_weight.shape[0]
                if not self.shared_qk:
                    b_q, b_k, b_v = self.in_proj_bias.split([kdim, kdim, vdim])
                else:
                    b_q, b_v = self.in_proj_bias.split([kdim, vdim])
                    b_k = b_q
            return _in_projection(q, k, v, w_q, w_k, w_v, b_q, b_k, b_v)


# Copied from https://github.com/pytorch/pytorch/blob/release/1.9/torch/nn/functional.py#L4836
def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> List[Tensor]:
    r"""
    Performs the in-projection step of the attention operation, using packed weights.
    Output is a triple containing projection tensors for query, key and value.
    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.
    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension
        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = w.shape[0] // 3
    if k is v:
        if q is k:
            # self-attention
            return F.linear(q, w, b).chunk(3, dim=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)


def _in_projection(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    b_q: Optional[Tensor] = None,
    b_k: Optional[Tensor] = None,
    b_v: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""
    Performs the in-projection step of the attention operation. This is simply
    a triple of linear projections, with shape constraints on the weights which
    ensure embedding dimension uniformity in the projected outputs.
    Output is a triple containing projection tensors for query, key and value.
    Args:
        q, k, v: query, key and value tensors to be projected.
        w_q, w_k, w_v: weights for q, k and v, respectively.
        b_q, b_k, b_v: optional biases for q, k and v, respectively.
    Shape:
        Inputs:
        - q: :math:`(Qdims..., Eq)` where Eq is the query embedding dimension and Qdims are any
            number of leading dimensions.
        - k: :math:`(Kdims..., Ek)` where Ek is the key embedding dimension and Kdims are any
            number of leading dimensions.
        - v: :math:`(Vdims..., Ev)` where Ev is the value embedding dimension and Vdims are any
            number of leading dimensions.
        - w_q: :math:`(Eq, Eq)`
        - w_k: :math:`(Eq, Ek)`
        - w_v: :math:`(Eq, Ev)`
        - b_q: :math:`(Eq)`
        - b_k: :math:`(Eq)`
        - b_v: :math:`(Eq)`
        Output: in output triple :math:`(q', k', v')`,
         - q': :math:`[Qdims..., Eq]`
         - k': :math:`[Kdims..., Eq]`
         - v': :math:`[Vdims..., Eq]`
    """
    Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
    assert Eq == Ek == Ev, 'query, key, and value must have the same dimension'
    qdim, kdim, vdim = w_q.shape[0], w_k.shape[0], w_v.shape[0]
    assert qdim == kdim, 'query and key must be projected to the same dimension'
    assert w_q.shape == (qdim, Eq), f"expecting query weights shape of {(qdim, Eq)}, but got {w_q.shape}"
    assert w_k.shape == (kdim, Ek), f"expecting key weights shape of {(kdim, Ek)}, but got {w_k.shape}"
    assert w_v.shape == (vdim, Ev), f"expecting value weights shape of {(vdim, Ev)}, but got {w_v.shape}"
    assert b_q is None or b_q.shape == (qdim,), f"expecting query bias shape of {(qdim,)}, but got {b_q.shape}"
    assert b_k is None or b_k.shape == (kdim,), f"expecting key bias shape of {(kdim,)}, but got {b_k.shape}"
    assert b_v is None or b_v.shape == (vdim,), f"expecting value bias shape of {(vdim,)}, but got {b_v.shape}"
    return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)


def _scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    r"""
    Computes scaled dot product attention on query, key and value tensors, using
    an optional attention mask if passed, and applying dropout if a probability
    greater than 0.0 is specified.
    Returns a tensor pair containing attended values and attention weights.
    Args:
        q, k, v: query, key and value tensors. See Shape section for shape details.
        attn_mask: optional tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dropout_p: dropout probability. If greater than 0.0, dropout is applied.
    Shape:
        - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
            and E is embedding dimension.
        - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
            shape :math:`(Nt, Ns)`.
        - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
            have shape :math:`(B, Nt, Ns)`
    """
    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = torch.bmm(q, k.transpose(-2, -1))
    if attn_mask is not None:
        attn += attn_mask
    attn = torch.softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn
