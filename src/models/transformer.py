import functools
import math
import copy
from typing import Optional, Union, Callable

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

from src.models.modules.masking import FullMask, LengthMask

from einops import repeat

import hydra

from src.models.modules.seq_common import ClassificationHead, PositionalEncoding, Mlp
from src.models.modules.multihead_attention import MultiheadAttention


# Adapted from https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer
class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        n_head: the number of heads in the multiheadattention models (required).
        d_inner: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, n_head=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, n_head=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, n_head, d_inner=2048, mha_cfg=None, attn_cfg=None, ffn_cfg=None,
                 dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.norm_first = norm_first
        if mha_cfg is None:
            mha_cfg = {}
        if attn_cfg is None:
            self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout,
                                                   batch_first=batch_first, **mha_cfg,
                                                   **factory_kwargs)
            self.attention_layer = None
        else:
            self.self_attn = MultiheadAttention(d_model, n_head, batch_first=batch_first,
                                                **mha_cfg, **factory_kwargs)
            self.attention_layer = hydra.utils.instantiate(attn_cfg, **factory_kwargs,
                                                           _recursive_=False)
        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation
        # Implementation of Feedforward model
        if ffn_cfg is None:
            self.ff = Mlp(d_model, hidden_features=d_inner,
                          act_fn=self.activation, drop=dropout, **factory_kwargs)
        else:
            self.ff = hydra.utils.instantiate(ffn_cfg, **factory_kwargs, _recursive_=False)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask=None,
                src_key_padding_mask: Optional[Tensor] = None, **kwargs) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), attn_mask=src_mask,
                                   key_padding_mask=src_key_padding_mask, **kwargs)
            x = x + self.ff(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, attn_mask=src_mask,
                                              key_padding_mask=src_key_padding_mask, **kwargs))
            x = self.norm2(x + self.ff(x))
        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor],
                  **kwargs) -> Tensor:
        if self.attention_layer is None:
            x, _ = self.self_attn(x, x, x, attn_mask=attn_mask,
                                  key_padding_mask=key_padding_mask,
                                  need_weights=False, **kwargs)
        else:
            x, _ = self.self_attn(self.attention_layer, x, x, x, attn_mask=attn_mask,
                                  key_padding_mask=key_padding_mask,
                                  need_weights=False, **kwargs)
        return self.dropout1(x)


# Adapted from https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoder
class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, n_head=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None, **kwargs) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, **kwargs)
        if self.norm is not None:
            output = self.norm(output)
        return output


# Adapted from https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer
class Transformer(nn.Module):
    r"""A transformer model. User is able to modify the attributes as needed. The architecture
    is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
    Processing Systems, pages 6000-6010. Users can build the BERT(https://arxiv.org/abs/1810.04805)
    model with corresponding parameters.

    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        n_head: the number of heads in the multiheadattention models (default=8).
        n_layer: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        d_inner: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, encoder and decoder layers will perform LayerNorms before
            other attention and feedforward operations, otherwise after. Default: ``False`` (after).

    Examples::
        >>> transformer_model = nn.Transformer(n_head=16, n_layer=12)
        >>> src = torch.rand((10, 32, 512))
        >>> tgt = torch.rand((20, 32, 512))
        >>> out = transformer_model(src, tgt)

    Note: A full example to apply nn.Transformer module for the word language model is available in
    https://github.com/pytorch/examples/tree/master/word_language_model
    """

    def __init__(self, d_model: int = 512, n_head: int = 8, n_layer: int = 6, d_inner: int = 2048,
                 mha_cfg=None, attn_cfg=None, ffn_cfg=None,
                 dropout: float = 0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.batch_first = batch_first
        encoder_layer = TransformerEncoderLayer(d_model, n_head, d_inner=d_inner,
                                                mha_cfg=mha_cfg, attn_cfg=attn_cfg, ffn_cfg=ffn_cfg,
                                                dropout=dropout,
                                                activation=activation,
                                                layer_norm_eps=layer_norm_eps,
                                                batch_first=batch_first,
                                                norm_first=norm_first,
                                                **factory_kwargs)
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.encoder = TransformerEncoder(encoder_layer, n_layer, encoder_norm)
        self._reset_parameters()

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None, **kwargs) -> Tensor:
        r"""Take in and process masked source/target sequences.

        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            src_mask: the additive mask for the src sequence (optional).
            tgt_mask: the additive mask for the tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
            src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).

        Shape:
            - src: :math:`(S, N, E)`, `(N, S, E)` if batch_first.
            - tgt: :math:`(T, N, E)`, `(N, T, E)` if batch_first.
            - src_mask: :math:`(S, S)`.
            - tgt_mask: :math:`(T, T)`.
            - memory_mask: :math:`(T, S)`.
            - src_key_padding_mask: :math:`(N, S)`.
            - tgt_key_padding_mask: :math:`(N, T)`.
            - memory_key_padding_mask: :math:`(N, S)`.

            Note: [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.
            [src/tgt/memory]_key_padding_mask provides specified elements in the key to be ignored by
            the attention. If a ByteTensor is provided, the non-zero positions will be ignored while the zero
            positions will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.

            - output: :math:`(T, N, E)`, `(N, T, E)` if batch_first.

            Note: Due to the multi-head attention architecture in the transformer model,
            the output sequence length of a transformer is same as the input sequence
            (i.e. target) length of the decode.

            where S is the source sequence length, T is the target sequence length, N is the
            batch size, E is the feature number

        Examples:
            >>> output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        """

        output = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return output

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class TransformerClassifier(nn.Module):

    def __init__(self, d_model: int, n_head: int, n_layer: int, d_inner: int,
                 num_classes: int, mha_cfg=None, attn_cfg=None, ffn_cfg=None, embedding_cfg=None,
                 pos_encoding_cfg=None, norm_first=False,
                 dropout: float = 0.1, activation: str = "relu", layer_norm_eps: float = 1e-5,
                 batch_first: bool = False, pooling_mode='MEAN') -> None:
        super().__init__()
        assert pooling_mode in ['MEAN', 'SUM', 'CLS'], 'pooling_mode not supported'
        self.pooling_mode = pooling_mode
        if pooling_mode == 'CLS':
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            if pos_encoding_cfg is not None and 'max_len' in pos_encoding_cfg:
                pos_encoding_cfg.max_len += 1
        self.word_emb = (nn.Identity() if embedding_cfg is None
                         else hydra.utils.instantiate(embedding_cfg, _recursive_=False))
        if pos_encoding_cfg is None:
            self.pos_encoder = PositionalEncoding(d_model, dropout, batch_first=batch_first)
        else:
            self.pos_encoder = hydra.utils.instantiate(pos_encoding_cfg, d_model,
                                                       batch_first=batch_first, _recursive_=False)
        self.batch_first = batch_first
        self.transformer = Transformer(d_model, n_head, n_layer, d_inner, mha_cfg, attn_cfg,
                                       ffn_cfg, dropout, activation, layer_norm_eps,
                                       batch_first, norm_first)
        self.classifier = ClassificationHead(d_model, d_inner, num_classes,
                                             pooling_mode=pooling_mode, batch_first=batch_first)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None, lengths=None, **kwargs) -> Tensor:
        if lengths is not None:
            src_key_padding_mask = LengthMask(lengths,
                                              max_len=src.size(1 if self.batch_first else 0),
                                              device=src.device)
        src = self.word_emb(src)
        if self.pooling_mode == 'CLS':
            cls = repeat(self.cls, '1 1 d -> b 1 d' if self.batch_first else '1 1 d -> 1 b d',
                         b=src.shape[0 if self.batch_first else 1])
            src = torch.cat([cls, src], dim=1 if self.batch_first else 0)
            # Adjust masks
            if src_key_padding_mask is not None:
                assert isinstance(src_key_padding_mask, LengthMask)
                src_key_padding_mask = LengthMask(src_key_padding_mask._lengths + 1,
                                                  max_len=src_key_padding_mask._max_len + 1,
                                                  device=src_key_padding_mask._lengths.device)
            if src_mask is not None:
                assert isinstance(src_mask, FullMask)
                src_mask = FullMask(F.pad(src_mask._mask, (1, 0, 1, 0), value=True))
        src = self.pos_encoder(src)
        outputs = self.transformer(src, src_mask=src_mask,
                                   src_key_padding_mask=src_key_padding_mask, **kwargs)
        return self.classifier(outputs)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
