import copy
from typing import Optional, Union, Callable

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

from einops import rearrange

import hydra

from src.models.modules.masking import LengthMask
from src.models.modules.seq_common import ClassificationHead, PositionalEncoding, Mlp
from src.models.modules.s4 import S4


# Adapted from https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#S4EncoderLayer
class S4EncoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, d_inner=2048, ffn_cfg=None, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.norm_first = norm_first
        self.s4 = S4(H=d_model, l_max=1024, transposed=True, dropout=dropout)
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
        if not isinstance(self.ff, nn.Identity):
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout2d(dropout)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(S4EncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, **kwargs) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in S4Sequence class.
        """
        x = src
        if self.norm_first:
            out, _ = self.s4(rearrange(self.norm1(x)), '... L d -> ... d L')
            out = self.dropout1(out)
            x = x + rearrange(out, '... d L -> ... L d')
            if not isinstance(self.ff, nn.Identity):
                x = x + self.ff(self.norm2(x))
        else:
            out, _ = self.s4(rearrange(x, '... L d -> ... d L'))
            out = self.dropout1(out)
            x = self.norm1(x + rearrange(out, '... d L -> ... L d'))
            if not isinstance(self.ff, nn.Identity):
                x = self.norm2(x + self.ff(x))
        return x


class S4Encoder(nn.Module):
    r"""S4Encoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the S4EncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.S4EncoderLayer(d_model=512)
        >>> transformer_encoder = nn.S4Encoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        # self.layers = _get_clones(encoder_layer, num_layers)
        self.layers = nn.ModuleList([encoder_layer() for i in range(num_layers)])

        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, **kwargs) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in S4Sequence class.
        """
        output = src
        for mod in self.layers:
            output = mod(output, **kwargs)
        if self.norm is not None:
            output = self.norm(output)
        return output


class S4Sequence(nn.Module):
    r"""
    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        n_layer: the number of sub-encoder-layers in the encoder (default=6).
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
        >>> transformer_model = nn.S4Sequence(n_layer=12)
        >>> src = torch.rand((10, 32, 512))
        >>> tgt = torch.rand((20, 32, 512))
        >>> out = transformer_model(src, tgt)

    Note: A full example to apply nn.S4Sequence module for the word language model is available in
    https://github.com/pytorch/examples/tree/master/word_language_model
    """

    def __init__(self, d_model: int = 512, n_layer: int = 6, d_inner: int = 2048,
                 ffn_cfg=None,
                 dropout: float = 0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.d_model = d_model
        self.batch_first = batch_first
        encoder_layer = lambda : S4EncoderLayer(d_model, d_inner=d_inner, ffn_cfg=ffn_cfg, dropout=dropout,
                                       activation=activation, layer_norm_eps=layer_norm_eps,
                                       batch_first=batch_first, norm_first=norm_first,
                                       **factory_kwargs)
        # encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        encoder_norm = None
        self.encoder = S4Encoder(encoder_layer, n_layer, encoder_norm)

    def forward(self, src: Tensor, **kwargs) -> Tensor:
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

        output = self.encoder(src)
        return output


class S4Classifier(nn.Module):

    def __init__(self, d_model: int, n_layer: int, d_inner: int, num_classes: int,
                 ffn_cfg=None, embedding_cfg=None, classifier_cfg=None,
                 norm_first=False, dropout: float = 0.1, activation: str = "relu",
                 layer_norm_eps: float = 1e-5,
                 batch_first: bool = False, pooling_mode='MEAN') -> None:
        super().__init__()
        assert pooling_mode in ['MEAN', 'SUM'], 'pooling_mode not supported'
        self.pooling_mode = pooling_mode
        self.embedding = (nn.Identity() if embedding_cfg is None
                          else hydra.utils.instantiate(embedding_cfg, _recursive_=False))
        self.batch_first = batch_first
        self.s4seq = S4Sequence(d_model, n_layer, d_inner, ffn_cfg, dropout, activation,
                                layer_norm_eps, batch_first, norm_first)
        if classifier_cfg is None:
            self.classifier = ClassificationHead(d_model, d_inner, num_classes,
                                                 pooling_mode=pooling_mode, batch_first=batch_first)
        else:
            self.classifier = hydra.utils.instantiate(
                classifier_cfg, d_model=d_model, d_inner=d_inner, num_classes=num_classes,
                pooling_mode=pooling_mode, batch_first=batch_first, _recursive_=False
            )

    def forward_features(self, src: Tensor, lengths=None, **kwargs) -> Tensor:
        if lengths is not None:
            src_key_padding_mask = LengthMask(lengths,
                                              max_len=src.size(1 if self.batch_first else 0),
                                              device=src.device)
        else:
            src_key_padding_mask = None
        src = self.embedding(src)
        features = self.s4seq(src, **kwargs)
        return features, src_key_padding_mask

    def forward(self, src: Tensor, lengths=None, **kwargs) -> Tensor:
        features, src_key_padding_mask = self.forward_features(src, lengths=lengths, **kwargs)
        return self.classifier(features, key_padding_mask=src_key_padding_mask)


class S4DualClassifier(S4Classifier):

    def forward(self, src1: Tensor, src2: Tensor,
                lengths1=None, lengths2=None,
                **kwargs) -> Tensor:
        features1, src1_key_padding_mask = self.forward_features(src1, lengths=lengths1, **kwargs)
        features2, src2_key_padding_mask = self.forward_features(src2, lengths=lengths2, **kwargs)
        return self.classifier(features1, features2,
                               key_padding_mask1=src1_key_padding_mask,
                               key_padding_mask2=src2_key_padding_mask)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
