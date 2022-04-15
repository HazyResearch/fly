import math
import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from src.models.layers.blocksparse_linear import BlockSparseLinear, FlatBlockButterflySparsityConfig


class TestFlatBlockButterflySparsityConfig:

    @pytest.mark.parametrize('butterfly_size,n_factors,block_size',
                             [(32, 3, 16), (16, 2, 32), (16, 3, 32), (16, 4, 32), (8, 2, 32), (8, 3, 32)])
    def test_parameters_512(self, butterfly_size, n_factors, block_size):
        in_features, out_features = 512, 2048
        self = FlatBlockButterflySparsityConfig(butterfly_size, n_factors, block=block_size,
                                                global_size=0)
        mask = self.make_layout(out_features, in_features)
        print(f'Saving: {mask.float().mean().item()}')
        batch_size = 3
        x = torch.randn(batch_size, in_features)
        s_cfg = {'_target_': 'src.models.layers.blocksparse_linear.FlatBlockButterflySparsityConfig',
                 'butterfly_size': butterfly_size,
                 'n_factors': n_factors,
                 'block': block_size}
        self = BlockSparseLinear(in_features, out_features, s_cfg, backend='dense')
        out = self(x)
        assert out.shape == (batch_size, out_features)

    @pytest.mark.parametrize('butterfly_size,n_factors,block_size',
                             [(32, 3, 8), (16, 2, 16), (16, 3, 16), (16, 4, 16), (8, 2, 32), (8, 3, 32)])
    def test_parameters_768(self, butterfly_size, n_factors, block_size):
        in_features, out_features = 768, 3072
        self = FlatBlockButterflySparsityConfig(butterfly_size, n_factors, block=block_size,
                                                global_size=0)
        mask = self.make_layout(out_features, in_features)
        print(f'Saving: {mask.float().mean().item()}')
        batch_size = 3
        x = torch.randn(batch_size, in_features)
        s_cfg = {'_target_': 'src.models.layers.blocksparse_linear.FlatBlockButterflySparsityConfig',
                 'butterfly_size': butterfly_size,
                 'n_factors': n_factors,
                 'block': block_size}
        self = BlockSparseLinear(in_features, out_features, s_cfg, backend='dense')
        out = self(x)
        assert out.shape == (batch_size, out_features)
