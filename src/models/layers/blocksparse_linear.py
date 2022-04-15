import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from einops import rearrange, repeat

import hydra

from src.utils.utils import get_logger
logger = get_logger()

from src.utils.padding import pad_to_multiple
try:
    from src.models.attention.blocksparse_matmul import matmul
except ImportError:
    logger.info('triton is not installed')
    matmul = None
from src.ops.butterfly_factor import butterfly_factor_to_matrix
from src.models.attention.blocksparse_utils import sparsify_tensor, densify_tensor


try:
    from pytorch_block_sparse import BlockSparseMatrix
    from pytorch_block_sparse.block_sparse_linear import BlockSparseLinearFunction
except ImportError:
    logger.info('pytorch_block_sparse is not installed')
    BlockSparseMatrix = None
    BlockSparseLinearFunction = None


class BlockSparseLinear(nn.Module):
    """
    Arguments
    ---------
        sparsity_config: optional: this parameter determins sparsity pattern configuration; it is based on SparsityConfig class.
    """
    def __init__(self, in_features, out_features, sparsity_config, bias=True,
                 backend='triton', weight_decay=True):
        """
        weight_decay: whether to mark the sparse weight as _no_weight_decay.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity_config = hydra.utils.instantiate(sparsity_config)
        self.block_size = self.sparsity_config.block
        self.in_features_extended = int(math.ceil(in_features / self.block_size)) * self.block_size
        self.out_features_extended = int(math.ceil(out_features / self.block_size)) * self.block_size

        # initialize sparse layout and register as buffer
        layout = self.sparsity_config.make_layout(self.out_features_extended,
                                                  self.in_features_extended)
        self.register_buffer("layout", layout)
        self.nnz_blocks = self.layout.sum().item()

        if backend is None:
            backend = 'huggingface' if self.block_size == 32 else 'triton'
        if backend not in ['huggingface', 'triton', 'dense']:
            raise NotImplementedError(f'backend {backend} not supported')
        if backend == 'huggingface':
            if self.block_size != 32:
                raise NotImplementedError(f'backend huggingface requires block size to be 32')
            if BlockSparseLinearFunction is None or BlockSparseMatrix is None:
                raise ImportError(f'backend huggingface but package pytorch_block_sparse cannot be imported')
        self.backend = backend

        self.weight = nn.Parameter(torch.empty(self.nnz_blocks, self.block_size, self.block_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        if self.backend == 'huggingface':
            self.weight = nn.Parameter(
                rearrange(self.weight, 'nnz blksz blksz1 -> (nnz blksz1) blksz').contiguous()
            )
        elif self.backend == 'triton':
            self.weight = nn.Parameter(
                rearrange(self.weight, 'nnz blksz blksz1 -> 1 nnz blksz blksz1')
            )
        if not weight_decay:
            self.weight._no_weight_decay = True
        self.ops_cache = dict()
        logger.info(f'Linear class {self.__class__}: saving={self.saving}')

    def reset_parameters(self) -> None:
        self.set_weights_from_dense_init(dense_init_fn_=partial(init.kaiming_uniform_, a=math.sqrt(5)))
        fan_in, fan_out = self.in_features_extended, self.out_features_extended
        if self.bias is not None:
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def set_weights_from_dense_init(self, dense_init_fn_):
        dense_weight = torch.empty(self.out_features_extended, self.in_features_extended,
                                   device=self.weight.device, dtype=self.weight.dtype)
        dense_init_fn_(dense_weight)
        # We scale depending on how many nonzero cols there are in each row.
        ncol = self.layout.shape[-1]
        n_nonzero_cols = self.layout.sum(dim=-1, keepdim=True)
        scaling = torch.sqrt(ncol / n_nonzero_cols)
        dense_weight *= repeat(scaling, 'm 1 -> (b m) 1', b=self.block_size)
        with torch.no_grad():
            self.weight.copy_(sparsify_tensor(rearrange(dense_weight, 'o i -> 1 o i'),
                                              rearrange(self.layout, 'o_blk i_blk -> 1 o_blk i_blk')))

    @property
    def saving(self):
        return self.nnz_blocks * self.block_size ** 2 / (self.in_features * self.out_features)

    # add to cache
    def get_ops(self):
        if self.backend not in self.ops_cache:
            if self.backend == 'triton':
                matmul_dds_op = matmul(self.layout.cpu(), self.block_size, 'dds',
                                    trans_a=False, trans_b=True)
                self.ops_cache[self.backend] = matmul_dds_op
            elif self.backend == 'huggingface':
                weight_bsm = BlockSparseMatrix(
                    (self.out_features_extended, self.in_features_extended),
                    self.layout.bool().to('cuda'),
                    data=self.weight,
                    block_shape=(self.block_size, self.block_size)
                )
                self.ops_cache[self.backend] = weight_bsm
            elif self.backend == 'dense':
                self.ops_cache[self.backend] = None
        return self.ops_cache[self.backend]

    def forward(self, x):
        """
        Arguments
        ---------
            x: (..., in_features)
        Return:
            out: (..., out_features)
        """
        if not x.is_cuda and self.backend != 'dense':
            raise NotImplementedError('Backend triton and huggingface only support CUDA tensors')

        in_features = x.shape[-1]
        if in_features < self.in_features_extended:
            x = F.pad(x, (0, self.in_features_extended - in_features))

        if self.backend == 'huggingface':
            weight_bsm = self.get_ops()
            output = BlockSparseLinearFunction.apply(x, self.weight, weight_bsm)
        elif self.backend == 'triton':
            matmul_dds_op = self.get_ops()
            batch_shape = x.shape[:-1]
            x = x.reshape(-1, x.shape[-1])
            batch_dim = x.shape[0]
            x = pad_to_multiple(x, multiple=self.block_size, dims=0)
            output = rearrange(matmul_dds_op(rearrange(x, 'b d -> 1 1 b d'), self.weight),
                               '1 1 b d -> b d')
            if output.shape[0] > batch_dim:
                output = output[:batch_dim, :]
            output = output.reshape(batch_shape + (output.shape[-1],))
        elif self.backend == 'dense':
            weight = rearrange(densify_tensor(self.weight, rearrange(self.layout, 'p r -> 1 p r')),
                               '1 m n -> m n')
            output = F.linear(x, weight)

        out_features_extended = output.shape[-1]
        if out_features_extended > self.out_features:
            output = output[..., :self.out_features]
        # Convert bias to output.dtype in case of AMP, otherwise bias and activation will be in FP32
        return (output + self.bias.to(dtype=output.dtype)) if self.bias is not None else output


class FlatBlockButterflySparsityConfig:

    def __init__(self, butterfly_size, n_factors, block=32, global_size=0, shuffle=False):
        """shuffle: apply channel_shuffle operation before the matmul as in ShuffleNet
        """
        self.block = block
        log_n = int(math.log2(butterfly_size))
        if butterfly_size != 2 ** log_n or butterfly_size < 2:
            raise NotImplementedError('butterfly_size must be a power of 2')
        if not (1 <= n_factors <= log_n):
            raise NotImplementedError('n_factors must be a between 1 and log_2(butterfly_size)')
        self.butterfly_size = butterfly_size
        self.n_factors = n_factors
        self.global_size = global_size
        self.shuffle = shuffle

    def make_layout(self, out_features, in_features):
        assert out_features % self.block == 0 and in_features % self.block == 0
        twiddle = torch.ones(self.butterfly_size // 2, 2, 2)
        layout = sum(butterfly_factor_to_matrix(twiddle, index) for index in range(self.n_factors))
        layout = layout.bool().int()
        if self.shuffle:
            log_n = int(math.log2(self.butterfly_size))
            ngroups = 2 ** (log_n - self.n_factors)
            layout = rearrange(layout, 'm (group c_per_group) -> m (c_per_group group)',
                               group=ngroups)
        # Convert from (butterfly_size, butterfly_size) mask to (out_features, in_features) mask
        layout = repeat(layout, 'b b1 -> (b f) (b1 f1)',
                        f=out_features // self.butterfly_size, f1=in_features // self.butterfly_size)
        if self.global_size > 0:
            layout[:self.global_size] = 1
            layout[:, :self.global_size] = 1
        # Convert from (out_features, in_features) mask to
        # (out_features // block, in_features // block) mask
        layout = rearrange(layout, '(p blksz) (r blksz1) -> p r (blksz blksz1)',
                           blksz=self.block, blksz1=self.block)
        return (layout > 0).any(dim=-1).int()
