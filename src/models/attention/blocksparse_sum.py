# Adapted from https://github.com/openai/triton/blob/master/python/triton/ops/blocksparse/softmax.py
import triton.language as tl
import triton
import torch

from src.models.attention.blocksparse_utils import sparsify_broadcast_tensor

def next_power_of_2(n):
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    return n


def num_warps(n):
    if n < 512:
        return 4
    if n < 2048:
        return 8
    return 16


@triton.heuristics({'num_warps': lambda *args, **meta: num_warps(args[3] * meta['BLOCK'])})
@triton.heuristics({'TN': lambda *args, **meta: next_power_of_2(args[3] * meta['BLOCK'])})
@triton.jit
def _forward(
    X, OUT, LUT, sizemax, stride_zx, stride_zout, stride_hout, **meta
):
    TN = meta['TN']
    BLOCK = meta['BLOCK']
    pidhm = tl.program_id(0)
    pidz = tl.program_id(1)
    # create index ranges
    rxm = pidhm % BLOCK
    rbm = pidhm // BLOCK
    rxn = tl.arange(0, TN) % BLOCK
    rbn = tl.arange(0, TN) // BLOCK
    # extract information from LUT
    header = LUT + rbm * 2
    size = tl.load(header + 0)
    offset = tl.load(header + 1)
    check = rbn < size
    rbmn = tl.where(check, rbn, size - 1)
    # block id and column id
    blockid = tl.load(LUT + offset + rbmn * 4 + 0)
    rowid = tl.load(LUT + offset + rbmn * 4 + 2)
    headid = tl.load(LUT + offset + rbmn * 4 + 3)
    # pointers to X
    px = X + pidz * stride_zx + blockid * BLOCK * BLOCK + rxm * BLOCK + rxn
    x = tl.load(px, mask=check, other=0)
    x = x.to(tl.float32)
    # computation
    out = tl.sum(x, axis=0)
    # pointers to OUT
    pout = OUT + pidz * stride_zout + headid * stride_hout + rowid * BLOCK + rxm
    tl.store(pout, out)


@triton.heuristics({'num_warps': lambda *args, **meta: num_warps(args[3] * meta['BLOCK'])})
@triton.heuristics({'TN': lambda *args, **meta: next_power_of_2(args[3]) * meta['BLOCK']})
@triton.jit
def _backward(DX, DOUT, LUT, sizemax, stride_zdx, stride_zdout, stride_hdout, **meta):
    pidhm = tl.program_id(0)
    pidz = tl.program_id(1)
    TN = meta['TN']
    BLOCK = meta['BLOCK']
    # create index ranges
    rxm = pidhm % BLOCK
    rbm = pidhm // BLOCK
    rxn = tl.arange(0, TN) % BLOCK
    rbn = tl.arange(0, TN) // BLOCK
    # extract information from look-up table
    header = LUT + rbm * 2
    size = tl.load(header + 0)
    offset = tl.load(header + 1)
    # bounds checking on lut
    check = rbn < size
    rbmn = tl.where(check, rbn, size - 1)
    # initialize pointers to block-sparse input
    blockid = tl.load(LUT + offset + rbmn * 4)
    rowid = tl.load(LUT + offset + rbmn * 4 + 2)
    headid = tl.load(LUT + offset + rbmn * 4 + 3)
    pdx = DX + pidz * stride_zdx + blockid * BLOCK * BLOCK + rxm * BLOCK + rxn
    pdout = DOUT + pidz * stride_zdout + headid * stride_hdout + rowid * BLOCK + rxm
    # Load
    # [2021-09-14] TD: Triton's broadcasting is very buggy, I have to read from dx (which is all
    # zeros) just so that I can broadcast dout (a scalar).
    dx_zeros = tl.load(pdx, mask=check, other=0)
    dout = tl.load(pdout)
    # Computation
    dx = dout - dx_zeros
    tl.store(pdx, dx, mask=check)


class _sum(torch.autograd.Function):
    @staticmethod
    def make_lut(layout, block, device):
        _empty = torch.tensor([], dtype=torch.int64, device=layout.device)
        sizes = _empty.clone()
        # sizes along rows
        for h in range(layout.shape[0]):
            sizes = torch.cat((sizes, layout[h, :, :].sum(-1)))
        # offsets in block format
        offsets = torch.zeros_like(sizes)
        offsets[1:] = torch.cumsum(sizes[:-1], dim=0)
        # block indices
        idx = torch.arange(layout.sum())
        head = layout.nonzero(as_tuple=False)[:, 0]
        rows = layout.nonzero(as_tuple=False)[:, 1]
        columns = layout.nonzero(as_tuple=False)[:, 2]
        core = torch.stack((idx, columns, rows, head), dim=1).view(-1)
        # construct look-up table
        offsets = offsets * 4 + 2 * sizes.numel()
        header = torch.stack((sizes, offsets), dim=1).view(-1)
        lut = torch.cat((header, core)).type(torch.int32).to(device)
        n_head = layout.shape[0]
        n_row = layout.shape[1] * block
        return lut, int(sizes.max()), n_head, n_row

    @staticmethod
    def forward(ctx, x, spdims, block, lut, maxlut, n_head, n_row, layout, bench, time):
        out = torch.zeros((x.shape[0], n_head, n_row), dtype=x.dtype, device=x.device)
        # run kernel
        M = x.shape[0]
        meta = {'BLOCK': block}
        grid = lambda opt: [spdims[0] * spdims[1] * block, M]
        _forward[grid](x, out, lut, maxlut, x.stride(0), out.stride(0), out.stride(1),
                       force_nc_cache=True, **meta)

        # save to context
        ctx.save_for_backward(x, lut, layout)
        ctx.spdims = spdims
        ctx.block = block
        ctx.maxlut = maxlut
        return out

    @staticmethod
    def backward(ctx, dout):
        # retrieve from context
        x, lut, layout = ctx.saved_tensors
        block = x.shape[-1]
        dx = sparsify_broadcast_tensor(dout, layout, block).expand(-1, -1, -1, block)
        # dx = torch.zeros_like(x)
        # run kernel
        # M = x.shape[0]
        # grid = lambda opt: [ctx.spdims[0] * ctx.spdims[1] * ctx.block, M]
        # _backward[grid](dx, dout, lut, ctx.maxlut, dx.stride(0), dout.stride(0), dout.stride(1),
        #                 force_nc_cache=True, BLOCK=ctx.block)
        return dx, None, None, None, None, None, None, None, None, None


class blocksparse_sum:

    apply_sum = _sum.apply

    def make_lut(self, device):
        key = (device, )
        if key not in self.lut_cache:
            self.lut_cache[key] = _sum.make_lut(self.layout, self.block, device)
        return self.lut_cache[key]

    def __init__(self, layout, block, bench=False):
        self.spdims = layout.shape
        self.layout = layout
        self.block = block
        self.bench = bench
        self.lut_cache = dict()

    def __call__(self, x):
        time_y = [None]
        lut, maxlut, n_head, n_row = self.make_lut(x.device)
        x = blocksparse_sum.apply_sum(
            x, self.spdims, self.block, lut, maxlut, n_head, n_row, self.layout, self.bench, time_y
        )
        return x
