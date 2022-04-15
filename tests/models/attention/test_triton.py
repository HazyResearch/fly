import torch
import triton
import pytest

# from triton.ops.blocksparse import matmul
from src.models.attention.blocksparse_matmul import matmul


@pytest.mark.parametrize(
    "MODE, TRANS_A, TRANS_B, BLOCK, DTYPE",
    [
        (mode, at, bt, block, dtype) for dtype in ["float32"] for mode in ["sdd"]
        for at in [False] for bt in [True] for block in [16]
    ],
)
# def test_matmul(MODE, TRANS_A, TRANS_B, BLOCK, DTYPE, Z=3, H=2, M=512, N=384, K=256):
def test_matmul(MODE, TRANS_A, TRANS_B, BLOCK, DTYPE, Z=3, H=2, M=64, N=64, K=48):
    DTYPE = {"float16": torch.float16, "float32": torch.float32}[DTYPE]
    # set seed
    torch.random.manual_seed(0)
    # create inputs
    a = torch.randn((Z, H, K, M) if TRANS_A else (Z, H, M, K), dtype=DTYPE, device="cuda")
    b = torch.randn((Z, H, N, K) if TRANS_B else (Z, H, K, N), dtype=DTYPE, device="cuda")
    shape = {
        "sdd": (M, N),
        "dsd": (a.shape[2], a.shape[3]),
        "dds": (b.shape[2], b.shape[3]),
    }[MODE]
    layout = torch.randint(2, (H, shape[0] // BLOCK, shape[1] // BLOCK))
    # triton result
    op = matmul(layout, BLOCK, MODE, trans_a=TRANS_A, trans_b=TRANS_B)
    ra = triton.testing.sparsify_tensor(a, layout, BLOCK) if MODE == "dsd" else a
    rb = triton.testing.sparsify_tensor(b, layout, BLOCK) if MODE == "dds" else b
    rc = triton.testing.catch_oor(lambda : op(ra, rb), pytest)
    # torch result
    ta = triton.testing.mask_tensor(a, layout, BLOCK) if MODE == "dsd" else a
    tb = triton.testing.mask_tensor(b, layout, BLOCK) if MODE == "dds" else b
    ta = ta.transpose(2, 3) if TRANS_A else ta
    tb = tb.transpose(2, 3) if TRANS_B else tb
    tc = torch.matmul(ta, tb)
    tc = triton.testing.mask_tensor(tc, layout, BLOCK) if MODE == "sdd" else tc
    tc = triton.testing.sparsify_tensor(tc, layout, BLOCK) if MODE == "sdd" else tc
    # compare
    assert torch.allclose(rc, tc)
