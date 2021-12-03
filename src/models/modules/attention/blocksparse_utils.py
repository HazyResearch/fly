import torch

from einops import rearrange, repeat


# Should do the same thing as https://github.com/openai/triton/blob/8bedcce9befbbe95d8fe0a082718edc4050e2831/python/triton/testing.py#L22
# but faster.
def sparsify_tensor(x, mask, block_size):
    """
    Arguments:
        x: (batch_size, n_head, T, S)
        mask: (n_head, T // block_size, S // block_size), with dtype torch.long
        block_size: int in {16, 32, 64, 128}
    Return:
        x_sparse: (batch_size, nnz(mask), block_size, block_size)
    """
    x_reshaped = rearrange(x, 'b h (p blk_sz) (r blk_sz1) -> b (h p r) blk_sz blk_sz1',
                           blk_sz=block_size, blk_sz1=block_size)
    return x_reshaped[:, mask.flatten().bool().to(x.device)]


def densify_tensor(x, mask, value=0.0):
    """
    Arguments:
        x: (batch_size, nnz, block_size, block_size)
        mask: (n_head, p, r), with dtype torch.long
    Return:
        x_dense: (batch_size, n_head, p * block_size, r * block_size)
    """
    mask = mask.bool().to(x.device)
    batch_size, nnz, block_size, block_size1 = x.shape
    n_head, p, r = mask.shape
    assert block_size == block_size1, 'Only square blocks are supported'
    assert nnz == mask.sum(), 'Mask has a different number of nonzero blocks than input'

    x_dense = torch.full((batch_size, n_head * p * r, block_size, block_size), value,
                          dtype=x.dtype, device=x.device)
    x_dense[:, mask.flatten()] = x
    return rearrange(x_dense, 'b (h p r) blk_sz blk_sz1 -> b h (p blk_sz) (r blk_sz1)', p=p, r=r)


# Should do the same thing as https://github.com/openai/triton/blob/8bedcce9befbbe95d8fe0a082718edc4050e2831/python/triton/testing.py#L52
# but faster.
def mask_tensor(x, mask, block_size, value=0.0):
    """
    Arguments:
        x: (batch_size, n_head, T, S)
        mask: (n_head, T // block_size, S // block_size), with dtype torch.long
        block_size: int in {16, 32, 64, 128}
    Return:
        x_sparse: (batch_size, nnz(mask), block_size, block_size)
    """
    n_head, p, r = mask.shape
    out = rearrange(x.clone(), 'b h (p blk_sz) (r blk_sz1) -> b (h p r) blk_sz blk_sz1',
                    blk_sz=block_size, blk_sz1=block_size)
    out[:, ~mask.flatten().bool().to(x.device)] = value
    return rearrange(out, 'b (h p r) blk_sz blk_sz1 -> b h (p blk_sz) (r blk_sz1)', p=p, r=r)


def sparsify_broadcast_tensor(x, mask, block_size):
    """
    Arguments:
        x: (batch_size, n_head, T)
        mask: (n_head, T // block_size, S // block_size), with dtype torch.long
        block_size: int in {16, 32, 64, 128}
    Return:
        x_sparse: (batch_size, nnz(mask), block_size, 1)
    """
    # x_expanded = repeat(x, 'b h (p blk_sz) -> b (h p r) blk_sz 1', blk_sz=block_size, r=mask.shape[2])
    # return x_expanded[:, mask.flatten().bool().to(x.device)]
    x_reshaped = rearrange(x, 'b h (p blk_sz) -> b h p blk_sz 1', blk_sz=block_size)
    h_idx, row_idx, _ = torch.nonzero(mask, as_tuple=True)
    return x_reshaped[:, h_idx, row_idx]


def block_frob_sqnorm_estimate(A, B, block_size, n_projs=None):
    """
    Estimate the Frobenius squared norm of the blocks of the matrix product A @ B, without
    materializing the product.
    Arguments:
        A: (m * block_size, k * block_size)
        B: (k * block_size, n * block_size)
        block_size: int
        n_projs: int, the number of random projections. Defaults to block_size // 4.
    Return:
        F: (m, n), where F[i, j] is the estimate of the Frobenius squared norm of the (i, j) block of
            C = A @ B.
    """
    if n_projs is None:
        n_projs = block_size // 4
    A_block = rearrange(A, '(m blk_sz) (k blk_sz1) -> m blk_sz k blk_sz1',
                        blk_sz=block_size, blk_sz1=block_size)
    B_block = rearrange(B, '(k blk_sz) (n blk_sz1) -> k blk_sz n blk_sz1',
                        blk_sz=block_size, blk_sz1=block_size)
    proj = torch.randn(k, block_size, n, n_projs, device=B.device) / n_projs ** 0.5
    C_block_sqnorm_estimate = torch.linalg.norm(
        torch.einsum('m s k t, k t n p -> m n s p', A_block, B_block @ proj),
        dim=(-1, -2)
    ) ** 2
    return C_block_sqnorm_estimate


if __name__ == '__main__':
    block_size = 32
    m = 5
    n = 7
    k = 3
    n_projs = 8
    # We use values from pretrained weights instead of random values just because random iid values
    # for A and B will yield Frob squared norm that are all of the same magnitude.
    # Pretrained weights are a bit more interesting
    import torchvision
    resnet18 = torchvision.models.resnet18(pretrained=True)
    A = resnet18.layer4[0].conv1.weight[:m * block_size, :k * block_size, 0, 0]
    B = resnet18.layer4[1].conv1.weight[:k * block_size, :n * block_size, 0, 0]
    C = A @ B
    C_block = rearrange(C, '(m blk_sz) (n blk_sz1) -> m n blk_sz blk_sz1',
                        blk_sz=block_size, blk_sz1=block_size)
    C_frob_sqnorm = torch.linalg.norm(C) ** 2
    proj = torch.randn(n * block_size, n_projs) / n_projs ** 0.5
    C_frob_sqnorm_estimate = torch.linalg.norm(A @ (B @ proj)) ** 2

    C_block_sqnorm = torch.linalg.norm(C_block, dim=(-1, -2)) ** 2
    C_block_sqnorm_estimate = block_frob_sqnorm_estimate(A, B, block_size)
    print(C_block_sqnorm)
    print(C_block_sqnorm_estimate)
