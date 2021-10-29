import torch
import torch.nn.functional as F


def kl(p, q):
    kl_dis = F.kl_div(p, q)
    return kl_dis


def mse(p, q):
    mse_loss = F.mse_loss(p, q)
    return mse_loss


def l1(p, q):
    l1_loss = F.l1_loss(p, q)
    return l1_loss


def smart_sort(x, permutation):
    d1, d2 = x.size()
    ret = x[
        torch.arange(d1).unsqueeze(1).repeat((1, d2)).flatten(),
        permutation.flatten()
    ].view(d1, d2)
    return ret


def sparsify(target, params_reduction):
    target_sparse = target.clone()
    N, target_l, seq_l = target_sparse.shape
    sorted_tensor, indices_tensor = torch.sort(target_sparse, dim=-1, descending=True)
    topk = int(round(seq_l*(1-params_reduction)))
    mask = torch.zeros_like(target_sparse, dtype=torch.bool).scatter_(-1, indices_tensor[:,:, :topk], 1)
    target_sparse[~mask] = float('-inf')  # To zero out these values, we set their logit to be -inf, so that after softmax they are zero
    return target_sparse, mask.bool()


def low_rank(target, sparsity):
    N, target_l, seq_l = target.shape
    target_lr = target.clone()
    try:
        u, s, v = torch.svd(target_lr)
        topk = int(round(seq_l * (1 - sparsity)))
        # assert torch.dist(target_lr, torch.matmul(torch.matmul(u, torch.diag_embed(s)), v.transpose(-2, -1)))<1e-2
        s[:, topk:] = 0
        target_lr = torch.matmul(torch.matmul(u, torch.diag_embed(s)), v.transpose(-2, -1))
        return target_lr, True
    except:                     # torch.svd may have convergence issues for GPU and CPU.
        return target_lr, False


def log_stats(approx, target):
    eps = 1e-5
    sparse_l1 = l1(approx, target)
    sparse_kl = kl(torch.log(approx+eps), target+eps)
    sparse_kl_inverse = kl(torch.log(target+eps), approx+eps)
    return torch.cat([sparse_l1.view(1), sparse_kl.view(1), sparse_kl_inverse.view(1)])


def compute_distance(target_raw, params_reduction, alpha=0.5):
    stats = torch.zeros([3, 3])
    target_raw[target_raw < -1e7] = float('-inf')
    target = F.softmax(target_raw, dim=-1)

    # sparse
    target_sparse, mask = sparsify(target_raw, params_reduction)
    stats[0, :] = log_stats(torch.softmax(target_sparse, dim=-1), target)

    # low_rank
    new_sparsity = 1-(1-params_reduction)/2
    target_lr, succeed = low_rank(target, new_sparsity)
    if succeed:
        target_lr[target_lr < 0] = 0.0
        target_lr = F.normalize(target_lr, p=1, dim=-1)
        stats[1, :] = log_stats(target_lr, target)

    # sparse+low_rank
    target_sparse = target.clone()
    params_sparse = alpha*(1-params_reduction)
    _, mask = sparsify(target, 1-params_sparse)
    target_sparse[~mask] = 0.0
    target_sparse_lr = target - target_sparse
    params_lr = (1-alpha)*(1-params_reduction)/2
    target_sparse_lr, succeed = low_rank(target_sparse_lr, 1-params_lr)
    if succeed:
        target_sparse_lr[target_sparse_lr < 0] = 0.0
        target_sparse_lr = F.normalize(target_sparse_lr + target_sparse, p=1, dim=-1)
        stats[2, :] = log_stats(target_sparse_lr, target)
    return stats, succeed & succeed


def compute_single_distance(target_raw, attn_mask, params_reduction, approx_type, alpha=0.5):
    stats = torch.zeros([1, 3])
    target_raw[target_raw < -1e7] = float('-inf')
    target = F.softmax(target_raw, dim=-1)
    succeed = True
    approx_target = 0

    # sparse
    if approx_type == "sparse":
        target_sparse, mask = sparsify(target_raw, params_reduction)
        if attn_mask is not None:
            target_sparse.masked_fill_(attn_mask, float('-inf'),)
        approx_target = torch.softmax(target_sparse, dim=-1)
        stats = log_stats(approx_target, target)

    # low_rank
    elif approx_type == "low_rank":
        new_sparsity = 1-(1-params_reduction)/2
        target_lr, succeed = low_rank(target, new_sparsity)
        if succeed:
            target_lr[target_lr < 0] = 0.0
            if attn_mask is not None:
                target_lr.masked_fill_(attn_mask, 0.0, )
            approx_target = F.normalize(target_lr, p=1, dim=-1)
            stats = log_stats(approx_target, target)

    # sparse+low_rank
    elif approx_type == "sparse_low_rank":
        target_sparse = target.clone()
        params_sparse = alpha*(1-params_reduction)
        _, mask = sparsify(target, 1-params_sparse)
        target_sparse[~mask] = 0.0
        target_sparse_lr = target - target_sparse
        params_lr = (1-alpha)*(1-params_reduction)/2
        target_sparse_lr, succeed = low_rank(target_sparse_lr, 1-params_lr)
        if succeed:
            target_sparse_lr[target_sparse_lr < 0] = 0.0
            target_sparse_lr += target_sparse
            if attn_mask is not None:
                target_sparse_lr.masked_fill_(attn_mask, 0.0, )
            approx_target = F.normalize(target_sparse_lr, p=1, dim=-1)
            stats = log_stats(approx_target, target)
    else:
        print("Approximation type is not implemented")
    return approx_target, stats, succeed
