import torch


def sparse_project(M, density):
    """Return a sparse mask of the largest entries of M in magnitude.
    """
    nparams = int(density * M.numel())
    # Implementation 1
    # sorted_idx = torch.argsort(M.abs().flatten(), descending=True)
    # threashold = M.abs().flatten()[sorted_idx[nparams]]
    # Implementation 2
    # threashold = M.abs().flatten().kthvalue(M.numel() - nparams).values
    # sparse_mask = M.abs() > threashold
    # Implementation 3
    _, topk_idx = torch.topk(M.abs().flatten(), nparams, sorted=False)
    sparse_mask = torch.zeros_like(M, dtype=torch.bool).flatten()
    # scatter_ is faster than index assignment for some reason
    sparse_mask.scatter_(dim=0, index=topk_idx, src=torch.ones_like(sparse_mask))
    # sparse_mask[topk_idx] = True
    sparse_mask = sparse_mask.reshape(M.shape)
    return sparse_mask
