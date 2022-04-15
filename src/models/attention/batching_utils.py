import torch
from einops import rearrange


def batched_index_select(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Params:
        values: (1 or n_hashes, batch, seqlen, dim)
        indices: (n_hashes, batch, seqlen)
    Return:
        (n_hashes, batch, seqlen, dim)
    """
    last_dim = values.shape[-1]
    indices_expanded = rearrange(indices, '... -> ... 1').expand(*indices.shape, last_dim)
    return values.expand(*indices_expanded.shape[:-2],
                         *values.shape[-2:]).gather(-2, indices_expanded)
