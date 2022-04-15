import torch
import torch.nn.functional as F


# Adapted from https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/autopadder.py
def pad_to_multiple(tensor, multiple, dims=-1, value=0):
    try:
        dims = list(dims)  # If dims is an iterable (e.g., List, Tuple)
    except:
        dims = [dims]
    # convert dims from negative to positive
    dims = [d if d >= 0 else tensor.ndim + d for d in dims]
    padding = [0] * (2 * tensor.ndim)
    for d in dims:
        size = tensor.size(d)
        # Pytorch's JIT doesn't like divmod
        # m, remainder = divmod(size, multiple)
        m = size // multiple
        remainder = size - m * multiple
        if remainder != 0:
            padding[2 * (tensor.ndim - d - 1) + 1] = multiple - remainder
    if all(p == 0 for p in padding):
        return tensor
    else:
        return F.pad(tensor, tuple(padding), value=value)
