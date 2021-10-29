import torch.nn.functional as F

from src.models.modules.masking import FullMask, LengthMask


def pad_mask(mask, pad_length, left=True, value=True):
    assert value in [True, False]
    assert isinstance(mask, (FullMask, LengthMask))
    if isinstance(mask, FullMask):
        pad = (pad_length, 0) if left else (0, pad_length)
        return FullMask(F.pad(mask._mask, pad, value=value))
    elif isinstance(mask, LengthMask):
        if value:
            return LengthMask(mask._lengths + pad_length, max_len=mask._max_len + pad_length,
                              device=mask._lengths.device)
        else:
            return LengthMask(mask._lengths, max_len=mask._max_len + pad_length,
                              device=mask._lengths.device)
