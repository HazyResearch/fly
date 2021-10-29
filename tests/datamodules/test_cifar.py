import os
from pathlib import Path
current_dir = Path(__file__).parent.absolute()

import pytest

import torch

from src.datamodules.cifar import CIFAR10, CIFAR100


def div_up(x: int, y: int) -> int:
    return (x + y - 1) // y


class TestCIFAR:

    @pytest.mark.parametrize('normalize', [False, True])
    @pytest.mark.parametrize('val_split', [0.2, 0.0])
    @pytest.mark.parametrize('to_int', [False, True])
    @pytest.mark.parametrize('data_augmentation', [None, 'standard', 'autoaugment'])
    @pytest.mark.parametrize('grayscale', [False, True])
    @pytest.mark.parametrize('sequential', [False, True])
    @pytest.mark.parametrize('cls', [CIFAR10, CIFAR100])
    def test_dims(self, cls, sequential, grayscale, data_augmentation, to_int, val_split,
                  normalize):
        if to_int and normalize:  # Not compatible
            return
        batch_size = 57
        seed = 2357
        data_dir = Path(os.getenv('DATA_DIR', current_dir.parent.parent / 'data')) / 'cifar'
        datamodule = cls(data_dir, sequential=sequential, grayscale=grayscale,
                         data_augmentation=data_augmentation, to_int=to_int, val_split=val_split,
                         normalize=normalize, batch_size=batch_size, seed=seed, shuffle=True)
        datamodule.prepare_data()
        datamodule.setup(stage='fit')
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        datamodule.setup(stage='test')
        test_loader = datamodule.test_dataloader()
        train_len = int(50000 * (1 - val_split))
        val_len = int(50000 * val_split)
        test_len = 10000
        assert len(train_loader) == div_up(train_len, batch_size)
        assert len(val_loader) == div_up(val_len, batch_size)
        assert len(test_loader) == div_up(test_len, batch_size)
        for loader in [train_loader] + ([] if val_split == 0.0 else [val_loader]) + [test_loader]:
            x, y = next(iter(loader))
            assert x.shape == (batch_size,) + datamodule.dims
            assert x.dtype == torch.float if not to_int else torch.long
            assert y.shape == (batch_size,)
            assert y.dtype == torch.long

        # Check that it's actually normalized
        if normalize and data_augmentation is None and val_split == 0.0:
            xs_ys = [(x, y) for x, y in train_loader]
            xs, ys = zip(*xs_ys)
            xs, ys = torch.cat(xs), torch.cat(ys)
            dims_to_reduce = (0, 2, 3) if not sequential else (0, 1)
            x_mean, x_std = xs.mean(dim=dims_to_reduce), xs.std(dim=dims_to_reduce)
            assert torch.allclose(x_mean, torch.zeros_like(x_mean), atol=1e-3)
            assert torch.allclose(x_std, torch.ones_like(x_mean), atol=1e-3)
