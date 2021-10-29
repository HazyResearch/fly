import os
from pathlib import Path
current_dir = Path(__file__).parent.absolute()

import pytest

import torch

from src.datamodules.pathfinder import PathFinder


def div_up(x: int, y: int) -> int:
    return (x + y - 1) // y


class TestPathFinder:

    @pytest.mark.parametrize('test_split', [0.1, 0.0])
    @pytest.mark.parametrize('val_split', [0.1, 0.0])
    @pytest.mark.parametrize('to_int', [False, True])
    @pytest.mark.parametrize('sequential', [False, True])
    @pytest.mark.parametrize('level', ['easy', 'intermediate', 'hard'])
    @pytest.mark.parametrize('resolution', [32, 64, 128, 256])
    @pytest.mark.parametrize('use_tar_dataset', [False, True])
    def test_dims(self, use_tar_dataset, resolution, level, sequential, to_int, val_split,
                  test_split):
        batch_size = 57
        seed = 2357
        data_dir = (Path(os.getenv('DATA_DIR', current_dir.parent.parent / 'data'))
                    / 'pathfinder')
        if use_tar_dataset:
            data_dir = data_dir / f'pathfinder{resolution}.tar'
        datamodule = PathFinder(data_dir, resolution, level, sequential=sequential, to_int=to_int,
                                val_split=val_split, test_split=test_split, batch_size=batch_size,
                                seed=seed, shuffle=True, num_workers=4)
        datamodule.prepare_data()
        datamodule.setup(stage='fit')
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        datamodule.setup(stage='test')
        test_loader = datamodule.test_dataloader()
        # There's an empty file in the pathfinder32 easy dataset
        dataset_len = 199999 if resolution == 32 and level == 'easy' else 200000
        assert (len(datamodule.dataset_train) + len(datamodule.dataset_val)
                + len(datamodule.dataset_test)) == dataset_len
        val_len = int(dataset_len * val_split)
        test_len = int(dataset_len * test_split)
        train_len = dataset_len - val_len - test_len
        assert len(train_loader) == div_up(train_len, batch_size)
        assert len(val_loader) == div_up(val_len, batch_size)
        assert len(test_loader) == div_up(test_len, batch_size)
        for loader in [train_loader] + (([] if val_split == 0.0 else [val_loader])
                                        + ([] if test_split == 0.0 else [test_loader])):
            x, y = next(iter(loader))
            assert x.shape == (batch_size,) + datamodule.dims
            assert x.dtype == torch.float if not to_int else torch.long
            assert y.shape == (batch_size,)
            assert y.dtype == torch.long

        if val_split == 0.0 and test_split == 0.0 and sequential and not to_int:
            xs_ys = [(x, y) for x, y in train_loader]
            xs, ys = zip(*xs_ys)
            xs, ys = torch.cat(xs), torch.cat(ys)
            xs = xs * 255
            x_mean, x_std = xs.float().mean(), xs.float().std()
            print(f"Pixel distribution: mean {x_mean.item()}, stddev {x_std.item()}")
