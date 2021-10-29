import os
from pathlib import Path
current_dir = Path(__file__).parent.absolute()

import pytest

from munch import Munch

import torch

from src.datamodules.language_modeling import WikiText2, WikiText103


def div_up(x: int, y: int) -> int:
    return (x + y - 1) // y


class TestWikiText2:

    @pytest.mark.parametrize('vocab_type', ['word', 'bpe'])
    def test_dims(self, vocab_type):
        batch_size = 32
        max_length = 192
        seed = 2357
        num_shards = 8
        data_dir = Path(os.getenv('DATA_DIR', current_dir.parent.parent / 'data')) / 'wikitext-2'
        datamodule = WikiText2(data_dir, vocab_type=vocab_type,
                               batch_size=batch_size, max_length=max_length,
                               roll_seed=seed, batch_first=True)
        # Fake a trainer
        datamodule.trainer = Munch(global_rank=2, world_size=num_shards)
        datamodule.prepare_data()
        datamodule.setup(stage='fit')
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        datamodule.setup(stage='test')
        test_loader = datamodule.test_dataloader()
        if vocab_type == 'word':
            train_len = 2088628
            val_len = 217646
            test_len = 245569
            # Subtract 1 because the target is 1 off from the dataset
            assert len(train_loader) == div_up(train_len - 1, batch_size * num_shards * max_length)
            assert len(val_loader) == div_up(val_len - 1, batch_size * num_shards * max_length)
            assert len(test_loader) == div_up(test_len - 1, batch_size * num_shards * max_length)
        for loader in [train_loader, val_loader, test_loader]:
            x, y, length, _ = next(iter(loader))
            assert x.dim() == 2
            assert x.shape[0] == batch_size and x.shape[1] <= max_length
            assert x.dtype == torch.long
            assert y.dim() == 2
            assert y.shape[0] == batch_size and y.shape[1] <= max_length
            assert y.dtype == torch.long
            assert isinstance(length, int)
            assert length <= max_length and length <= x.shape[1]


class TestWikiText103:

    @pytest.mark.parametrize('vocab_type', ['word', 'bpe'])
    def test_dims(self, vocab_type):
        batch_size = 32
        max_length = 192
        seed = 2357
        num_shards = 8
        data_dir = Path(os.getenv('DATA_DIR', current_dir.parent.parent / 'data')) / 'wikitext-103'
        datamodule = WikiText103(data_dir, vocab_type=vocab_type,
                                 batch_size=batch_size, max_length=max_length,
                                 roll_seed=seed, batch_first=True)
        # Fake a trainer
        datamodule.trainer = Munch(global_rank=2, world_size=num_shards)
        datamodule.prepare_data()
        datamodule.setup(stage='fit')
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        datamodule.setup(stage='test')
        test_loader = datamodule.test_dataloader()
        if vocab_type == 'word':
            train_len = 103227021
            val_len = 217646
            test_len = 245569
            # Subtract 1 because the target is 1 off from the dataset
            assert len(train_loader) == div_up(train_len - 1, batch_size * num_shards * max_length)
            assert len(val_loader) == div_up(val_len - 1, batch_size * num_shards * max_length)
            assert len(test_loader) == div_up(test_len - 1, batch_size * num_shards * max_length)
        for loader in [train_loader, val_loader, test_loader]:
            x, y, length, _ = next(iter(loader))
            assert x.dim() == 2
            assert x.shape[0] == batch_size and x.shape[1] <= max_length
            assert x.dtype == torch.long
            assert y.dim() == 2
            assert y.shape[0] == batch_size and y.shape[1] <= max_length
            assert y.dtype == torch.long
            assert isinstance(length, int)
            assert length <= max_length and length <= x.shape[1]
