import os
from pathlib import Path
current_dir = Path(__file__).parent.absolute()

import pytest

import torch

from src.datamodules.language_modeling_hf import LMDataModule


def div_up(x: int, y: int) -> int:
    return (x + y - 1) // y


class TestLMDataModule:

    def test_wikitext2(self):
        batch_size = 57
        dataset_name = 'wikitext'
        dataset_config_name = 'wikitext-2-raw-v1'
        data_dir = Path(os.getenv('DATA_DIR', current_dir.parent.parent / 'data'))
        cache_dir = data_dir / 'wikitext-2' / 'cache'
        block_size = 1024
        datamodule = LMDataModule(dataset_name, tokenizer_name='gpt2',
                                  dataset_config_name=dataset_config_name,
                                  block_size=block_size, cache_dir=cache_dir,
                                  batch_size=batch_size, num_workers=4, shuffle=True)
        datamodule.prepare_data()
        datamodule.setup(stage='fit')
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        datamodule.setup(stage='test')
        test_loader = datamodule.test_dataloader()
        train_len = 2391884
        val_len = 247289
        test_len = 283287
        # Since we discard some text, check that length is at least 90%
        # of what we expect
        assert 0.9 <= len(train_loader) / div_up(train_len, batch_size * block_size) <= 1.0
        assert 0.9 <= len(val_loader) / div_up(val_len, batch_size * block_size) <= 1.0
        assert 0.9 <= len(test_loader) / div_up(test_len, batch_size * block_size) <= 1.0

        for loader in [train_loader, val_loader, test_loader]:
            x = next(iter(loader))['input_ids']
            assert x.dim() == 2
            assert x.shape == (batch_size, block_size)
            assert x.dtype == torch.long

    def test_wikitext103(self):
        batch_size = 57
        dataset_name = 'wikitext'
        dataset_config_name = 'wikitext-103-raw-v1'
        data_dir = Path(os.getenv('DATA_DIR', current_dir.parent.parent / 'data'))
        cache_dir = data_dir / 'wikitext-103' / 'cache'
        block_size = 1024
        datamodule = LMDataModule(dataset_name, tokenizer_name='gpt2',
                                  dataset_config_name=dataset_config_name,
                                  block_size=block_size, cache_dir=cache_dir,
                                  batch_size=batch_size, num_workers=4, shuffle=True)
        datamodule.prepare_data()
        datamodule.setup(stage='fit')
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        datamodule.setup(stage='test')
        test_loader = datamodule.test_dataloader()
        train_len = 117920140
        val_len = 247289
        test_len = 283287
        # Since we discard some text, check that length is at least 90%
        # of what we expect
        assert 0.9 <= len(train_loader) / div_up(train_len, batch_size * block_size) <= 1.0
        assert 0.9 <= len(val_loader) / div_up(val_len, batch_size * block_size) <= 1.0
        assert 0.9 <= len(test_loader) / div_up(test_len, batch_size * block_size) <= 1.0

        for loader in [train_loader, val_loader, test_loader]:
            x = next(iter(loader))['input_ids']
            assert x.dim() == 2
            assert x.shape == (batch_size, block_size)
            assert x.dtype == torch.long
