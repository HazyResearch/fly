import os
from pathlib import Path
current_dir = Path(__file__).parent.absolute()

import pytest

import torch

from src.datamodules.imdb import IMDB


def div_up(x: int, y: int) -> int:
    return (x + y - 1) // y


class TestIMDB:

    @pytest.mark.parametrize('val_split', [0.2, 0.0])
    @pytest.mark.parametrize('append_eos', [False, True])
    @pytest.mark.parametrize('append_bos', [False, True])
    @pytest.mark.parametrize('tokenizer_type', ['word', 'char'])
    def test_dims(self, tokenizer_type, append_bos, append_eos, val_split):
        batch_size = 57
        max_length = 1000
        seed = 2357
        vocab_min_freq = 5 if tokenizer_type == 'word' else 1
        data_dir = Path(os.getenv('DATA_DIR', current_dir.parent.parent / 'data'))
        cache_dir = data_dir / 'imdb' / 'cache'
        datamodule = IMDB(data_dir, cache_dir, max_length=max_length, tokenizer_type=tokenizer_type,
                          vocab_min_freq=vocab_min_freq, append_bos=append_bos,
                          append_eos=append_eos, val_split=val_split, batch_size=batch_size,
                          num_workers=4, seed=seed, shuffle=True)
        datamodule.prepare_data()
        datamodule.setup(stage='fit')
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        datamodule.setup(stage='test')
        test_loader = datamodule.test_dataloader()
        train_len = int(25000 * (1 - val_split))
        val_len = int(25000 * val_split) if val_split != 0.0 else 25000
        test_len = 25000
        assert len(train_loader) == div_up(train_len, batch_size)
        assert len(val_loader) == div_up(val_len, batch_size)
        assert len(test_loader) == div_up(test_len, batch_size)
        if tokenizer_type == 'char':
            assert datamodule.vocab_size <= 258  # Might need 2 extra for bos and eos
        for loader in [train_loader, val_loader, test_loader]:
            x, y, lengths = next(iter(loader))
            assert x.dim() == 2
            assert x.shape[0] == batch_size and x.shape[1] <= max_length
            assert x.dtype == torch.long
            assert y.shape == (batch_size,)
            assert y.dtype == torch.long
            assert lengths.shape == (batch_size,)
            assert lengths.dtype == torch.long
            assert torch.all(lengths <= max_length) and torch.all(lengths <= x.shape[1])
            if append_bos:
                assert torch.all(x[:, 0] == datamodule.vocab['<bos>'])
            if append_eos:
                assert torch.all(x[torch.arange(batch_size), lengths - 1]
                                 == datamodule.vocab['<eos>'])

        if val_split == 0.0 and not append_bos and not append_eos:
            l = torch.cat([lengths for _, _, lengths in train_loader])
            print(f"""Sequence length distribution: min {l.min().item()}, max {l.max().item()},
                    mean {l.float().mean().item()}, stddev {l.float().std().item()}""")
