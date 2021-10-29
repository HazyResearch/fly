import os
from pathlib import Path
current_dir = Path(__file__).parent.absolute()

import pytest

import torch

from src.datamodules.aan import AAN


def div_up(x: int, y: int) -> int:
    return (x + y - 1) // y


class TestAAN:

    @pytest.mark.parametrize('append_eos', [False, True])
    @pytest.mark.parametrize('append_bos', [False, True])
    def test_dims(self, append_bos, append_eos):
        batch_size = 57
        max_length = 4000
        data_dir = Path(os.getenv('DATA_DIR', current_dir.parent.parent / 'data')) / 'aan' / 'tsv_data'
        cache_dir = data_dir.parent / 'cache'
        datamodule = AAN(data_dir, cache_dir, max_length=max_length, append_bos=append_bos,
                         append_eos=append_eos, batch_size=batch_size, shuffle=True,
                         num_workers=4)
        datamodule.prepare_data()
        datamodule.setup(stage='fit')
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        datamodule.setup(stage='test')
        test_loader = datamodule.test_dataloader()
        train_len = 147086
        val_len = 18090
        test_len = 17437
        assert len(train_loader) == div_up(train_len, batch_size)
        assert len(val_loader) == div_up(val_len, batch_size)
        assert len(test_loader) == div_up(test_len, batch_size)
        assert datamodule.vocab_size <= 258  # Might need 2 extra for bos and eos
        for loader in [train_loader, val_loader, test_loader]:
            x1, x2, y, lengths1, lengths2 = next(iter(loader))
            assert x1.dim() == 2
            assert x1.shape[0] == batch_size and x1.shape[1] <= max_length
            assert x1.dtype == torch.long
            assert x2.dim() == 2
            assert x2.shape[0] == batch_size and x2.shape[1] <= max_length
            assert x2.dtype == torch.long
            assert y.shape == (batch_size,)
            assert y.dtype == torch.long
            assert lengths1.shape == (batch_size,)
            assert lengths1.dtype == torch.long
            assert torch.all(lengths1 <= max_length) and torch.all(lengths1 <= x1.shape[1])
            assert lengths2.shape == (batch_size,)
            assert lengths2.dtype == torch.long
            assert torch.all(lengths2 <= max_length) and torch.all(lengths2 <= x2.shape[1])
            if append_bos:
                assert torch.all(x1[:, 0] == datamodule.vocab['<bos>'])
                assert torch.all(x2[:, 0] == datamodule.vocab['<bos>'])
            if append_eos:
                assert torch.all(x1[torch.arange(batch_size), lengths1 - 1]
                                 == datamodule.vocab['<eos>'])
                assert torch.all(x2[torch.arange(batch_size), lengths2 - 1]
                                 == datamodule.vocab['<eos>'])

        if not append_bos and not append_eos:
            for loader in [train_loader, val_loader, test_loader]:
                l1, l2 = zip(*[(lengths1, lengths2) for _, _, _, lengths1, lengths2 in loader])
                l1, l2 = torch.cat(l1), torch.cat(l2)
                print(f"""Sequence1 length distribution: min {l1.min().item()}, max {l1.max().item()},
                        mean {l1.float().mean().item()}, stddev {l1.float().std().item()}""")
                print(f"""Sequence2 length distribution: min {l2.min().item()}, max {l2.max().item()},
                        mean {l2.float().mean().item()}, stddev {l2.float().std().item()}""")
