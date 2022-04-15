import os
from pathlib import Path
current_dir = Path(__file__).parent.absolute()

import pytest

import torch

from src.datamodules.masked_language_modeling import MLMDataModule


def div_up(x: int, y: int) -> int:
    return (x + y - 1) // y


class TestMLMDataModule:

    def test_wikitext2(self):
        batch_size = 7
        dataset_name = 'wikitext'
        dataset_config_name = 'wikitext-2-raw-v1'
        data_dir = Path(os.getenv('DATA_DIR', current_dir.parent.parent / 'data'))
        cache_dir = data_dir / 'wikitext-2' / 'cache'
        max_length = 512
        dupe_factor = 2
        datamodule = MLMDataModule(dataset_name, tokenizer_name='bert-base-uncased',
                                   dataset_config_name=dataset_config_name,
                                   max_length=max_length, cache_dir=cache_dir,
                                   dupe_factor=dupe_factor, batch_size=batch_size,
                                   num_workers_preprocess=4)
        datamodule.prepare_data()
        datamodule.setup(stage='fit')
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        datamodule.setup(stage='test')
        test_loader = datamodule.test_dataloader()
        train_len = 69293
        val_len = 7222
        test_len = 8353
        assert len(train_loader) == div_up(train_len, batch_size)
        assert len(val_loader) == div_up(val_len, batch_size)
        assert len(test_loader) == div_up(test_len, batch_size)
        for loader in [train_loader, val_loader, test_loader]:
            batch = next(iter(loader))
            assert batch.keys() == {'input_ids', 'labels', 'attention_mask', 'token_type_ids', 'next_sentence_label'}
            seqlen = batch['input_ids'].shape[-1]
            assert batch['input_ids'].shape == (batch_size, seqlen)
            assert batch['input_ids'].dtype == torch.long
            assert batch['labels'].shape == (batch_size, seqlen)
            assert batch['labels'].dtype == torch.long
            assert batch['attention_mask'].shape == (batch_size, seqlen)
            assert batch['attention_mask'].dtype == torch.long
            assert batch['token_type_ids'].shape == (batch_size, seqlen)
            assert batch['token_type_ids'].dtype == torch.long
            assert batch['next_sentence_label'].shape == (batch_size,)
            assert batch['next_sentence_label'].dtype == torch.bool

    def test_wikipedia(self):
        batch_size = 8
        dataset_name = 'wikipedia'
        dataset_config_name = '20200501.en'
        data_dir = Path(os.getenv('DATA_DIR', current_dir.parent.parent / 'data'))
        cache_dir = data_dir / 'wikipedia' / 'cache'
        max_length = 512
        dupe_factor = 2
        datamodule = MLMDataModule(dataset_name, tokenizer_name='bert-base-uncased',
                                   dataset_config_name=dataset_config_name,
                                   max_length=max_length, cache_dir=cache_dir,
                                   dupe_factor=dupe_factor, batch_size=batch_size,
                                   num_workers_preprocess=32)
        datamodule.prepare_data()
        datamodule.setup(stage='fit')
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        datamodule.setup(stage='test')
        test_loader = datamodule.test_dataloader()
        for loader in [train_loader, val_loader, test_loader]:
            batch = next(iter(loader))
            assert batch.keys() == {'input_ids', 'labels', 'attention_mask', 'token_type_ids', 'next_sentence_label'}
            seqlen = batch['input_ids'].shape[-1]
            assert batch['input_ids'].shape == (batch_size, seqlen)
            assert batch['input_ids'].dtype == torch.long
            assert batch['labels'].shape == (batch_size, seqlen)
            assert batch['labels'].dtype == torch.long
            assert batch['attention_mask'].shape == (batch_size, seqlen)
            assert batch['attention_mask'].dtype == torch.bool
            assert batch['token_type_ids'].shape == (batch_size, seqlen)
            assert batch['token_type_ids'].dtype == torch.bool
            assert batch['next_sentence_label'].shape == (batch_size,)
            assert batch['next_sentence_label'].dtype == torch.bool

    @pytest.mark.parametrize('max_length', [128, 512])
    def test_wikipedia_from_text(self, max_length):
        batch_size = 8
        data_dir = Path(os.getenv('DATA_DIR', current_dir.parent.parent / 'data')) / 'bert' / 'wikicorpus_text'
        path = str(data_dir)
        cache_dir = data_dir.parent / 'wikicorpus' / 'cache'
        dupe_factor = 5
        datamodule = MLMDataModule(path, tokenizer_name='bert-base-uncased',
                                   max_length=max_length, cache_dir=cache_dir,
                                   dupe_factor=dupe_factor, batch_size=batch_size,
                                   num_workers_preprocess=64)
        datamodule.prepare_data()
        datamodule.setup(stage='fit')
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        datamodule.setup(stage='test')
        test_loader = datamodule.test_dataloader()
        for loader in [train_loader, val_loader, test_loader]:
            batch = next(iter(loader))
            assert batch.keys() == {'input_ids', 'labels', 'attention_mask', 'token_type_ids', 'next_sentence_label'}
            seqlen = batch['input_ids'].shape[-1]
            assert batch['input_ids'].shape == (batch_size, seqlen)
            assert batch['input_ids'].dtype == torch.long
            assert batch['labels'].shape == (batch_size, seqlen)
            assert batch['labels'].dtype == torch.long
            assert batch['attention_mask'].shape == (batch_size, seqlen)
            # Could be bool or long, depending on whether the sequences were padding by the tokenizer
            assert batch['attention_mask'].dtype in [torch.bool, torch.long]
            assert batch['token_type_ids'].shape == (batch_size, seqlen)
            assert batch['token_type_ids'].dtype in [torch.bool, torch.long]
            assert batch['next_sentence_label'].shape == (batch_size,)
            assert batch['next_sentence_label'].dtype == torch.bool

    @pytest.mark.parametrize('max_length', [128, 512])
    def test_bookcorpus(self, max_length):
        batch_size = 8
        # "bookcorpus" has already processed the books into sentences, which is not what we want
        dataset_name = 'bookcorpusopen'
        data_dir = Path(os.getenv('DATA_DIR', current_dir.parent.parent / 'data')) / 'bert' / 'bookcorpus'
        # cache_dir = data_dir / 'cache'
        cache_dir = None
        dupe_factor = 5
        datamodule = MLMDataModule(dataset_name, tokenizer_name='bert-base-uncased',
                                   max_length=max_length, cache_dir=cache_dir,
                                   dupe_factor=dupe_factor, batch_size=batch_size,
                                   num_workers_preprocess=64)
        datamodule.prepare_data()
        datamodule.setup(stage='fit')
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        datamodule.setup(stage='test')
        test_loader = datamodule.test_dataloader()
        for loader in [train_loader, val_loader, test_loader]:
            batch = next(iter(loader))
            assert batch.keys() == {'input_ids', 'labels', 'attention_mask', 'token_type_ids', 'next_sentence_label'}
            seqlen = batch['input_ids'].shape[-1]
            assert batch['input_ids'].shape == (batch_size, seqlen)
            assert batch['input_ids'].dtype == torch.long
            assert batch['labels'].shape == (batch_size, seqlen)
            assert batch['labels'].dtype == torch.long
            assert batch['attention_mask'].shape == (batch_size, seqlen)
            assert batch['attention_mask'].dtype in [torch.bool, torch.long]
            assert batch['token_type_ids'].shape == (batch_size, seqlen)
            assert batch['token_type_ids'].dtype in [torch.bool, torch.long]
            assert batch['next_sentence_label'].shape == (batch_size,)
            assert batch['next_sentence_label'].dtype == torch.bool

    @pytest.mark.parametrize('max_length', [128, 512])
    def test_wikibook_from_text(self, max_length):
        batch_size = 8
        data_dir_common = Path(os.getenv('DATA_DIR', current_dir.parent.parent / 'data')) / 'bert'
        path = [str(data_dir_common / 'wikicorpus_text'), 'bookcorpusopen']
        cache_dir = [data_dir_common / 'wikicorpus' / 'cache', data_dir_common / 'bookcorpus' / 'cache']
        dupe_factor = 5
        datamodule = MLMDataModule(path, tokenizer_name='bert-base-uncased',
                                   max_length=max_length, cache_dir=cache_dir,
                                   dupe_factor=dupe_factor, batch_size=batch_size,
                                   num_workers_preprocess=64)
        datamodule.prepare_data()
        datamodule.setup(stage='fit')
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        datamodule.setup(stage='test')
        test_loader = datamodule.test_dataloader()
        for loader in [train_loader, val_loader, test_loader]:
            batch = next(iter(loader))
            assert batch.keys() == {'input_ids', 'labels', 'attention_mask', 'token_type_ids', 'next_sentence_label'}
            seqlen = batch['input_ids'].shape[-1]
            assert batch['input_ids'].shape == (batch_size, seqlen)
            assert batch['input_ids'].dtype == torch.long
            assert batch['labels'].shape == (batch_size, seqlen)
            assert batch['labels'].dtype == torch.long
            assert batch['attention_mask'].shape == (batch_size, seqlen)
            assert batch['attention_mask'].dtype in [torch.bool, torch.long]
            assert batch['token_type_ids'].shape == (batch_size, seqlen)
            assert batch['token_type_ids'].dtype in [torch.bool, torch.long]
            assert batch['next_sentence_label'].shape == (batch_size,)
            assert batch['next_sentence_label'].dtype == torch.bool
