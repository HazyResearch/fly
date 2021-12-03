from pathlib import Path
current_dir = Path(__file__).parent.absolute()

import pickle
import logging
from typing import Any, List, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import torchtext
from datasets import load_dataset, DatasetDict

from pytorch_lightning import LightningDataModule


# Adapted from https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2_lstm.ipynb
class IMDB(LightningDataModule):

    dataset_name = 'imdb'
    num_classes = 2

    def __init__(self, data_dir=current_dir, cache_dir=None, max_length=512, tokenizer_type='word',
                 vocab_min_freq=1, append_bos=False, append_eos=False, val_split=0.0,
                 batch_size=32, num_workers=1, seed=42, shuffle=False, pin_memory=False,
                 drop_last=False, **kwargs):
        """If cache_dir is not None, we'll cache the processed dataset there.
        """
        super().__init__(**kwargs)
        self.data_dir = Path(data_dir).expanduser()
        self.cache_dir = None if cache_dir is None else Path(cache_dir).expanduser()
        self.max_length = max_length
        assert tokenizer_type in ['word', 'char'], f'tokenizer_type {tokenizer_type} not supported'
        self.tokenizer_type = tokenizer_type
        self.vocab_min_freq = vocab_min_freq
        self.append_bos = append_bos
        self.append_eos = append_eos
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def prepare_data(self):
        if self.cache_dir is None:  # Just download the dataset
            load_dataset(self.dataset_name, cache_dir=self.data_dir)
        else:  # Process the dataset and save it
            self.process_dataset()

    def setup(self, stage=None):
        if stage == 'test' and hasattr(self, 'dataset_test'):
            return
        dataset, self.tokenizer, self.vocab = self.process_dataset()
        self.vocab_size = len(self.vocab)
        dataset.set_format(type='torch', columns=['input_ids', 'label'])

        # Create all splits
        dataset_train, dataset_test = dataset['train'], dataset['test']
        if self.val_split == 0.0:  # Use test set as val set, as done in the LRA paper
            self.dataset_train, self.dataset_val = dataset_train, dataset_test
        else:
            train_val = dataset_train.train_test_split(test_size=self.val_split, seed=self.seed)
            self.dataset_train, self.dataset_val = train_val['train'], train_val['test']
        self.dataset_test = dataset_test

        def collate_batch(batch):
            xs, ys = zip(*[(data['input_ids'], data['label']) for data in batch])
            lengths = torch.tensor([len(x) for x in xs])
            xs = nn.utils.rnn.pad_sequence(xs, padding_value=self.vocab['<pad>'], batch_first=True)
            ys = torch.tensor(ys)
            return xs, ys, lengths

        self.collate_fn = collate_batch

    def process_dataset(self):
        cache_dir = None if self.cache_dir is None else self.cache_dir / self._cache_dir_name
        if cache_dir is not None:
            if cache_dir.is_dir():
                return self._load_from_cache(cache_dir)

        dataset = load_dataset(self.dataset_name, cache_dir=self.data_dir)
        dataset = DatasetDict(train=dataset['train'], test=dataset['test'])
        if self.tokenizer_type == 'word':
            tokenizer = torchtext.data.utils.get_tokenizer('spacy', language='en_core_web_sm')
        else:  # self.tokenizer_type == 'char'
            tokenizer = list  # Just convert a string to a list of chars
        # Account for <bos> and <eos> tokens
        max_length = self.max_length - int(self.append_bos) - int(self.append_eos)
        tokenize = lambda example: {'tokens': tokenizer(example['text'])[:max_length]}
        dataset = dataset.map(tokenize, remove_columns=['text'], keep_in_memory=True,
                              load_from_cache_file=False, num_proc=max(self.num_workers, 1))
        vocab = torchtext.vocab.build_vocab_from_iterator(
            dataset['train']['tokens'],
            min_freq=self.vocab_min_freq,
            specials=(['<pad>', '<unk>']
                      + (['<bos>'] if self.append_bos else [])
                      + (['<eos>'] if self.append_eos else []))
        )
        vocab.set_default_index(vocab['<unk>'])

        numericalize = lambda example: {'input_ids': vocab(
            (['<bos>'] if self.append_bos else [])
            + example['tokens']
            + (['<eos>'] if self.append_eos else [])
        )}
        dataset = dataset.map(numericalize, remove_columns=['tokens'], keep_in_memory=True,
                              load_from_cache_file=False, num_proc=max(self.num_workers, 1))

        if cache_dir is not None:
            self._save_to_cache(dataset, tokenizer, vocab, cache_dir)
        return dataset, tokenizer, vocab

    def _save_to_cache(self, dataset, tokenizer, vocab, cache_dir):
        cache_dir = self.cache_dir / self._cache_dir_name
        logger = logging.getLogger(__name__)
        logger.info(f'Saving to cache at {str(cache_dir)}')
        dataset.save_to_disk(str(cache_dir))
        with open(cache_dir / 'tokenizer.pkl', 'wb') as f:
            pickle.dump(tokenizer, f)
        with open(cache_dir / 'vocab.pkl', 'wb') as f:
            pickle.dump(vocab, f)

    def _load_from_cache(self, cache_dir):
        assert cache_dir.is_dir()
        logger = logging.getLogger(__name__)
        logger.info(f'Load from cache at {str(cache_dir)}')
        dataset = DatasetDict.load_from_disk(str(cache_dir))
        with open(cache_dir / 'tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        with open(cache_dir / 'vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
        return dataset, tokenizer, vocab

    @property
    def _cache_dir_name(self):
        return f'max_length-{self.max_length}-tokenizer_type-{self.tokenizer_type}-vocab_min_freq-{self.vocab_min_freq}-append_bos-{self.append_bos}-append_eos-{self.append_eos}'

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """ The train dataloader """
        return self._data_loader(self.dataset_train, shuffle=self.shuffle)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        return self._data_loader(self.dataset_val)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader """
        return self._data_loader(self.dataset_test)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )
