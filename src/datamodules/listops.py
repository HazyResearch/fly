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


# LRA tokenizer renames ']' to 'X' and delete parentheses as their tokenizer removes
# non-alphanumeric characters.
# https://github.com/google-research/long-range-arena/blob/264227cbf9591e39dd596d2dc935297a2070bdfe/lra_benchmarks/listops/input_pipeline.py#L46
def listops_tokenizer(s):
    return s.translate({ord(']'): ord('X'), ord('('): None, ord(')'): None}).split()


class ListOps(LightningDataModule):

    num_classes = 10

    def __init__(self, data_dir=current_dir, cache_dir=None, max_length=2000, append_bos=False,
                 append_eos=False, batch_size=32, num_workers=1, shuffle=False, pin_memory=False,
                 drop_last=False, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = Path(data_dir).expanduser()
        self.cache_dir = None if cache_dir is None else Path(cache_dir).expanduser()
        self.max_length = max_length
        self.append_bos = append_bos
        self.append_eos = append_eos
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def prepare_data(self):
        if self.cache_dir is None:
            for split in ['train', 'val', 'test']:
                split_path = self.data_dir / f'basic_{split}.tsv'
                if not split_path.is_file():
                    raise FileNotFoundError(f"""
                    File {str(split_path)} not found.
                    To get the dataset, download lra_release.gz from
                    https://github.com/google-research/long-range-arena,
                    then unzip it with tar -xvf lra_release.gz.
                    Then point data_dir to the listops-1000 directory.
                    """)
        else:  # Process the dataset and save it
            self.process_dataset()

    def setup(self, stage=None):
        if stage == 'test' and hasattr(self, 'dataset_test'):
            return
        dataset, self.tokenizer, self.vocab = self.process_dataset()
        self.vocab_size = len(self.vocab)
        dataset.set_format(type='torch', columns=['input_ids', 'Target'])
        self.dataset_train, self.dataset_val, self.dataset_test = (
            dataset['train'], dataset['val'], dataset['test']
        )

        def collate_batch(batch):
            xs, ys = zip(*[(data['input_ids'], data['Target']) for data in batch])
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

        dataset = load_dataset('csv',
                               data_files={'train': str(self.data_dir / 'basic_train.tsv'),
                                           'val': str(self.data_dir / 'basic_val.tsv'),
                                           'test': str(self.data_dir / 'basic_test.tsv')},
                               delimiter='\t',
                               keep_in_memory=True)

        tokenizer = listops_tokenizer
        # Account for <bos> and <eos> tokens
        max_length = self.max_length - int(self.append_bos) - int(self.append_eos)
        tokenize = lambda example: {'tokens': tokenizer(example['Source'])[:max_length]}
        dataset = dataset.map(tokenize, remove_columns=['Source'], keep_in_memory=True,
                              load_from_cache_file=False, num_proc=max(self.num_workers, 1))
        vocab = torchtext.vocab.build_vocab_from_iterator(
            dataset['train']['tokens'],
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
        return f'max_length-{self.max_length}-append_bos-{self.append_bos}-append_eos-{self.append_eos}'

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
