# Adapted from https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/Transformer-XL/pytorch/data_utils.py
# https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/data_utils.py
# https://github.com/pytorch/examples/blob/master/word_language_model/main.py
# https://github.com/HazyResearch/hippo/blob/master/dataloaders/lm.py

import subprocess
from pathlib import Path
current_dir = Path(__file__).parent.absolute()

import numpy as np

import torch

from pytorch_lightning import LightningDataModule

from src.datamodules.datasets.vocabulary import OpenAIVocab, Vocab
from src.utils.distributed import sync_workers
from src.utils.utils import get_logger

logger = get_logger()


class LMOrderedIterator(object):
    def __init__(self, data, bsz, bptt, device='cpu', mem_len=None, ext_len=None, warmup=True,
                 roll_seed=None,  # roll data based on seed
                 batch_first=False,
                 shard_id=0, num_shards=1,  # For distributed training
    ):
        """
            data -- LongTensor -- the LongTensor is strictly ordered
            bsz; batch size *per shard* (i.e. per GPU)
        """
        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0
        self.mem_len = mem_len
        self.warmup = warmup
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.roll_seed = roll_seed
        self.batch_first = batch_first

        self.device = device

        total_bsz = bsz * num_shards
        # Work out how cleanly we can divide the dataset into total_bsz parts.
        n_step = data.size(0) // total_bsz

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data[:n_step * total_bsz]

        # Evenly divide the data across the bsz batches.
        self.data = data.view(total_bsz, -1).t().contiguous().pin_memory()  # (..., batch_size)

        if mem_len and warmup:
            self.warmup_batches = (mem_len + bptt - 1) // bptt
            self.warmup_elems = self.warmup_batches * bptt

            warmup_data = self.data.roll((self.warmup_elems, 1), (0, 1))[:self.warmup_elems]
            self.data = torch.cat((warmup_data, self.data))

        # Partition data for DistributedDataParallel
        self.data = self.data.chunk(num_shards, dim=1)[shard_id]

        # Number of mini-batches
        # Need to subtract 1 because target is data shifted by 1
        self.n_batch = (self.data.size(0) - 1 + self.bptt - 1) // self.bptt

        self.last_iter = None
        self.epoch = -1

    def roll(self, seed):
        rng = torch.Generator()
        rng.manual_seed(seed)
        for i in range(self.data.size(1)):
            row = self.data[:, i]
            shift = torch.randint(0, self.data.size(0), (1,), generator=rng)
            row = torch.cat((row[shift:], row[:shift]))
            self.data[:, i] = row

    def get_batch(self, i, bptt=None):
        """ Get batch starting at token index i """
        if bptt is None:
            bptt = self.bptt

        seq_len = min(bptt, self.data.size(0) - 1 - i)

        end_idx = i + seq_len
        beg_idx = max(0, i - self.ext_len)

        data = self.data[beg_idx:end_idx].to(self.device, non_blocking=True)
        target = self.data[i+1:i+1+seq_len].to(self.device, non_blocking=True)

        if self.mem_len and self.warmup:
            warm = i >= self.warmup_elems
        else:
            warm = True

        if self.batch_first:
            return data.t(), target.t(), seq_len, warm
        else:
            return data, target, seq_len, warm

    def get_fixlen_iter(self, start=0):
        if start != 0:
            start += self.bptt
        for i in range(start, self.data.size(0) - 1, self.bptt):
            self.last_iter = i
            yield self.get_batch(i)

    def get_varlen_iter(self, start=0, std=5, min_len=5, max_deviation=3):
        max_length = self.bptt + max_deviation * std
        i = start
        while True:
            bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
            bptt = min(max_length, max(min_len, int(np.random.normal(bptt, std))))
            data, target, seq_len = self.get_batch(i, bptt)
            i += seq_len
            if self.batch_first:
                yield data.t(), target.t(), seq_len
            else:
                yield data, target, seq_len
            if i >= self.data.size(0) - 2:
                break

    def __iter__(self):
        self.epoch += 1
        if self.roll_seed is not None:
            self.roll(self.roll_seed + self.epoch)
        return self.get_fixlen_iter()

    def __len__(self):
        return self.n_batch


class WikiText2(LightningDataModule):

    name = 'wt2'
    vocab_kwargs = {'special': ['<eos>'], 'lower_case': False}
    encode_kwargs = {'ordered': True}

    def __init__(self, data_dir, vocab_type='word', batch_size=32, max_length=1024,
                 val_batch_size=None, val_max_length=None, roll_seed=None, batch_first=False):
        super().__init__()
        self.data_dir = Path(data_dir).expanduser()
        if vocab_type not in ['word', 'bpe']:
            raise RuntimeError('Unsupported vocab')
        self.vocab_type = vocab_type
        self.batch_size = batch_size
        self.max_length = max_length
        self.val_batch_size = val_batch_size if val_batch_size is not None else self.batch_size
        self.val_max_length = val_max_length if val_max_length is not None else self.max_length
        self.roll_seed = roll_seed
        self.batch_first = batch_first

    def prepare_data(self):
        if not self.data_dir.is_dir():
            subprocess.run([str(current_dir / 'datasets' / 'getdata.sh'), self.name,
                            str(self.data_dir.parent.absolute())], check=True)
        if not (self.data_dir / self._cache_file_name).is_file():
            self.process_dataset()

    def setup(self, stage=None):
        if stage == 'test' and hasattr(self, 'dataset_test'):
            return
        self.vocab, self.dataset_train, self.dataset_val, self.dataset_test = self.process_dataset()

    def process_dataset(self):
        if (self.data_dir / self._cache_file_name).is_file():
            return self._load_from_cache()
        else:
            logger.info(f'Producing dataset {self.name}...')
            if self.vocab_type == 'word':
                vocab = Vocab(**self.vocab_kwargs)
            elif self.vocab_type == 'bpe':
                vocab = OpenAIVocab()
            else:
                raise RuntimeError('Unsupported vocab')
            vocab = self._vocab_count(vocab)
            vocab.build_vocab()
            train = vocab.encode_file(str(self.data_dir / 'train.txt'), **self.encode_kwargs)
            val = vocab.encode_file(str(self.data_dir / 'valid.txt'), **self.encode_kwargs)
            test = vocab.encode_file(str(self.data_dir / 'test.txt'), **self.encode_kwargs)
            self._save_to_cache((vocab, train, val, test))
            return vocab, train, val, test

    def _vocab_count(self, vocab):
        vocab.count_file(self.data_dir / 'train.txt')
        vocab.count_file(self.data_dir / 'valid.txt')
        vocab.count_file(self.data_dir / 'test.txt')
        return vocab

    def _save_to_cache(self, obj):
        cache_path = self.data_dir / self._cache_file_name
        with sync_workers() as rank:
            if rank == 0:
                try:
                    torch.save(obj, cache_path)
                    logger.info(f'Saved dataset to {cache_path}')
                except:
                    pass

    def _load_from_cache(self):
        cache_path = self.data_dir / self._cache_file_name
        if cache_path.is_file():
            logger.info(f'Loading cached dataset from {str(cache_path)}')
            return torch.load(cache_path)
        else:
            raise FileNotFoundError(f'Cache file {str(cache_path)} does not exist.')

    @property
    def _cache_file_name(self):
        return f'cache.{self.vocab_type}.pt'

    def train_dataloader(self, *args, **kwargs):
        shard_id = self.trainer.global_rank
        num_shards = self.trainer.world_size
        return LMOrderedIterator(self.dataset_train, bsz=self.batch_size, bptt=self.max_length,
                                 roll_seed=self.roll_seed, batch_first=self.batch_first,
                                 shard_id=shard_id, num_shards=num_shards)

    def val_dataloader(self, *args, **kwargs):
        shard_id = self.trainer.global_rank
        num_shards = self.trainer.world_size
        return LMOrderedIterator(self.dataset_val, bsz=self.val_batch_size,
                                 bptt=self.val_max_length, batch_first=self.batch_first,
                                 shard_id=shard_id, num_shards=num_shards)

    def test_dataloader(self, *args, **kwargs):
        shard_id = self.trainer.global_rank
        num_shards = self.trainer.world_size
        return LMOrderedIterator(self.dataset_test, bsz=self.val_batch_size,
                                 bptt=self.val_max_length, batch_first=self.batch_first,
                                 shard_id=shard_id, num_shards=num_shards)


class WikiText103(WikiText2):

    name = 'wt103'

    def _vocab_count(self, vocab):
        vocab.count_file(self.data_dir / 'train.txt')
        return vocab
