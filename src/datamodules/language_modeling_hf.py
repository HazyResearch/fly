# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py
from itertools import chain
from pathlib import Path
import pickle
from typing import Any, List, Union

from torch.utils.data.dataloader import DataLoader, Dataset
from transformers import AutoTokenizer, default_data_collator
from datasets import load_dataset, DatasetDict

from pytorch_lightning import LightningDataModule

from src.utils.utils import get_logger
logger = get_logger()


class LMDataModule(LightningDataModule):
    def __init__(self, dataset_name, tokenizer_name, dataset_config_name=None, block_size=1024,
                 cache_dir=None, batch_size=32, num_workers=1, shuffle=False, pin_memory=False,
                 drop_last=False):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.tokenizer_name = tokenizer_name
        self.cache_dir = None if cache_dir is None else Path(cache_dir).expanduser()
        self.block_size = block_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def prepare_data(self):
        if self.cache_dir is None:  # Just download the dataset
            load_dataset(self.dataset_name)
        else:  # Process the dataset and save it
            self.process_dataset()

    def setup(self, stage=None):
        if stage == 'test' and hasattr(self, 'dataset_test'):
            return
        dataset, self.tokenizer = self.process_dataset()
        self.vocab_size = len(self.tokenizer)
        # dataset.set_format(type='torch', columns=['input_ids'])
        # Create all splits
        self.dataset_train, self.dataset_val, self.dataset_test = (
            dataset['train'], dataset['validation'], dataset['test']
        )
        # Data collator will default to DataCollatorWithPadding, so we change it.
        self.collate_fn = default_data_collator

    def process_dataset(self):
        cache_dir = None if self.cache_dir is None else self.cache_dir / self._cache_dir_name
        if cache_dir is not None:
            if cache_dir.is_dir():
                return self._load_from_cache(cache_dir)

        raw_datasets = load_dataset(self.dataset_name, self.dataset_config_name)
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)

        # Preprocessing the datasets.
        # First we tokenize all the texts.
        column_names = raw_datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]
        tokenize = lambda example: tokenizer(example[text_column_name])
        tokenized_datasets = raw_datasets.map(
            tokenize,
            batched=True,
            num_proc=max(self.num_workers, 1),
            remove_columns=column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )

        # concat_text = list(chain(*tokenized_datasets['train']['input_ids']))

        if self.block_size is None:
            block_size = tokenizer.model_max_length
            if block_size > 1024:
                logger.warning(
                    f"The tokenizer picked seems to have a very large `model_max_length` "
                    f"({tokenizer.model_max_length}). "
                    "Picking 1024 instead. You can change that default value by passing dataset.block_size=x."
                )
            block_size = 1024
        else:
            if self.block_size > tokenizer.model_max_length:
                logger.warning(
                    f"The block_size passed ({self.block_size}) is larger than the maximum length for the model"
                    f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
                )
            block_size = min(self.block_size, tokenizer.model_max_length)

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
        # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
        # to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=max(self.num_workers, 1),
            load_from_cache_file=False,
            desc=f"Grouping texts in chunks of {block_size}",
        )
        if cache_dir is not None:
            self._save_to_cache(lm_datasets, tokenizer, cache_dir)
        return lm_datasets, tokenizer

    def _save_to_cache(self, dataset, tokenizer, cache_dir):
        cache_dir = self.cache_dir / self._cache_dir_name
        logger.info(f'Saving to cache at {str(cache_dir)}')
        dataset.save_to_disk(str(cache_dir))
        with open(cache_dir / 'tokenizer.pkl', 'wb') as f:
            pickle.dump(tokenizer, f)

    def _load_from_cache(self, cache_dir):
        assert cache_dir.is_dir()
        logger.info(f'Load from cache at {str(cache_dir)}')
        dataset = DatasetDict.load_from_disk(str(cache_dir))
        with open(cache_dir / 'tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        return dataset, tokenizer

    @property
    def _cache_dir_name(self):
        return f'block_size-{self.block_size}-tokenizer_name-{self.tokenizer_name}'

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
