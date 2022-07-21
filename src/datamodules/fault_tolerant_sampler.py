# Adapted from https://github.com/Lightning-AI/lightning/blob/2845e7565dbe6b765ae32870e7d2bc456529c30a/tests/tests_pytorch/utilities/test_auto_restart.py#L1397
from typing import Iterator

import torch
from torch.utils.data import RandomSampler, DistributedSampler


class RandomFaultTolerantSampler(RandomSampler):

    def __init__(self, *args, generator=None, **kwargs):
        # generator = torch.Generator().manual_seed(seed)
        # super().__init__(*args, generator=generator, **kwargs)
        # TD [2022-07-17]: We don't force the seed to be zero. We generate random seed,
        # which should be reproducible if pl.seed_everything was called before hand.
        # This means that changing the seed of the experiment will also change the
        # sampling order.
        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator().manual_seed(seed)
        super().__init__(*args, generator=generator, **kwargs)
        self.counter = 0
        self.restarting = False

    def state_dict(self):
        return {"random_state": self.state, "counter": self.counter}

    def load_state_dict(self, state_dict):
        self.generator.set_state(state_dict.get("random_state"))
        self.counter = state_dict["counter"]
        self.restarting = True

    def __len__(self):
        return len(self.data_source) - self.counter

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)

        self.state = self.generator.get_state()
        indices = torch.randperm(n, generator=self.generator).tolist()

        if not self.restarting:
            self.counter = 0
        else:
            indices = indices[self.counter:]
            self.restarting = False

        for index in indices:
            self.counter += 1
            yield index

        self.counter = 0


class FaultTolerantDistributedSampler(DistributedSampler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0
        self.restarting = False

    def state_dict(self):
        return {"epoch": self.epoch, "counter": self.counter}

    def load_state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]
        self.counter = state_dict["counter"]
        self.restarting = True

    def __len__(self) -> int:
        return self.num_samples - self.counter

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        if not self.restarting:
            self.counter = 0
        else:
            indices = indices[self.counter:]
            self.restarting = False

        for index in indices:
            self.counter += 1
            yield index


# class DistributedSampler(Sampler):
#     r"""Sampler that restricts data loading to a subset of the dataset.

#     It is especially useful in conjunction with
#     :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
#     process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
#     :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
#     original dataset that is exclusive to it.

#     .. note::
#         Dataset is assumed to be of constant size and that any instance of it always
#         returns the same elements in the same order.

#     Args:
#         dataset: Dataset used for sampling.
#         num_replicas (int, optional): Number of processes participating in
#             distributed training. By default, :attr:`world_size` is retrieved from the
#             current distributed group.
#         rank (int, optional): Rank of the current process within :attr:`num_replicas`.
#             By default, :attr:`rank` is retrieved from the current distributed
#             group.
#         shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
#             indices.
#         seed (int, optional): random seed used to shuffle the sampler if
#             :attr:`shuffle=True`. This number should be identical across all
#             processes in the distributed group. Default: ``0``.
#         drop_last (bool, optional): if ``True``, then the sampler will drop the
#             tail of the data to make it evenly divisible across the number of
#             replicas. If ``False``, the sampler will add extra indices to make
#             the data evenly divisible across the replicas. Default: ``False``.

#     .. warning::
#         In distributed mode, calling the :meth:`set_epoch` method at
#         the beginning of each epoch **before** creating the :class:`DataLoader` iterator
#         is necessary to make shuffling work properly across multiple epochs. Otherwise,
#         the same ordering will be always used.

#     Example::

#         >>> sampler = DistributedSampler(dataset) if is_distributed else None
#         >>> loader = DataLoader(dataset, shuffle=(sampler is None),
#         ...                     sampler=sampler)
#         >>> for epoch in range(start_epoch, n_epochs):
#         ...     if is_distributed:
#         ...         sampler.set_epoch(epoch)
#         ...     train(loader)
#     """

#     def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
#                  rank: Optional[int] = None, shuffle: bool = True,
#                  seed: int = 0, drop_last: bool = False) -> None:
#         if num_replicas is None:
#             if not dist.is_available():
#                 raise RuntimeError("Requires distributed package to be available")
#             num_replicas = dist.get_world_size()
#         if rank is None:
#             if not dist.is_available():
#                 raise RuntimeError("Requires distributed package to be available")
#             rank = dist.get_rank()
#         if rank >= num_replicas or rank < 0:
#             raise ValueError(
#                 "Invalid rank {}, rank should be in the interval"
#                 " [0, {}]".format(rank, num_replicas - 1))
#         self.dataset = dataset
#         self.num_replicas = num_replicas
#         self.rank = rank
#         self.epoch = 0
#         self.drop_last = drop_last
#         # If the dataset length is evenly divisible by # of replicas, then there
#         # is no need to drop any data, since the dataset will be split equally.
#         if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
#             # Split to nearest available length that is evenly divisible.
#             # This is to ensure each rank receives the same amount of data when
#             # using this Sampler.
#             self.num_samples = math.ceil(
#                 (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
#             )
#         else:
#             self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
#         self.total_size = self.num_samples * self.num_replicas
#         self.shuffle = shuffle
#         self.seed = seed

#     def __iter__(self) -> Iterator[T_co]:
#         if self.shuffle:
#             # deterministically shuffle based on epoch and seed
#             g = torch.Generator()
#             g.manual_seed(self.seed + self.epoch)
#             indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
#         else:
#             indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

#         if not self.drop_last:
#             # add extra samples to make it evenly divisible
#             padding_size = self.total_size - len(indices)
#             if padding_size <= len(indices):
#                 indices += indices[:padding_size]
#             else:
#                 indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
#         else:
#             # remove tail of data to make it evenly divisible.
#             indices = indices[:self.total_size]
#         assert len(indices) == self.total_size

#         # subsample
#         indices = indices[self.rank:self.total_size:self.num_replicas]
#         assert len(indices) == self.num_samples

#         return iter(indices)

#     def __len__(self) -> int:
#         return self.num_samples


# class RandomSampler(Sampler[int]):
#     r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
#     If with replacement, then user can specify :attr:`num_samples` to draw.

#     Args:
#         data_source (Dataset): dataset to sample from
#         replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
#         num_samples (int): number of samples to draw, default=`len(dataset)`.
#         generator (Generator): Generator used in sampling.
#     """
#     data_source: Sized

#     def __init__(self, data_source: Sized, num_samples: Optional[int] = None, generator=None) -> None:
#         self.data_source = data_source
#         self._num_samples = num_samples
#         self.generator = generator

#     @property
#     def num_samples(self) -> int:
#         # dataset size might change at runtime
#         if self._num_samples is None:
#             return len(self.data_source)
#         return self._num_samples

#     def __iter__(self) -> Iterator[int]:
#         n = len(self.data_source)
#         if self.generator is None:
#             seed = int(torch.empty((), dtype=torch.int64).random_().item())
#             generator = torch.Generator()
#             generator.manual_seed(seed)
#         else:
#             generator = self.generator

#         for _ in range(self.num_samples // n):
#             yield from torch.randperm(n, generator=generator).tolist()
#         yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]

#     def __len__(self) -> int:
#         return self.num_samples

# @pytest.mark.parametrize(
#     ["train_dataset_cls", "val_dataset_cls"],
#     [
#         ([RandomFaultTolerantDataset, RandomFaultTolerantDataset], [RandomFaultTolerantDataset]),
#     ],
# )
# @pytest.mark.parametrize("val_check_interval", [0.5])
# @mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "2"})
# def test_fault_tolerant_manual_mode(val_check_interval, train_dataset_cls, val_dataset_cls, tmpdir):
#     class TestModel(BoringModel):
#         def __init__(self, should_fail: bool = False):
#             super().__init__()
#             self.layer = torch.nn.Linear(1, 2)
#             self.should_fail = should_fail
#             self.batches = []

#         def training_step(self, batch, batch_idx):
#             if self.should_fail and batch_idx == 7:
#                 raise CustomException
#             self.batches.append(batch)
#             losses = []
#             for b in batch:
#                 losses.append(super().training_step(b, batch_idx)["loss"])
#             return torch.stack(losses).mean()

#         def validation_step(self, batch, batch_idx, dataloader_idx=0):
#             pass

#         validation_epoch_end = None

#         def _create_dataloader_kwargs(self, dataset_class, dataset_len, seed, num_workers):
#             dl_kwargs = {}
#             dl_kwargs["dataset"] = dataset_class(dataset_len, 1, seed=seed)
#             dl_kwargs["sampler"] = RandomFaultTolerantSampler(dl_kwargs["dataset"], seed=seed)
#             dl_kwargs["num_workers"] = num_workers
#             dl_kwargs["batch_size"] = 1
#             return dl_kwargs

#         def train_dataloader(self):
#             return [
#                 DataLoader(
#                     **self._create_dataloader_kwargs(
#                         dataset_class, 10, seed, seed + 1 if val_check_interval == 1.0 else 0
#                     )
#                 )
#                 for seed, dataset_class in enumerate(train_dataset_cls)
#             ]

#         def val_dataloader(self):
#             return [
#                 DataLoader(**self._create_dataloader_kwargs(dataset_class, 1, seed, 0))
#                 for seed, dataset_class in enumerate(val_dataset_cls)
#             ]

#         def configure_optimizers(self):
#             optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.001)
#             lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
#             return [optimizer], [lr_scheduler]

#     seed_everything(42)
#     model = TestModel()
#     trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, val_check_interval=val_check_interval)
#     trainer.fit(model)
#     total_batches = model.batches
#     total_weight = deepcopy(model.layer.weight)
#     trainer.train_dataloader = None

#     seed_everything(42)
#     model = TestModel(should_fail=True)
#     trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, val_check_interval=val_check_interval)
#     with pytest.raises(CustomException):
#         trainer.fit(model)
#     trainer.train_dataloader = None
#     failed_batches = model.batches
#     failed_weight = deepcopy(model.layer.weight)

#     checkpoint_path = str(tmpdir / ".pl_auto_save.ckpt")
#     assert os.path.exists(checkpoint_path)

#     seed_everything(42)
#     model = TestModel()
#     trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, val_check_interval=val_check_interval)
#     trainer.fit(model, ckpt_path=checkpoint_path)
#     trainer.train_dataloader = None
#     restart_batches = model.batches

#     torch_test_assert_close(total_batches, failed_batches + restart_batches)
#     assert not torch.equal(total_weight, failed_weight)
#     assert torch.equal(total_weight, model.layer.weight)
