from typing import Any, List, Dict, Tuple, Union, Optional, Callable, cast

from pathlib import Path, PurePath
from PIL import Image

from fs.tarfs import TarFS

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split, get_worker_info

from einops.layers.torch import Rearrange, Reduce

# [2021-08-19] TD: Somehow I get segfault if I import pytorch_lightning *after* torchvision
from pytorch_lightning import LightningDataModule

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import has_file_allowed_extension


# There's an empty file in the dataset
PATHFINDER_BLACKLIST = {'pathfinder32/curv_baseline/imgs/0/sample_172.png'}


def pil_loader_grayscale(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        return Image.open(f).convert('L')


class PathFinderDataset(ImageFolder):
    """Path Finder dataset."""

    def __init__(
                self,
                root: str,
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                is_valid_file: Optional[Callable[[str], bool]] = None,
        ) -> None:
            super().__init__(root, loader=pil_loader_grayscale, transform=transform,
                             target_transform=target_transform, is_valid_file=is_valid_file)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Override this so it doesn't call the parent's method
        """
        return [], {}

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).
        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.
        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.
        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.
        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        # We ignore class_to_idx
        directory = Path(directory).expanduser()

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:

            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        path_list = sorted(list((directory / 'metadata').glob('*.npy')), key=lambda path: int(path.stem))
        if not path_list:
          raise FileNotFoundError(f'No metadata found at {str(directory)}')
        # Get the 'pathfinder32/curv_baseline part of data_dir
        data_dir_stem = Path().joinpath(*directory.parts[-2:])
        instances = []
        for metadata_file in path_list:
            with open(metadata_file, 'r') as f:
                for metadata in f.read().splitlines():
                    metadata = metadata.split()
                    image_path = Path(metadata[0]) / metadata[1]
                    if (is_valid_file(str(image_path))
                        and str(data_dir_stem / image_path) not in PATHFINDER_BLACKLIST):
                        label = int(metadata[3])
                        instances.append((str(directory / image_path), label))
        return instances


class PathFinderTarDataset(PathFinderDataset):
    """Path Finder dataset."""

    def __init__(
                self,
                archive: str,
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                is_valid_file: Optional[Callable[[str], bool]] = None,
                root_in_archive: str = '',
        ) -> None:
            self.root_in_archive = PurePath(root_in_archive)
            # open tar file. in a multiprocessing setting (e.g. DataLoader workers), we
            # have to open one file handle per worker (stored as the tar_obj dict), since
            # when the multiprocessing method is 'fork', the workers share this TarDataset.
            # we want one file handle per worker because TarFile is not thread-safe.
            # As done in https://github.com/jotaf98/simple-tar-dataset/blob/master/tardataset.py
            worker = get_worker_info()
            worker = worker.id if worker else None
            self.tar_fs = {worker: TarFS(str(Path(archive).expanduser()))}
            ImageFolder.__init__(self, archive, loader=None, transform=transform,
                                 target_transform=target_transform, is_valid_file=is_valid_file)

    def make_dataset(
        self,
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).
        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.
        Args:
            directory (str): archive dataset directory, corresponding to ``self.archive``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.
        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.
        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        # We ignore directory and class_to_idx
        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:

            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        metadata_fs = self.get_tar_fs().opendir(str(self.root_in_archive / 'metadata'))
        path_list = sorted(list(metadata_fs.filterdir('/', files=['*.npy'])),
                           key=lambda path_info: int(path_info.stem))
        if not path_list:
            raise FileNotFoundError(f'No metadata found in {str(self.root)}')

        # Get the 'pathfinder32/curv_baseline part of data_dir
        data_dir_stem = PurePath().joinpath(*self.root_in_archive.parts[-2:])
        instances = []
        for metadata_file in path_list:
            for metadata in metadata_fs.readtext(metadata_file.name).splitlines():
                  metadata = metadata.split()
                  image_path = PurePath(metadata[0]) / metadata[1]
                  if (is_valid_file(str(image_path))
                      and str(data_dir_stem / image_path) not in PATHFINDER_BLACKLIST):
                      label = int(metadata[3])
                      instances.append((str(self.root_in_archive / image_path), label))
        return instances

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        with self.get_tar_fs().openbin(path) as f:
            sample = Image.open(f).convert('L')  # Open in grayscale
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def get_tar_fs(self):
        worker = get_worker_info()
        worker = worker.id if worker else None
        if worker not in self.tar_fs:
            self.tar_fs[worker] = TarFS(str(Path(self.root).expanduser()))
        return self.tar_fs[worker]

    def __del__(self):
        """Close the TarFile file handles on exit."""
        for o in self.tar_fs.values():
            o.close()

    def __getstate__(self):
        """Serialize without the TarFile references, for multiprocessing compatibility."""
        state = dict(self.__dict__)
        state['tar_fs'] = {}
        return state


class PathFinder(LightningDataModule):

    num_classes = 2

    def __init__(self, data_dir, resolution, level, sequential=False, to_int=False, pool=1, val_split=0.1,
                 test_split=0.1, batch_size=32, num_workers=1, seed=42, shuffle=False,
                 pin_memory=False, drop_last=False, **kwargs):
        """If data_dir points to a tar file (e.g., pathfinder/pathfinder.tar), we support reading
        directly from that tar file without extraction.
        That tar file should have the same structure as the pathfinder dir: e.g., it should contain
        pathfinder32/curv_contour_length_14 in the archive.
        """
        super().__init__(**kwargs)
        assert resolution in [32, 64, 128, 256]
        self.resolution = resolution
        assert level in ['easy', 'intermediate', 'hard']
        self.level = level
        level_dir = {'easy': 'curv_baseline', 'intermediate': 'curv_contour_length_9',
                     'hard': 'curv_contour_length_14'}[level]
        self.prefix_dir = Path(f'pathfinder{resolution}') / level_dir
        self.data_dir = Path(data_dir).expanduser()
        self.use_tar_dataset = self.data_dir.suffix == '.tar'
        self.sequential = sequential
        self.to_int = to_int
        self.pool = pool
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        if not sequential:
            self.dims = (1, resolution, resolution)
        else:
            self.dims = (resolution * resolution, 1) if not to_int else (resolution * resolution,)
        if to_int:
            self.vocab_size = 256

    def default_transforms(self):
        transform_list = [transforms.ToTensor()]
        if self.pool > 1:
            transform_list.append(Reduce('1 (h h2) (w w2) -> 1 h w', 'mean', h2=self.pool, w2=self.pool))
        if self.to_int:
            transform_list.append(transforms.Lambda(lambda x: (x * 255).long()))
        if self.sequential:
            # If to_int, it makes more sense to get rid of the channel dimension
            transform_list.append(Rearrange('1 h w -> (h w)') if self.to_int
                                  else Rearrange('1 h w -> (h w) 1'))
        return transforms.Compose(transform_list)

    def prepare_data(self):
        if self.use_tar_dataset:
            if not self.data_dir.is_file():
                raise FileNotFoundError(f"""
                Tar file {str(self.data_dir)} not found.
                To get the dataset, download lra_release.gz from
                https://github.com/google-research/long-range-arena,
                then unzip it with tar -xvf lra_release.gz.
                Then compress the pathfinderX (X=32, 64, 128, 256) directory into a tar file:
                tar -cvf pathfinder32.tar pathfinder32
                Then point data_dir to the pathfinder32.tar file.
                """)
        else:
            if not (self.data_dir / self.prefix_dir).is_dir():
                raise FileNotFoundError(f"""
                Directory {str(self.data_dir / self.prefix_dir)} not found.
                To get the dataset, download lra_release.gz from
                https://github.com/google-research/long-range-arena,
                then unzip it with tar -xvf lra_release.gz.
                Then point data_dir to the directory that contains pathfinderX, where X is the
                resolution (either 32, 64, 128, or 256).
                """)

    def setup(self, stage=None):
        if stage == 'test' and hasattr(self, 'dataset_test'):
            return
        # [2021-08-18] TD: I ran into RuntimeError: Too many open files.
        # https://github.com/pytorch/pytorch/issues/11201
        torch.multiprocessing.set_sharing_strategy('file_system')
        if self.use_tar_dataset:
            dataset = PathFinderTarDataset(str(self.data_dir), root_in_archive=str(self.prefix_dir),
                                           transform=self.default_transforms())
        else:
            dataset = PathFinderDataset(self.data_dir / self.prefix_dir,
                                        transform=self.default_transforms())
        len_dataset = len(dataset)
        val_len = int(self.val_split * len_dataset)
        test_len = int(self.test_split * len_dataset)
        train_len = len_dataset - val_len - test_len
        self.dataset_train, self.dataset_val, self.dataset_test = random_split(
            dataset, [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(self.seed)
        )

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
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                drop_last=self.drop_last,
                pin_memory=self.pin_memory
            )
