from typing import Any, List, Dict, Tuple, Optional, Callable, cast

import logging
import time
import pickle
from pathlib import Path, PurePath
from PIL import Image

from fs.tarfs import TarFS
from fs.zipfs import ZipFS

from torch.utils.data import get_worker_info

from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import has_file_allowed_extension


# https://www.python.org/dev/peps/pep-0616/
def removeprefix(self: str, prefix: str, /) -> str:
    if self.startswith(prefix):
        return self[len(prefix):]
    else:
        return self[:]


class ArchiveImageFolder(ImageFolder):
    """Dataset that supports Tar/Zip archives (uncompressed), with a folder per class.

    Similarly to torchvision.datasets.ImageFolder, assumes that the images inside
    the Tar archive are arranged in this way by default:

      root/dog/xxx.png
      root/dog/xxy.png
      root/dog/[...]/xxz.png

      root/cat/123.png
      root/cat/nsdf3.png
      root/cat/[...]/asd932_.png

    Args:
      archive (string or TarDataset): Path to the Tar file containing the dataset.
        Alternatively, pass in a TarDataset object to reuse its cached information;
        this is useful for loading different subsets within the same archive.
      root_in_archive (string): Root folder within the archive, directly below
        the folders with class names.
      extensions (tuple): Extensions (strings starting with a dot), only files
        with these extensions will be iterated. Default: png/jpg/jpeg.
      is_valid_file (callable): Optional function that takes file information as
        input (tarfile.TarInfo) and outputs True for files that need to be
        iterated; overrides extensions argument.
        Example: lambda m: m.isfile() and m.name.endswith('.png')
      transform (callable): Function applied to each image by __getitem__ (see
        torchvision.transforms). Default: ToTensor (convert PIL image to tensor).

    Attributes:
      samples (list): Image file names to iterate.
      targets (list): Numeric label corresponding to each image.
      class_to_idx (dict): Maps class names to numeric labels.
      idx_to_class (dict): Maps numeric labels to class names.
    """

    def __init__(
                self,
                archive: str,
                cache_dir: Optional[str] = None,
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                is_valid_file: Optional[Callable[[str], bool]] = None,
                root_in_archive: str = '',
    ) -> None:
        assert archive.endswith('.tar') or archive.endswith('.zip'), 'Only .tar and .zip are supported'
        self._fs_cls = TarFS if archive.endswith('.tar') else ZipFS
        self.root_in_archive = PurePath(root_in_archive)
        self.cache_dir = None if cache_dir is None else Path(cache_dir).expanduser()
        # open tar/zip file. in a multiprocessing setting (e.g. DataLoader workers), we
        # have to open one file handle per worker (stored as the tar_obj dict), since
        # when the multiprocessing method is 'fork', the workers share this TarDataset.
        # we want one file handle per worker because TarFile is not thread-safe.
        # As done in https://github.com/jotaf98/simple-tar-dataset/blob/master/tardataset.py
        logger = logging.getLogger(__name__)
        logger.info(f'Reading archive headers from {str(archive)} with root_in_archive {root_in_archive}')
        t = time.time()
        worker = get_worker_info()
        worker = worker.id if worker else None
        self.archive_fs = {worker: self._fs_cls(str(Path(archive).expanduser()))}
        super().__init__(archive, loader=None, transform=transform,
                         target_transform=target_transform, is_valid_file=is_valid_file)
        logger.info(f'Done in {float(time.time() - t):.1f} seconds.')

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folders in a dataset.
        See :class:`DatasetFolder` for details.
        """
        # We ignore directory and assume that directory == self.root
        if self.cache_dir is not None:
            try:
                return load_from_cache(self.cache_dir / self._cache_dir_name / 'classes.pkl')
            except FileNotFoundError:
                pass

        archive_fs = self.get_archive_fs().opendir(str(self.root_in_archive))
        classes = sorted(entry.name for entry in archive_fs.scandir('/') if entry.is_dir)
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {str(self.root_in_archive)} inside {self.root}.")
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        if self.cache_dir is not None:
            save_to_cache(self.cache_dir / self._cache_dir_name / 'classes.pkl',
                          (classes, class_to_idx))
        return classes, class_to_idx

    # Adapted from https://github.com/pytorch/vision/blob/main/torchvision/datasets/folder.py
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
        # We ignore directory and assume that directory == self.root
        if self.cache_dir is not None:
            try:
                return load_from_cache(self.cache_dir / self._cache_dir_name / 'samples.pkl')
            except FileNotFoundError:
                pass

        if class_to_idx is None:
            _, class_to_idx = self.find_classes(directory)
        elif not class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:

            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        archive_fs = self.get_archive_fs().opendir(str(self.root_in_archive))
        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir_info = archive_fs.getinfo(target_class)
            if not target_dir_info.is_dir:
                continue
            for root, _, fnames in sorted(archive_fs.walk(target_class)):
                # root starts with '/' because it's the root in this directory
                # That messes up the path joining, so we remove the '/'
                root = removeprefix(root, '/')
                for fname in sorted(fnames, key=lambda info: info.name):
                    if is_valid_file(fname.name):
                        path = self.root_in_archive / root / fname.name
                        item = str(path), class_index
                        instances.append(item)
                        if target_class not in available_classes:
                            available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {', '.join(extensions)}"
            raise FileNotFoundError(msg)

        if self.cache_dir is not None:
            save_to_cache(self.cache_dir / self._cache_dir_name / 'samples.pkl', instances)

        return instances

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        with self.get_archive_fs().openbin(path) as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    @property
    def _cache_dir_name(self):
        return f'root_in_archive-{str(self.root_in_archive)}'

    def get_archive_fs(self):
        worker = get_worker_info()
        worker = worker.id if worker else None
        if worker not in self.archive_fs:
            self.archive_fs[worker] = self._fs_cls(str(Path(self.root).expanduser()))
        return self.archive_fs[worker]

    def __del__(self):
        """Close the TarFile file handles on exit."""
        for o in self.archive_fs.values():
            o.close()

    def __getstate__(self):
        """Serialize without the TarFile references, for multiprocessing compatibility."""
        state = dict(self.__dict__)
        state['archive_fs'] = {}
        return state


def save_to_cache(path, obj):
    path = Path(path)
    logger = logging.getLogger(__name__)
    logger.info(f'Saving to cache at {str(path)}')
    path.parent.mkdir(exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_from_cache(path):
    path = Path(path)
    if not path.is_file():
      raise FileNotFoundError(f'File {str(path)} not found')
    logger = logging.getLogger(__name__)
    logger.info(f'Load from cache at {str(path)}')
    with open(path, 'rb') as f:
        return pickle.load(f)
