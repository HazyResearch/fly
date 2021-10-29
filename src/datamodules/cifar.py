from pathlib import Path
current_dir = Path(__file__).parent.absolute()

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

# [2021-06-30] TD: Somehow I get segfault if I import pl_bolts *after* torchvision
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from torchvision import transforms, datasets

from src.utils.utils import get_logger


def cifar10_grayscale_normalization():
    return transforms.Normalize(mean=122.6 / 255.0, std=61.0 / 255.0)


def cifar100_normalization():
    return transforms.Normalize(
        mean=[x / 255.0 for x in [129.3, 124.1, 112.4]],
        std=[x / 255.0 for x in [68.2, 65.4, 70.4]],
    )

def cifar100_grayscale_normalization():
    return transforms.Normalize(mean=124.3 / 255.0, std=63.9 / 255.0)


# Adapted from https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/datamodules/cifar10_datamodule.py
class CIFAR10(CIFAR10DataModule):

    def __init__(self, data_dir=current_dir, sequential=False, grayscale=False,
                 data_augmentation=None, resize=None, to_int=False, **kwargs):
        super().__init__(data_dir, **kwargs)
        self.data_augmentation = data_augmentation
        self.grayscale = grayscale
        self.sequential = sequential
        self.to_int = to_int
        self.resize = resize
        logger = get_logger()
        logger.info(f'Datamodule {self.__class__}: normalize={self.normalize}')
        if to_int:
            assert not self.normalize, 'to_int option is not compatible with normalize option'

        assert data_augmentation in [None, 'standard', 'autoaugment']
        if data_augmentation is not None:
            if resize is not None:
                augment_list = [transforms.Resize([resize[0], resize[1]]),
                                 transforms.RandomCrop(resize[0], padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 ]
            else:
                augment_list = [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                ]
            if data_augmentation == 'autoaugment':
                from src.utils.autoaug import CIFAR10Policy
                augment_list += [CIFAR10Policy()]
            # By default it only converts to Tensor and normalizes
            self.train_transforms = transforms.Compose(augment_list
                                                       + self.default_transforms().transforms)
        if not sequential:
            if not grayscale:
                if resize is not None:
                    self.dims = (3, resize[0], resize[1])
                else:
                    self.dims = (3, 32, 32)
            else:
                if resize is not None:
                    self.dims = (1, resize[0], resize[1])
                else:
                    self.dims = (1, 32, 32)
        else:
            if not grayscale:
                self.dims = (1024, 3)
            else:
                self.dims = (1024, 1) if not to_int else (1024,)
        if to_int and grayscale:
            self.vocab_size = 256

    def default_transforms(self):
        transform_list = [] if not self.grayscale else [transforms.Grayscale()]
        transform_list.append(transforms.ToTensor())
        if self.normalize:
            transform_list.append(self.normalize_fn())
        if self.to_int:
            transform_list.append(transforms.Lambda(lambda x: (x * 255).long()))
        if self.sequential:
            # If grayscale and to_int, it makes more sense to get rid of the channel dimension
            transform_list.append(Rearrange('1 h w -> (h w)') if self.grayscale and self.to_int
                                  else Rearrange('c h w -> (h w) c'))
        if self.resize is not None:
            transform_list.append(transforms.Resize([self.resize[0], self.resize[1]]))
        return transforms.Compose(transform_list)

    def normalize_fn(self):
        return cifar10_normalization() if not self.grayscale else cifar10_grayscale_normalization()


class CIFAR100(CIFAR10):

    dataset_cls = datasets.CIFAR100

    @property
    def num_classes(self):
        return 100

    def normalize_fn(self):
        return (cifar100_normalization() if not self.grayscale
                else cifar100_grayscale_normalization())
