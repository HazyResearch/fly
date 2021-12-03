from pathlib import Path
current_dir = Path(__file__).parent.absolute()

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

# [2021-06-30] TD: Somehow I get segfault if I import pl_bolts *after* torchvision
from pl_bolts.datamodules import CIFAR10DataModule
from torchvision import transforms, datasets

from src.utils.utils import get_logger
from src.utils.tuples import to_2tuple


# From https://github.com/PyTorchLightning/lightning-bolts/blob/bd392ad858039290c72c20cc3f10df39384e90b9/pl_bolts/transforms/dataset_normalizations.py#L20
def cifar10_normalization():
    return transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )


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

    default_image_size = (32, 32)

    def __init__(self, data_dir=current_dir, sequential=False, grayscale=False,
                 data_augmentation=None, image_size=32, to_int=False, **kwargs):
        super().__init__(data_dir, **kwargs)
        self.data_augmentation = data_augmentation
        self.grayscale = grayscale
        self.sequential = sequential
        self.to_int = to_int
        self.image_size = to_2tuple(image_size)
        logger = get_logger()
        logger.info(f'Datamodule {self.__class__}: normalize={self.normalize}')
        if to_int:
            assert not self.normalize, 'to_int option is not compatible with normalize option'
        self._set_augmentation()
        self.dims = self._calculate_dimensions()
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
        return transforms.Compose(transform_list)

    def normalize_fn(self):
        return cifar10_normalization() if not self.grayscale else cifar10_grayscale_normalization()

    def _set_augmentation(self, data_augmentation=None):
        assert data_augmentation in [None, 'standard', 'autoaugment']
        augment_list = []
        if self.image_size != self.default_image_size:
            augment_list.append(transforms.Resize(self.image_size))
            self.val_transforms = self.test_transforms = transforms.Compose(
                augment_list + self.default_transforms().transforms
            )
        if data_augmentation is not None:
            if data_augmentation == 'standard':
                augment_list += [
                    transforms.RandomCrop(self.image_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                ]
            elif data_augmentation == 'autoaugment':
                from src.utils.autoaug import CIFAR10Policy
                augment_list += [CIFAR10Policy()]
            # By default it only converts to Tensor and normalizes
            self.train_transforms = transforms.Compose(augment_list
                                                       + self.default_transforms().transforms)

    def _calculate_dimensions(self):
        nchannels = 3 if not self.grayscale else 1
        if not self.sequential:
            return (nchannels, self.image_size[0], self.image_size[1])
        else:
            length = self.image_size[0] * self.image_size[1]
            return (length, nchannels) if not (self.grayscale and self.to_int) else (length,)


class CIFAR100(CIFAR10):

    dataset_cls = datasets.CIFAR100

    @property
    def num_classes(self):
        return 100

    def normalize_fn(self):
        return (cifar100_normalization() if not self.grayscale
                else cifar100_grayscale_normalization())
