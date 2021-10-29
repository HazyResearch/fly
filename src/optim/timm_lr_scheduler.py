import torch
from torch.optim import Optimizer

from timm.scheduler import CosineLRScheduler


class TimmCosineLRScheduler(CosineLRScheduler):
    """ Wrap timm.scheduler.CosineLRScheduler so we can call scheduler.step() without passing in epoch.
    It supports resuming as well.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_epoch = -1
        self.step(epoch=0)

    def step(self, epoch=None):
        if epoch is None:
            self._last_epoch += 1
        else:
            self._last_epoch = epoch
        super().step(epoch=self._last_epoch)
