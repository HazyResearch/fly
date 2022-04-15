import torch
from torch import Tensor

from torchmetrics import Metric, Accuracy


class AccuracyMine(Accuracy):
    """Wrap torchmetrics.Accuracy to take argmax of y in case of Mixup.
    """
    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        super().update(preds, target.argmax(dim=-1) if target.is_floating_point() else target)


# TD [2022-02-10] torchmetrics.Accuracy doesn't work with negative ignore_index yet
# https://github.com/PyTorchLightning/metrics/pull/362
class AccuracyIgnoreIndex(Metric):

    def __init__(self, ignore_index=None, compute_on_step=True, dist_sync_on_step=False,
                 process_group=None, dist_sync_fn=None):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.ignore_index = ignore_index
        # Total number of sequences in all batches
        self.add_state('total', default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx='sum')
        # Total number of correct predictions
        self.add_state('correct', default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if preds.is_floating_point():
            preds = preds.argmax(dim=-1)
        matched = (preds == target)
        if self.ignore_index is not None:
            matched = matched[target != self.ignore_index]
        self.total += matched.numel()
        self.correct += matched.count_nonzero()

    def compute(self):
        """
        Returns perplexity across all workers and resets to 0 :attr:`nll_sum` and :attr:`total`.
        """
        if self.total.eq(0):
            return None
        return self.correct / self.total
