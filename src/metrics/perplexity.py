# Inspired by https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/metrics/sequence_perplexity.py
# But we compute the perplexity correctly: exp(average(nll)), not average(exp(nll))

import torch
import torch.nn.functional as F
from torchmetrics import Metric

__all__ = ['Perplexity']


class Perplexity(Metric):
    """
    This class computes mean perplexity across the batches of sequences.
    You have to provide ``logits`` (float tensor of shape [batch_size x seq_length x vocab_size]) and
    ``labels`` (int tensor of shape [batch_size x seq_length] with values from the range [0, vocab_size-1])
    to the :meth:`update` method. If some of the sequences are shorter than seq_length, you can also provide
    an optional argument ``mask`` (bool tensor of shape [batch_size x seq_length]) which masks out tokens
    not participating in perplexity computation.
    See :doc:`PyTorch Lightning Metrics<pytorch-lightning:metrics>` for the metric usage instructions.
    Args:
        compute_on_step:
            Forward only calls ``update()`` and returns ``None`` if this is set to ``False``. default: ``True``
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()`` before returning the value at the step.
        process_group:
            Specify the process group on which synchronization is called. default: ``None`` (which selects the entire
                world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When ``None``, DDP will be used
                to perform the allgather.
    """

    def __init__(self, compute_on_step=True, dist_sync_on_step=False, process_group=None, dist_sync_fn=None):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        # Total sum of exponentiated average negative log likelihoods
        self.add_state('nll_mean', default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx='mean')
        # Total number of sequences in all batches
        self.add_state('numel', default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx='sum')

    def update(self, logits: torch.Tensor, labels: torch.Tensor, mask=None):
        # if mask is None:
        #     mask = torch.ones_like(labels)
        # mask = mask.to(logits.dtype)

        # log_probs = torch.log_softmax(logits, dim=-1)
        # target_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        # nll = -(target_log_probs * mask).sum()
        # self.numel += mask.sum().long()
        # self.nll_sum += nll
        # TODO: ignoring mask rn
        current_sum = self.nll_mean.double() * self.numel
        self.numel += labels.numel()
        loss = F.cross_entropy(logits, labels)
        self.nll_mean = (current_sum + loss.double() * labels.numel()) / self.numel

    def compute(self):
        """
        Returns perplexity across all workers and resets to 0 :attr:`nll_sum` and :attr:`numel`.
        """
        if self.numel.eq(0):
            return None
        # return (self.nll_sum / self.numel).exp()
        return self.nll_mean.exp()
