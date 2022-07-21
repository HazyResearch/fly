import torch
from torch.optim import Optimizer


def InvSqrt(optimizer: Optimizer, num_warmup_steps: int):
    """ Originally used for Transformer (in Attention is all you need)
    We use the formula from the original paper.
    Refer to other implementations:
    - Nvidia: https://github.com/NVIDIA/DeepLearningExamples/blob/233287038c96734bf5c94a3adf5f3d08f54838d8/PyTorch/LanguageModeling/Transformer-XL/pytorch/train.py#L915
    - LRA: https://github.com/google-research/long-range-arena/blob/264227cbf9591e39dd596d2dc935297a2070bdfe/lra_benchmarks/utils/train_utils.py#L87
    Note that the max learning rate is then original_lr / num_warmup_steps ** 0.5,
    *not* original_lr.
    Fairseq has a different implementation where the max learning rate is original_lr (I think):
    https://github.com/pytorch/fairseq/blob/master/fairseq/optim/lr_scheduler/inverse_square_root_schedule.py
    """

    def lr_lambda(current_step):
        # return a multiplier instead of a learning rate
        if current_step == 0 and num_warmup_steps == 0:
            return 1.
        else:
            return (1. / (current_step ** 0.5) if current_step > num_warmup_steps
                    else current_step / (num_warmup_steps ** 1.5))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
