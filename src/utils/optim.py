import inspect

import torch.nn as nn

import hydra

from src.models.modules.seq_common import PositionalEncoding


def group_parameters_for_optimizer(model, optimizer_cfg, bias_weight_decay=False,
                                   normalization_weight_decay=False):
    """Set weight_decay=0.0 for parameters in model.no_weight_decay, for parameters with
    attribute _no_weight_decay==True, for bias parameters if bias_weight_decay==False, for
    normalization parameters if normalization_weight_decay==False
    """
    # Get the weight decay from the config, or from the default value of the optimizer constructor
    # if it's not specified in the config.
    if 'weight_decay' in optimizer_cfg:
        weight_decay = optimizer_cfg.weight_decay
    else:
        # https://stackoverflow.com/questions/12627118/get-a-function-arguments-default-value
        signature = inspect.signature(hydra.utils.get_class(optimizer_cfg._target_))
        if 'weight_decay' in signature.parameters:
            weight_decay = signature.parameters['weight_decay'].default
            if weight_decay is inspect.Parameter.empty:
                weight_decay = 0.0
        else:
            weight_decay = 0.0

    if weight_decay == 0.0:  # None of the parameters have weight decay anyway
        return model.parameters()

    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    else:
        skip = set()

    # Adapted from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L134
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (nn.Linear, )
    blacklist_weight_modules = (nn.Embedding, PositionalEncoding)
    if not normalization_weight_decay:
        blacklist_weight_modules += (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                     nn.LazyBatchNorm1d, nn.LazyBatchNorm2d, nn.LazyBatchNorm3d,
                                     nn.GroupNorm, nn.SyncBatchNorm,
                                     nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                                     nn.LayerNorm, nn.LocalResponseNorm)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            if not p.requires_grad:
                continue  # frozen weights
            if fpn in skip:
                no_decay.add(fpn)
            elif getattr(p, '_no_weight_decay', False):
                no_decay.add(fpn)
            elif not bias_weight_decay and pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    # special case the position embedding parameter in the root GPT module as not decayed
    if 'pos_emb' in param_dict:
        no_decay.add('pos_emb')

    decay = decay | (param_dict.keys() - no_decay)
    # validate that we considered every parameter
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"Parameters {str(inter_params)} made it into both decay/no_decay sets!"
    assert len(param_dict.keys() - union_params) == 0, f"parameters {str(param_dict.keys() - union_params)}  were not separated into either decay/no_decay set!"

    return [
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
    ]
