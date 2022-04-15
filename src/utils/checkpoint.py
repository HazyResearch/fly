import re
from pathlib import Path

import torch


def load_checkpoint(path, device='cpu'):
    path = Path(path).expanduser()
    is_deepspeed = False
    if path.is_dir():  # DeepSpeed checkpoint
        is_deepspeed = True
        latest_path = path / 'latest'
        if latest_path.is_file():
            with open(latest_path, 'r') as fd:
                tag = fd.read().strip()
        else:
            raise ValueError(f"Unable to find 'latest' file at {latest_path}")
        path /= f'{tag}/mp_rank_00_model_states.pt'
    state_dict = torch.load(path, map_location=device)
    if is_deepspeed:
        state_dict = state_dict['module']

        # Replace the names of some of the submodules
        def key_mapping(key):
            return re.sub(r'^module.model.', '', key)

        state_dict = {key_mapping(k): v for k, v in state_dict.items()}
    return state_dict


def blockdiag_to_dense_mlp_bert(state_dict):
    from src.ops.blockdiag_multiply import blockdiag_weight_to_dense_weight
    names = {name for name in state_dict
             if re.match('bert.encoder.layer.(\d+).(mlp.fc(1|2)|(intermediate|output).dense).weight',
                         name)}
    for name in names:
        state_dict[name] = blockdiag_weight_to_dense_weight(state_dict[name])
    return state_dict
