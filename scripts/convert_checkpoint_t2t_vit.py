"""Convert T2T-ViT checkpoints to be compatible with our rewrite
"""
import re
import sys
import shutil
from pathlib import Path

import numpy as np
import torch


def main():
    for file_name in sys.argv[1:]:
        path = Path(file_name).expanduser()
        if not str(path).endswith('.og'):  # Back up original checkpoint
            path_og = Path(str(path) + '.og')
            shutil.copy2(path, path_og)
        state_dict = torch.load(path, map_location='cpu')
        # T2T-ViT checkpoint is nested in the key 'state_dict_ema'
        if state_dict.keys() == {'state_dict_ema'}:
            state_dict = state_dict['state_dict_ema']

        # Replace the names of some of the submodules
        def key_mapping(key):
            if key == 'pos_embed':
                return 'pos_embed.pe'
            elif key.startswith('tokens_to_token.'):
                return re.sub('^tokens_to_token.', 'patch_embed.', key)
            else:
                return key

        state_dict = {key_mapping(k): v for k, v in state_dict.items()}
        torch.save(state_dict, path)

if __name__ == '__main__':
    main()
