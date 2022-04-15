import torch
import torch.nn as nn

from einops import rearrange


class RelativeL2(nn.Module):

    def forward(self, x, y):
        x = rearrange(x, 'b ... -> b (...)')
        y = rearrange(y, 'b ... -> b (...)')
        diff_norms = torch.linalg.norm(x - y, ord=2, dim=-1)
        y_norms = torch.linalg.norm(y, ord=2, dim=-1)
        return (diff_norms / y_norms).mean()
