import pytest

import torch

from timm.scheduler import CosineLRScheduler

from src.optim.timm_lr_scheduler import TimmCosineLRScheduler


def test_lr():
    n_epochs = 310
    model = torch.nn.Linear(3, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.03)

    kwargs = dict(t_initial=300, lr_min=1e-5, decay_rate=0.1, warmup_lr_init=1e-6, warmup_t=10,
                  cycle_limit=1)
    scheduler_timm = CosineLRScheduler(optimizer, **kwargs)
    scheduler_timm.step(epoch=0)

    lrs_timm = []
    for epoch in range(n_epochs):
        lrs_timm.append(optimizer.param_groups[0]['lr'])
        scheduler_timm.step(epoch = epoch + 1)
    lrs_timm = torch.tensor(lrs_timm)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.03)
    scheduler_mine = TimmCosineLRScheduler(optimizer, **kwargs)
    lrs_mine = []
    for epoch in range(n_epochs):
        lrs_mine.append(optimizer.param_groups[0]['lr'])
        scheduler_mine.step()
    lrs_mine = torch.tensor(lrs_mine)

    assert torch.allclose(lrs_timm, lrs_mine, atol=1e-7, rtol=1e-5)
