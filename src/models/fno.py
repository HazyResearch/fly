# Copied from https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_1d.py
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from src.models.layers.spectral_conv import SpectralConv1d, SpectralConv2d


class FourierOperator1d(nn.Module):

    def __init__(self, modes, width):
        super().__init__()
        self.modes = modes
        self.width = width
        self.conv = SpectralConv1d(self.width, self.width, self.modes)
        self.w = nn.Conv1d(self.width, self.width, 1)

    def forward(self, x):
        return F.gelu(self.conv(x) + self.w(x))


class FNO1d(nn.Module):
    def __init__(self, modes, width, nlayers=4, padding=0):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.nlayers = nlayers
        self.padding = padding  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2, self.width)  # input channel is 2: (a(x), x)

        self.layers = nn.Sequential(*[FourierOperator1d(self.modes1, self.width)
                                      for _ in range(self.nlayers)])

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.stack([x, grid], dim=-1)
        x = self.fc0(x)
        x = rearrange(x, 'b x c -> b c x')
        if self.padding != 0:
            x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        # FNO code doesn't apply activation on the last block, but we do for code's simplicity.
        # Performance seems about the same.
        x = self.layers(x)

        if self.padding != 0:
            x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = rearrange(x, 'b c x -> b x c')
        x = self.fc2(F.gelu(self.fc1(x)))
        return rearrange(x, 'b x 1 -> b x')

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        return repeat(torch.linspace(0, 1, size_x, dtype=torch.float, device=device),
                      'x -> b x', b=batchsize)


class FourierOperator2d(nn.Module):

    def __init__(self, modes1, modes2, width):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.conv = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w = nn.Conv2d(self.width, self.width, 1)

    def forward(self, x):
        return F.gelu(self.conv(x) + self.w(x))


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, nlayers=4, padding=0):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.nlayers = nlayers
        self.padding = padding  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(3, self.width)  # input channel is 3: (a(x, y), x, y)

        self.layers = nn.Sequential(*[FourierOperator2d(self.modes1, self.modes2, self.width)
                                      for _ in range(self.nlayers)])

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((rearrange(x, 'b x y -> b x y 1'), grid), dim=-1)
        x = self.fc0(x)
        x = rearrange(x, 'b x y c -> b c x y')
        if self.padding != 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        # FNO code doesn't apply activation on the last block, but we do for code's simplicity.
        # Performance seems about the same
        x = self.layers(x)

        if self.padding != 0:
            x = x[..., :-self.padding, :-self.padding]
        x = rearrange(x, 'b c x y -> b x y c')
        x = self.fc2(F.gelu(self.fc1(x)))
        return rearrange(x, 'b x y 1 -> b x y')

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = repeat(torch.linspace(0, 1, size_x, dtype=torch.float, device=device),
                       'x -> b x y', b=batchsize, y=size_y)
        gridy = repeat(torch.linspace(0, 1, size_y, dtype=torch.float, device=device),
                       'y -> b x y', b=batchsize, x=size_x)
        return torch.stack([gridx, gridy], dim=-1)
