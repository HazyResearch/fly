# From https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_1d.py
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.nn.utils.parametrize as parametrize


# class RealToComplex(nn.Module):

#     def forward(self, x):
#         return torch.view_as_complex(x)

    # [2022-01-06] TD: Pytorch's parameterize complains if we have right_inverse,
    # since it assumes that parameterize keeps the dtype.
    # def right_inverse(self, x_cplx):
    #     return torch.view_as_real(x_cplx)



################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        # self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        # weight is stored as real to avoid issue with Adam not working on complex parameters
        # FNO code initializes with rand but we initializes with randn as that seems more natural.
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, 2))
        # Need unsafe=True since we're changing the dtype of the parameters
        # parametrize.register_parametrization(self, 'weights1', RealToComplex(), unsafe=True)

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x, norm='ortho')

        # Multiply relevant Fourier modes
        # out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        # out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        weights1 = torch.view_as_complex(self.weights1)
        out_ft = F.pad(self.compl_mul1d(x_ft[:, :, :self.modes1], weights1),
                       (0, x_ft.shape[-1] - self.modes1))

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1), norm='ortho')
        return x


################################################################
# 2d fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        # weight is stored as real to avoid issue with Adam not working on complex parameters
        # FNO code initializes with rand but we initializes with randn as that seems more natural.
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, 2))
        self.weights2 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, 2))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x, norm='ortho')

        # Multiply relevant Fourier modes
        out_ft = torch.zeros_like(x_ft)
        weights1 = torch.view_as_complex(self.weights1)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], weights1)
        weights2 = torch.view_as_complex(self.weights2)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm='ortho')
        return x
