import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch
import torch.nn.functional as F
from utils import SpectralNorm


class SpectralNormConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding, transpose, bias):
        super(SpectralNormConv, self).__init__()
        self.conv = nn.Identity()
        if transpose:
            self.conv = SpectralNorm(nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding, bias=bias))
        else:
            self.conv = SpectralNorm(nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=bias))

    def forward(self, x):
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding, transpose=False, is_bn=True, activate='relu',
                 bias=False):
        super(ConvBlock, self).__init__()
        ac = nn.Identity()
        bn = nn.BatchNorm2d(out_dim) if is_bn else nn.Identity()
        if activate == 'relu':
            ac = nn.ReLU(inplace=True)
        elif activate == 'leakyrelu':
            ac = nn.LeakyReLU(0.1, inplace=True)
        elif activate == 'tanh':
            ac = nn.Tanh()
        self.net = nn.Sequential(
            SpectralNormConv(in_dim, out_dim, kernel_size, stride, padding, transpose, bias),
            bn,
            ac,
        )
        # if transpose:
        #     self.net = nn.Sequential(
        #         spectral_norm(nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding, bias=bias)),
        #         # nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding, bias=bias),
        #         bn,
        #         ac,
        #     )
        # else:
        #     self.net = nn.Sequential(
        #         spectral_norm(nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=bias)),
        #         # nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=bias),
        #         bn,
        #         ac,
        #     )

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64):
        super(Generator, self).__init__()
        self.block1 = ConvBlock(nz, ngf * 8, 4, 1, 0, transpose=True)  # (b, 512, 4, 4)
        self.block2 = ConvBlock(ngf * 8, ngf * 4, 4, 2, 1, transpose=True)  # (b, 256, 8, 8)
        self.block3 = ConvBlock(ngf * 4, ngf * 2, 4, 2, 1, transpose=True)  # (b, 128, 16, 16)
        self.block4 = ConvBlock(ngf * 2, ngf * 1, 4, 2, 1, transpose=True)  # (b, 64, 32, 32)
        self.block5 = ConvBlock(ngf * 1, 3, 4, 2, 1, transpose=True, is_bn=False, activate='tanh')  # (b, 3, 64, 64)


    def forward(self, z):
        """
        :param z:  (b, 100, 1, 1)
        :return:
        """
        x = self.block1(z)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super(Discriminator, self).__init__()
        self.block1 = ConvBlock(3, ndf, 4, 2, 1, transpose=False, is_bn=False,
                                activate='leakyrelu')  # (b, 64, 32, 32)
        self.block2 = ConvBlock(ndf, ndf * 2, 4, 2, 1, transpose=False, is_bn=False,
                                activate='leakyrelu')  # (b, 128, 16, 16)
        self.block3 = ConvBlock(ndf * 2, ndf * 4, 4, 2, 1, transpose=False, is_bn=False,
                                activate='leakyrelu')  # (b, 256, 8, 8)
        self.block4 = ConvBlock(ndf * 4, ndf * 8, 4, 2, 1, transpose=False, is_bn=False,
                                activate='leakyrelu')  # (b, 512, 4, 4)
        self.block5 = nn.Conv2d(ndf * 8, 1, 4, 1, 0)  # (b, 1, 1, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x.squeeze()
