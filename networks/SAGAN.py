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


class Attn(nn.Module):
    def __init__(self, in_dim):
        super(Attn, self).__init__()
        self.q_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.k_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.v_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.sf = nn.Softmax(dim=-1)

    def forward(self, x, l2=False):
        b, c, w, h = x.shape
        q = self.q_conv(x).view(b, -1, w * h).permute(0, 2, 1)  # (b, n, c_bar)
        k = self.k_conv(x).view(b, -1, w * h)  # (b, c_bar, n)
        v = self.v_conv(x).view(b, -1, w * h)  # (b, c, n)
        attn_matrix = q @ k  # (b, n, n)
        attention = self.sf(attn_matrix)  # (b, n, n)
        out = v @ attention.permute(0, 2, 1)  # (b, c, n)
        out = out.view(b, -1, w, h)  # (b, c, w, h)
        out = self.gamma * out + x
        return out, attention


class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64):
        super(Generator, self).__init__()
        self.block1 = ConvBlock(nz, ngf * 8, 4, 1, 0, transpose=True)  # (b, 512, 4, 4)
        self.block2 = ConvBlock(ngf * 8, ngf * 4, 4, 2, 1, transpose=True)  # (b, 256, 8, 8)
        self.block3 = ConvBlock(ngf * 4, ngf * 2, 4, 2, 1, transpose=True)  # (b, 128, 16, 16)
        self.block4 = ConvBlock(ngf * 2, ngf * 1, 4, 2, 1, transpose=True)  # (b, 64, 32, 32)
        self.block5 = ConvBlock(ngf * 1, 3, 4, 2, 1, transpose=True, is_bn=False, activate='tanh')  # (b, 3, 64, 64)

        self.attn1 = Attn(128)
        self.attn2 = Attn(64)

    def forward(self, z):
        """
        :param z:  (b, 100, 1, 1)
        :return:
        """
        x = self.block1(z)
        x = self.block2(x)
        x = self.block3(x)
        x, attention1 = self.attn1(x)
        x = self.block4(x)
        x, attention2 = self.attn2(x)
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

        self.attn1 = Attn(256)
        self.attn2 = Attn(512)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x, attention1 = self.attn1(x)
        x = self.block4(x)
        x, attention2 = self.attn2(x)
        x = self.block5(x)
        return F.sigmoid(x.squeeze())
