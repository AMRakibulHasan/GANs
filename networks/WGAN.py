import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, nz, ngf):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),  # (ngf*8, 4, 4)
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),  # (ngf*4, 8, 8)
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),  # (ngf*2, 16, 16)
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),  # (ngf, 32, 32)
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),  # (img_ize, 64, 64)
            nn.Tanh(),
        )

    def forward(self, z):
        # print(z.shape)
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, ndf):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),  # (ndf, 32, 32)
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),  # (2*ndf, 16, 16)
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),  # (4*ndf, 8, 8)
            nn.BatchNorm2d(4*ndf),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),   # (8*ndf, 4, 4)
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(ndf * 8, 1, 4, 2, 0, bias=False),      # (1, 1, 1)
            # nn.Sigmoid(),
        )

    def forward(self, img):
        return self.net(img).squeeze()

