import torch.nn as nn
import torch.nn.functional as F
import torch


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.cri = nn.BCELoss()

    def forward(self, real, fake=None):
        real_label = torch.ones_like(real).cuda()
        loss = self.cri(real, real_label)
        if fake is not None:
            fake_label = torch.zeros_like(fake).cuda()
            loss = loss + self.cri(fake, fake_label)

        return loss


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, real, fake=None):
        loss = F.relu(1 - real).mean()
        if fake is not None:
            loss = loss + F.relu(1 + fake).mean()

        return loss


class WassersteinLoss(nn.Module):
    def __init__(self):
        super(WassersteinLoss, self).__init__()

    def forward(self, real, fake=None):
        loss = -real.mean()
        if fake is not None:
            loss = loss + fake.mean()
        return loss
