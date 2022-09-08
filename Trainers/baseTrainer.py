import torch
import torch.nn as nn
from dataset.dataLoader import DL
from utils import safe_create_dir
from torchvision.utils import save_image
import time


class BaseTrainer:
    def __init__(self, args):
        self.args = args
        if 'CelebA' in self.args.data_path:
            self.model = 'CelebA'
        safe_create_dir('results')
        safe_create_dir('results/CelebA')
        safe_create_dir('results/CelebA/%s' % self.args.model)
        safe_create_dir('results/CelebA/%s/Img' % self.args.model)
        safe_create_dir('results/CelebA/%s/Img/fake/' % self.args.model)
        safe_create_dir('results/CelebA/%s/Img/real/' % self.args.model)
        self.save_path = 'results/CelebA/%s/' % self.args.model
        self.rank = torch.distributed.get_rank()
        self.start = time.time()
        self._init_data()
        self._init_model()

    def _init_model(self):
        """
        子类继承
        :return:
        """

        pass

    def _init_data(self):
        self.data = DL(self.args)
        self.sampler = self.data.sampler
        self.dl = self.data.dl

    def save_model(self):
        self.gen.cpu()
        self.dis.cpu()
        torch.save(self.gen.state_dict(), self.save_path + 'gen.pt')
        torch.save(self.dis.state_dict(), self.save_path + 'dis.pt')
        self.gen.cuda()
        self.dis.cuda()

    def load_model(self):
        self.gen.load_state_dict(torch.load(self.save_path + 'gen.pt'))
        self.gen.cuda()
        self.gen = torch.nn.parallel.DistributedDataParallel(self.gen, device_ids=[self.rank], output_device=self.rank)

        self.dis.load_state_dict(torch.load(self.save_path + 'dis.pt'))
        self.dis.cuda()
        self.dis = torch.nn.parallel.DistributedDataParallel(self.dis, device_ids=[self.rank], output_device=self.rank)

    def train(self):
        """
        子类继承这个方法
        :return:
        """
        pass

    @torch.no_grad()
    def val(self, imgs, epoch):
        """
        验证，用来绘图
        :return:
        """
        self.gen.eval()
        noise = torch.randn((100, self.args.nz, 1, 1)).cuda()
        save_image((imgs[:100] * 0.5) + 0.5, self.save_path + 'Img/real/real_%d.png' % epoch, nrow=10, padding=True)
        fake_x = self.gen(noise)
        save_image((fake_x[:100] * 0.5) + 0.5, self.save_path + 'Img/fake/fake_%d.png' % epoch, nrow=10, padding=True)

        self.gen.train()
