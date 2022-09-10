import torch
import torch.nn as nn
from dataset.dataLoader import DL
from utils import safe_create_dir
from torchvision.utils import save_image
import time
import os
from tqdm import tqdm


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
        gen = {'net': self.gen.state_dict(),
               'opt': self.gen_opt.state_dict()}
        dis = {'net': self.dis.state_dict(),
               'opt': self.dis_opt.state_dict()}
        torch.save(gen, self.save_path + 'gen.pt')
        torch.save(dis, self.save_path + 'dis.pt')
        self.gen.cuda()
        self.dis.cuda()

    def load_model(self):
        gen = torch.load(self.save_path + 'gen.pt')
        dis = torch.load(self.save_path + 'dis.pt')
        self.gen.load_state_dict(gen['net'])
        self.gen.cuda()
        self.gen_opt.load_state_dict(gen['opt'])
        self.gen = torch.nn.parallel.DistributedDataParallel(self.gen, device_ids=[self.rank], output_device=self.rank)

        self.dis.load_state_dict(dis['net'])
        self.dis.cuda()
        self.dis_opt.load_state_dict(dis['opt'])
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
        noise = torch.randn((64, self.args.nz, 1, 1)).cuda()
        save_image((imgs[:64] * 0.5) + 0.5, self.save_path + 'Img/real/real_%d.png' % epoch, nrow=8, padding=True)
        fake_x = self.gen(noise)
        save_image((fake_x[:64] * 0.5) + 0.5, self.save_path + 'Img/fake/fake_%d.png' % epoch, nrow=8, padding=True)

        self.gen.train()

    @torch.no_grad()
    def test(self):
        self.load_model()
        self.gen.eval()
        safe_create_dir(os.path.join(self.save_path, 'Img', 'test_fake'))
        safe_create_dir(os.path.join(self.save_path, 'Img', 'test_real'))
        fake_path = os.path.join(self.save_path, 'Img', 'test_fake')
        real_path = os.path.join(self.save_path, 'Img', 'test_real')
        for batch, inputs in enumerate(tqdm(self.dl, ncols=100), 0):
            b = inputs.shape[0]
            inputs = inputs.cuda()
            noise_z = torch.randn((b, self.args.nz, 1, 1)).cuda()
            fake_x = self.gen(noise_z)
            save_image((inputs[:64] * 0.5) + 0.5, os.path.join(real_path, '%s.png' % batch))
            save_image((fake_x[:64] * 0.5) + 0.5, os.path.join(fake_path, '%s.png' % batch))

