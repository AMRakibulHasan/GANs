import torch
import torch.nn as nn
from Trainers.baseTrainer import BaseTrainer
from networks.SAGAN import Generator, Discriminator
from utils import weights_init
import time
import torch.nn.functional as F


class SAGAN(BaseTrainer):
    def __init__(self, args):
        super(SAGAN, self).__init__(args)
        if self.rank == 0:
            print('sagan...')

    def _init_model(self):
        self.gen = Generator(self.args.nz, self.args.ngf)
        self.dis = Discriminator(self.args.ndf)

        self.gen = nn.SyncBatchNorm.convert_sync_batchnorm(self.gen).cuda()
        self.dis = nn.SyncBatchNorm.convert_sync_batchnorm(self.dis).cuda()
        # self.gen.apply(weights_init)
        # self.dis.apply(weights_init)
        self.gen_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, self.gen.parameters()), 1e-4, betas=(0.0, 0.9))
        self.dis_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, self.dis.parameters()), 4e-4, betas=(0.0, 0.9))
        self.cri = nn.BCELoss()

    def train(self):
        patten = "[%03d/%03d][%03d/%03d]  Loss_D: %.4f"
        print(self.gen)
        print(self.dis)

        for epoch in range(self.args.epochs):
            cur_inputs = None
            for batch, inputs in enumerate(self.dl, 0):
                b = inputs.shape[0]
                self.gen_opt.zero_grad()
                self.dis_opt.zero_grad()
                inputs = inputs.cuda()
                cur_inputs = inputs.clone()
                # (1) update D
                d_real_loss = self.dis(inputs)

                # d_real_loss = F.relu(1 - d_real_loss).mean()    #  figine
                real_label = torch.ones_like(d_real_loss).cuda()    # DCGAN
                d_real_loss = self.cri(d_real_loss, real_label)     # DCGAN

                d_real_loss.backward()
                noise_z = torch.randn((b, self.args.nz, 1, 1)).cuda()
                fake_x = self.gen(noise_z)
                d_fake_loss = self.dis(fake_x.detach())

                # d_fake_loss = F.relu(1 + d_fake_loss)    # figine
                fake_label = torch.zeros_like(d_fake_loss).cuda()     # DCGAN
                d_fake_loss = self.cri(d_fake_loss, fake_label)       # DCGAN

                d_fake_loss.mean().backward()
                self.dis_opt.step()

                # (2) update G
                noise_z = torch.randn((b, self.args.nz, 1, 1)).cuda()
                fake_x = self.gen(noise_z)
                g_fake_loss = self.dis(fake_x)

                # g_fake_loss = -g_fake_loss.mean()    # figine
                g_fake_loss = self.cri(g_fake_loss, real_label)   # DCGAN

                g_fake_loss.backward()
                self.gen_opt.step()
                # for param in self.dis.parameters():
                #     torch.clip_(param.data, -self.args.c, self.args.c)
                if batch % self.args.log_steps == 0:
                    if self.rank == 0:
                        print(patten % (
                            epoch,
                            self.args.epochs,
                            batch,
                            len(self.dl),
                            (d_fake_loss - d_real_loss).mean().item()
                        ))

            self.val(cur_inputs, epoch)
            if epoch % 5 == 0:
                self.save_model()

        end = time.time()
        print('sum cost: %.4fs' % (end - self.start))