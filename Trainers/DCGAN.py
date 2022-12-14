import torch
import torch.nn as nn
from Trainers.baseTrainer import BaseTrainer
from networks.DCGAN import Generator, Discriminator
from utils import weights_init
import time
from losses import BCELoss
import torch.nn.functional as F


class DCGAN(BaseTrainer):
    def __init__(self, args):
        super(DCGAN, self).__init__(args)
        if self.rank == 0:
            print("dcgan...")

    def _init_model(self):
        self.gen = Generator(self.args.nz, self.args.ngf)
        self.gen = nn.SyncBatchNorm.convert_sync_batchnorm(self.gen).cuda()
        self.dis = Discriminator(self.args.ndf)
        self.dis = nn.SyncBatchNorm.convert_sync_batchnorm(self.dis).cuda()
        self.gen.apply(weights_init)
        self.dis.apply(weights_init)
        self.gen = nn.parallel.DistributedDataParallel(self.gen, device_ids=[self.rank], output_device=self.rank)
        self.dis = nn.parallel.DistributedDataParallel(self.dis, device_ids=[self.rank], output_device=self.rank)
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=self.args.lr, betas=(self.args.beta1, 0.999))
        self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr=self.args.lr, betas=(self.args.beta1, 0.999))
        self.cri = BCELoss()

    def train(self):
        # Loss  maximize log(D(x)) + log(1-D(G(z)))
        patten = "[%03d/%03d]  IS: %.4f   IS_std: %.4f   FID: %.4f"
        for epoch in range(self.args.epochs):
            self.sampler.set_epoch(epoch)
            cur_inputs = None
            for batch, inputs in enumerate(self.dl, 0):
                self.gen_opt.zero_grad()
                self.dis_opt.zero_grad()
                cur_inputs = inputs.clone()
                b = inputs.shape[0]
                # (1) Update D network
                inputs = inputs.cuda()
                real_s = self.dis(inputs)
                real_label = torch.ones_like(real_s).cuda()
                errD_real = F.binary_cross_entropy(real_s, real_label)
                errD_real.backward()
                # D_x = real_s.mean().item()
                noise = torch.randn((b, self.args.nz, 1, 1)).cuda()
                fake_x = self.gen(noise)
                fake_s = self.dis(fake_x.detach())
                fake_label = torch.zeros_like(fake_s).cuda()
                errD_fake = F.binary_cross_entropy(fake_s, fake_label)
                # D_loss = self.cri(real_s, fake_s)
                errD_fake.backward()
                self.dis_opt.step()

                # (2) Update G network
                noise = torch.randn((b, self.args.nz, 1, 1)).cuda()
                # print('real:', inputs.shape)
                # print('noise:', noise.shape)
                fake_x = self.gen(noise)
                # print('fake_x:', fake_x.shape)
                fake_s = self.dis(fake_x)
                G_loss = self.cri(fake_s)
                # errG = self.cri(fake_s, real_label)
                G_loss.backward()
                self.gen_opt.step()


                # if batch % self.args.log_steps == 0:
                #     if self.rank == 0:
                #         print(patten % (
                #             epoch,
                #             self.args.epochs,
                #             batch,
                #             len(self.dl),
                #             errD_fake.item() + errD_real.item(),
                #             errG.item(),
                #             D_x,
                #             D_fake_x,
                #         ))

            self.val(cur_inputs, epoch)
            IS, IS_std, FID = self.evaluate()
            if self.rank == 0:
                print(patten % (
                    epoch,
                    self.args.epochs,
                    IS,
                    IS_std,
                    FID,
                ))
            if epoch % 5 == 0:
                self.save_model()

        end = time.time()
        if self.rank == 0:
            print('sum cost: %.4fs' % (end-self.start))
            print()
            print()
            print()


