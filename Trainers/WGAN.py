import torch
import torch.nn as nn
from Trainers.baseTrainer import BaseTrainer
from networks.DCGAN import Generator, Discriminator
from utils import weights_init
import time


class WGAN(BaseTrainer):
    def __init__(self, args):
        super(WGAN, self).__init__(args)
        if self.rank == 0:
            print("训练wgan...")

    def _init_model(self):
        self.gen = Generator(self.args.nz, self.args.ngf)
        self.gen = nn.SyncBatchNorm.convert_sync_batchnorm(self.gen).cuda()
        self.dis = Discriminator(self.args.ndf)
        self.dis = nn.SyncBatchNorm.convert_sync_batchnorm(self.dis).cuda()
        # self.gen.apply(weights_init)
        # self.dis.apply(weights_init)
        self.gen = nn.parallel.DistributedDataParallel(self.gen, device_ids=[self.rank], output_device=self.rank)
        self.dis = nn.parallel.DistributedDataParallel(self.dis, device_ids=[self.rank], output_device=self.rank)
        self.gen_opt = torch.optim.RMSprop(self.gen.parameters(), lr=self.args.lr)
        self.dis_opt = torch.optim.RMSprop(self.dis.parameters(), lr=self.args.lr)
        # self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=self.args.lr)
        # self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr=self.args.lr)
        # self.cri = nn.BCELoss()

    def train(self):
        # Loss  maximize log(D(x)) + log(1-D(G(z)))
        patten = "[%03d/%03d][%03d/%03d]  Loss_D: %.4f"
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
                # print(next(self.dis.parameters()).device)
                real_s = self.dis(inputs)
                real_label = torch.ones_like(real_s).cuda()
                # errD_real = self.cri(real_s, real_label)
                # real_s.backward()
                D_x = real_s.mean().item()
                noise = torch.randn((b, self.args.nz, 1, 1)).cuda()
                fake_x = self.gen(noise)
                fake_s = self.dis(fake_x.detach())
                errD = (fake_s - real_s).mean()
                errD.backward()
                self.dis_opt.step()
                for param in self.dis.parameters():
                    torch.clip_(param.data, -self.args.c, self.args.c)
                # fake_label = torch.zeros_like(fake_s).cuda()
                # errD_fake = self.cri(fake_s, fake_label)
                # errD_fake.backward()


                # (2) Update G network
                noise = torch.randn((b, self.args.nz, 1, 1)).cuda()
                # print('real:', inputs.shape)
                # print('noise:', noise.shape)
                fake_x = self.gen(noise)
                # print('fake_x:', fake_x.shape)
                fake_s = self.dis(fake_x)
                # errG = self.cri(fake_s, real_label)
                (-1*fake_s).mean().backward()
                self.gen_opt.step()
                # for param in self.gen.parameters():
                #     torch.clip_(param, -self.args.c, self.args.c)


                D_fake_x = fake_s.mean().item()

                if batch % self.args.log_steps == 0:
                    if self.rank == 0:
                        print(patten % (
                            epoch,
                            self.args.epochs,
                            batch,
                            len(self.dl),
                            errD.item()

                        ))

            self.val(cur_inputs, epoch)
            if epoch % 5 == 0:
                self.save_model()

        end = time.time()
        print('sum cost: %.4fs' % (end-self.start))


