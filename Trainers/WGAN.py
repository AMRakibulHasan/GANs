import torch
import torch.nn as nn
from Trainers.baseTrainer import BaseTrainer
from networks.DCGAN import Generator, Discriminator
from utils import weights_init
import time
from losses import WassersteinLoss


class WGAN(BaseTrainer):
    def __init__(self, args):
        super(WGAN, self).__init__(args)
        if self.rank == 0:
            print("wgan...")

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
        self.cri = WassersteinLoss()

    def train(self):
        # Loss  maximize log(D(x)) + log(1-D(G(z)))
        patten = "[%03d/%03d]  IS: %.4f   IS_std: %.4f   FID: %.4f"
        for epoch in range(self.args.epochs):
            self.sampler.set_epoch(epoch)
            cur_inputs = None
            for batch, inputs in enumerate(self.dl, 0):
                # print('************* batch %d  *************' % batch)
                self.gen_opt.zero_grad()
                self.dis_opt.zero_grad()
                cur_inputs = inputs.clone()
                b = inputs.shape[0]
                # (1) Update D network
                inputs = inputs.cuda()
                real_s = self.dis(inputs)
                noise = torch.randn((b, self.args.nz, 1, 1)).cuda()
                fake_x = self.gen(noise)

                (-real_s).mean().backward(retain_graph=True)
                fake_s = self.dis(fake_x.detach())

                # D_loss = self.cri(real_s, fake_s)
                # D_loss.backward()
                fake_s.mean().backward(retain_graph=True)

                self.dis_opt.step()
                for param in self.dis.parameters():
                    torch.clip_(param.data, -self.args.c, self.args.c)



                # (2) Update G network
                noise = torch.randn((b, self.args.nz, 1, 1)).cuda()

                fake_x = self.gen(noise)
                # print('fake_x:', fake_x.shape)
                fake_s = self.dis(fake_x)
                # errG = self.cri(fake_s, real_label)
                G_loss = self.cri(fake_s)
                # (-1*fake_s).mean().backward()
                G_loss.backward()
                self.gen_opt.step()
                # for param in self.gen.parameters():
                #     torch.clip_(param, -self.args.c, self.args.c)


                D_fake_x = fake_s.mean().item()

                # if batch % self.args.log_steps == 0:
                #     if self.rank == 0:
                #         print(patten % (
                #             epoch,
                #             self.args.epochs,
                #             batch,
                #             len(self.dl),
                #             errD.item()
                #
                #         ))

            self.val(cur_inputs, epoch)
            IS, IS_std, FID = self.evaluate()
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
        print('sum cost: %.4fs' % (end - self.start))
        print()
        print()
        print()
