from Trainers.baseTrainer import BaseTrainer
from networks.GNGAN import Generator, Discriminator
from losses import HingeLoss
import torch.nn as nn
import torch
from utils import normalize_gradient, weights_init
import time


class GNGAN(BaseTrainer):
    def __init__(self, args):
        super(GNGAN, self).__init__(args)
        if self.rank == 0:
            print('GNGAN...')

    def _init_model(self):
        self.gen = Generator(self.args.nz, self.args.ngf)
        self.dis = Discriminator(self.args.ndf)

        self.gen = nn.SyncBatchNorm.convert_sync_batchnorm(self.gen).cuda()
        self.dis = nn.SyncBatchNorm.convert_sync_batchnorm(self.dis).cuda()
        self.gen.apply(weights_init)
        self.dis.apply(weights_init)
        self.gen_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, self.gen.parameters()), 2e-4,
                                        betas=(0.0, 0.9))
        self.dis_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, self.dis.parameters()), 2e-4,
                                        betas=(0.0, 0.9))
        self.cri = HingeLoss()

    def train(self):
        patten = "[%03d/%03d]  IS: %.4f   IS_std: %.4f   FID: %.4f"
        for epoch in range(self.args.epochs):
            cur_inputs = None
            self.sampler.set_epoch(epoch)
            for batch, inputs in enumerate(self.dl):
                inputs = inputs.cuda()

                cur_inputs = inputs.clone()

                # (1) Update D
                x = torch.split(inputs, self.args.batch_size_D, dim=0)
                x = iter(x)
                for _ in range(self.args.ndis):
                    self.dis_opt.zero_grad()
                    real_x = next(x)
                    b = real_x.shape[0]
                    noise_z = torch.randn((b, self.args.nz, 1, 1)).cuda()
                    with torch.no_grad():
                        fake_x = self.gen(noise_z).detach()
                    real_fake = torch.cat([real_x, fake_x], dim=0)
                    preds = normalize_gradient(self.dis, real_fake)
                    D_real_loss, D_fake_loss = torch.split(preds, b, dim=0)
                    D_loss = self.cri(D_real_loss, D_fake_loss)
                    D_loss.backward()

                    self.dis_opt.step()
                    # for param in self.dis.parameters():
                    #     torch.clip_(param.data, -self.args.c, self.args.c)

                # (2) Update G
                self.gen_opt.zero_grad()
                noise_z = torch.randn((self.args.batch_size_G, self.args.nz, 1, 1)).cuda()
                fake_x = self.gen(noise_z)
                preds = normalize_gradient(self.dis, fake_x)
                G_fake_loss = self.cri(preds)
                G_fake_loss.backward()
                self.gen_opt.step()
                # if batch % self.args.log_steps == 0:
                #     if self.rank == 0:
                #         print(patten % (
                #             epoch,
                #             self.args.epochs,
                #             batch,
                #             len(self.dl),
                #             G_fake_loss.item() - D_loss.item(),
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
            print('sum cost: %.4fs' % (end - self.start))
            print()
            print()
            print()