import argparse
import torch
import os
from Trainers.DCGAN import DCGAN
from Trainers.WGAN import WGAN
from Trainers.SAGAN import SAGAN
from Trainers.GNGAN import GNGAN
from utils import create_evaluate_npz


# os.environ['CUDA_VISIBLE_DEVICE'] = '0,1'

parse = argparse.ArgumentParser('DCGAN')
parse.add_argument('--data_path', type=str, help="数据集所在路径", default='../MyData/cv/CelebA/image/')
parse.add_argument('--batch_size', type=int, help="批次大小", default=64)
parse.add_argument('--epochs', type=int, help="轮次", default=300)
parse.add_argument('--nz', type=int, help="随机初始化的噪声的大小", default=100)
parse.add_argument('--ngf', type=int, help="生成器的放缩倍数", default=64)
parse.add_argument('--ndf', type=int, help="鉴别器的放缩倍数", default=64)
parse.add_argument('--lr', type=float, help="学习率", default=2e-4)
parse.add_argument('--beta1', type=float, help="adam优化器的参数", default=0.5)
parse.add_argument('--log_steps', type=int, help="多少batch打印一次日志", default=1)
# parse.add_argument('--save_steps', type=int, help="多少个batch保存一次", default=10)
# parse.add_argument('--save_strategy', type=str, choices=['step', 'epoch'], default='epoch')

parse.add_argument('--c', type=float, help="wgan的参数clip值", default=0.01)
parse.add_argument('--mode', type=str, choices=['test', 'train'], default='train')
parse.add_argument('--model', type=str, choices=['dcgan', 'wgan', 'sagan', 'gngan'], default='dcgan')
parse.add_argument('--num_workers', type=int, help="dataloader的参数", default=4)
parse.add_argument('--img_size', type=int, help="图片大小", default=64)
parse.add_argument('--create_npz', action='store_true', help="Whether to create a dataset for evaluation?")

# GNGAN
parse.add_argument('--ndis', type=int, help="the number of discriminator updates per generator", default=5)
parse.add_argument('--batch_size_D', type=int, help="batch size of Discriminator", default=64)
parse.add_argument('--batch_size_G', type=int, help="batch size of Generator", default=128)
args = parse.parse_args()

if args.model == 'gngan':
    args.batch_size = args.batch_size_D * args.ndis

torch.distributed.init_process_group(backend='nccl', world_size=torch.cuda.device_count())
rank = torch.distributed.get_rank()
torch.cuda.set_device(rank)

if __name__ == "__main__":
    trainer = None
    if args.model == 'dcgan':
        trainer = DCGAN(args)
    elif args.model == 'wgan':
        trainer = WGAN(args)
    elif args.model == 'sagan':
        trainer = SAGAN(args)
    elif args.model == 'gngan':
        trainer = GNGAN(args)

    if args.create_npz:
        create_evaluate_npz(args)

    elif args.mode == 'train':
        trainer.train()

    elif args.mode == 'test':
        trainer.test()
