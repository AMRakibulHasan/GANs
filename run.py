import argparse
import torch
import os
from Trainers.DCGAN import DCGAN
from Trainers.WGAN import WGAN
from Trainers.SAGAN import SAGAN

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

parse.add_argument('--model', type=str, choices=['dcgan', 'wgan', 'sagan'], default='dcgan')
parse.add_argument('--num_workers', type=int, help="dataloader的参数", default=4)
parse.add_argument('--img_size', type=int, help="图片大小", default=64)

args = parse.parse_args()

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

    trainer.train()
