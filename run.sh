#!/bin/sh

# train DCGAN
#torchrun --nproc_per_node=1 run.py --log_steps 10 --model dcgan --epochs 300 --batch_size 64

# train WGAN
#torchrun --nproc_per_node=1 run.py --log_steps 10 --model wgan --epochs 2000 --lr 5e-5 --batch_size 200

# train SAGAN
torchrun --nproc_per_node=1 run.py --log_steps 10 --model sagan --epochs 1000 --batch_size 64

