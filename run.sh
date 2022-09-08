#!/bin/sh

# train DCGAN
#torchrun --nproc_per_node=3 run.py --log_steps 10 --model dcgan --epochs 300

# train WGAN
torchrun --nproc_per_node=3 run.py --log_steps 10 --model wgan --epochs 2000 --lr 1e-3 --batch_size 500