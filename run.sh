#!/bin/sh

# [1] Train


# train DCGAN
torchrun --nproc_per_node=2 run.py --log_steps 10 --model dcgan --epochs 300 --batch_size 64

# train SAGAN
torchrun --nproc_per_node=2 run.py --log_steps 10 --model sagan --epochs 300 --batch_size 64

# train WGAN
torchrun --nproc_per_node=2 run.py --log_steps 10 --model wgan --epochs 300 --lr 5e-5 --batch_size 64

# train SNGAN
torchrun --nproc_per_node=2 run.py --log_steps 10 --model dcgan --epochs 300 --batch_size 64

# train GNGAN
torchrun --nproc_per_node=2 run.py --log_steps 10 --model gngan --epochs 300 --ndis 1 --nz 128



# [2] Test

# test DCGAN
#torchrun --nproc_per_node=1 run.py --model dcgan --batch_size 64 --mode test

# test SAGAN
#torchrun --nproc_per_node=1 run.py --model sagan --batch_size 64 --mode test

# test WGAN
#torchrun --nproc_per_node=1 run.py --model wgan --batch_size 64 --mode test




# create_npz
#torchrun --nproc_per_node=1 run.py --create_npz --batch_size 3000