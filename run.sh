#!/bin/sh

torchrun --nproc_per_node=3 run.py --log_steps 10 --model dcgan --epochs 300