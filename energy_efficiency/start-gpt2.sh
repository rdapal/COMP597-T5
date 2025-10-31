#!/bin/bash

# ==============================
# Run file for GPT2 experiments
# ==============================

### run GPT2 simple trainer
# torchrun --nproc-per-node=1 --master-addr="localhost" --master-port=12355 launch.py --model gpt2 --trainer simple --batch_size 4 --dataset_split "train[:100]" --train_stats simple "$@"

### run GPT2 on specific GPUs use --include or --exclude
deepspeed --include localhost:0,1,2,3 --bind_cores_to_rank launch.py --model gpt2 --trainer deepspeed --batch_size 4 --dataset_split "train[:100]" --learning_rate 1e-6 --train_stats simple "$@"