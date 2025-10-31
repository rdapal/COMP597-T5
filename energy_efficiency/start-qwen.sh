#!/bin/bash

# =============================
# Run file for QWEN experiments
# =============================
# NOTE: enable this first command if it does not matter which GPU is used. 

### run qwen deepspeed with the simple stats
# deepspeed --num_nodes=1 --num_gpus=8 --bind_cores_to_rank launch.py --model qwen-moe --trainer deepspeed --batch_size 4 --dataset_split "train[:400]" --qwen_num_experts 64 --learning_rate 1e-6 --train_stats simple "$@"
### run qwen deepspeed with the codecarbon stats
# deepspeed --num_nodes=1 --num_gpus=8 --bind_cores_to_rank launch.py --model qwen-moe --trainer deepspeed --batch_size 4 --dataset_split "train[:]" --qwen_num_experts 64 --learning_rate 1e-6 --train_stats codecarbon --project_name "qwen-moe-64-deepspeed" --run_num 1 "$@"

### to run on specific GPUs use --include or --exclude
deepspeed --include localhost:4,5,6,7 --bind_cores_to_rank launch.py --model qwen-moe --trainer deepspeed --batch_size 4 --dataset_split "train[:]" --qwen_num_experts 64 --learning_rate 1e-6 --train_stats simple "$@"

### to run with the custom trainer
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# torchrun --nproc-per-node=4 --master-addr="localhost" --master-port=12355 launch.py --model qwen-moe --trainer custom-distributed --batch_size 4 --dataset_split "train[:100]" --qwen_num_experts 64 --learning_rate 1e-6 --train_stats simple "$@"
