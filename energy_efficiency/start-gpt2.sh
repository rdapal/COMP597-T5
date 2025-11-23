#!/bin/bash

# ==============================
# Run file for GPT2 example - how to use the launch script to run GPT2 with simple trainer
# ==============================

### run GPT2 Simple Trainer
python launch.py --model gpt2 --trainer simple --batch_size 4 --learning_rate 1e-6 --dataset_split "train[:100]" --train_stats simple "$@"

### run GPT2 with CodeCarbon tracking
python launch.py --model gpt2 --trainer simple --batch_size 4 --learning_rate 1e-6 --dataset_split "train[:100]" --train_stats codecarbon --project_name "gpt2-codecarbon" --run_num 1 "$@"