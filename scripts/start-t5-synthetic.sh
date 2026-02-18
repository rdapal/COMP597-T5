#!/bin/bash

# ==============================
# T5 with MilaBench Synthetic Data
# ==============================
# 
# This uses synthetic random token generation matching MilaBench's approach.
#
# Parameters from MILA Bench:
#   vocab_size: 32128 (T5's vocabulary)
#   train_length: 512
#   n: 4 (unique samples)
#   repeat: calculated from the split
#

SCRIPTS_DIR=$(readlink -f -n $(dirname \$0))

echo "========================================"
echo "T5 Benchmark with MilaBench Synthetic Data"
echo "========================================"
echo "vocab_size=32128, train_length=512, n=4"
echo ""

# Run T5 with synthetic data
${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model t5 \
    --trainer simple \
    --data synthetic \
    --batch_size 4 \
    --learning_rate 1e-6 \
    --data_configs.dataset.split '"train[:500]"' \
    --trainer_stats simple
