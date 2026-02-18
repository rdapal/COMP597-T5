#!/bin/bash

# ==============================
# T5 Hardware Analysis with Synthetic Data
# ==============================
# 
# Runs T5 with synthetic data and hardware monitoring
# Outputs CSV files to the specified directory for plotting
#

SCRIPTS_DIR=$(readlink -f -n $(dirname \$0))

echo "========================================"
echo "T5 Hardware Analysis"
echo "========================================"
echo "Data: Synthetic (MILABENCH)"
echo "Stats: Hardware monitoring"
echo ""

# Run T5 with hardware monitoring
${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model t5 \
    --trainer simple \
    --data synthetic \
    --batch_size 4 \
    --learning_rate 1e-6 \
    --data_configs.dataset.split '"train[:500]"' \
    --trainer_stats hardware \
    --trainer_stats_configs.hardware.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/t5-hardware'
