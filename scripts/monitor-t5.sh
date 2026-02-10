#!/bin/bash

# ==============================
# Monitor GPU while running T5
# ==============================

SCRIPTS_DIR=$(readlink -f -n $(dirname \$0))

# Run T5 with GPU monitoring
# nvidia-smi logs GPU stats every 1 second to a file

${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model t5 \
    --trainer simple \
    --batch_size 4 \
    --learning_rate 1e-6 \
    --data_configs.dataset.split '"train[:100]"' \
    --data_configs.dataset.name "allenai/c4" \
    --data_configs.dataset.train_files '${COMP597_JOB_STORAGE_PARTITION}/example/c4/downloaded/multilingual/c4-en.tfrecord-00000-of-*.json.gz' \
    --trainer_stats simple
