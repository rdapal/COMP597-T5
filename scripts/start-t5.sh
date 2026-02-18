#!/bin/bash

# ==============================
# Run file for T5 example
# ==============================

SCRIPTS_DIR=$(readlink -f -n $(dirname \$0))
REPO_DIR=$(readlink -f -n ${SCRIPTS_DIR}/..)

### run T5 Simple Trainer (currently no codecarbon)
${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model t5 \
    --trainer simple \
    --batch_size 4 \
    --learning_rate 1e-6 \
    --data_configs.dataset.split '"train[:100]"' \
    --data_configs.dataset.name "bench/t5" \
    --data_configs.dataset.train_files '${COMP597_JOB_STORAGE_PARTITION}/example/c4/downloaded/multilingual/c4-en.tfrecord-00000-of-*.json.gz' \
    --trainer_stats simple

### run T5 with CodeCarbon tracking
# ${SCRIPTS_DIR}/srun.sh \
#     --logging.level INFO \
#     --model t5 \
#     --trainer simple \
#     --batch_size 4 \
#     --learning_rate 1e-6 \
#     --data_configs.dataset.split '"train[:100]"' \
#     --data_configs.dataset.name "allenai/c4" \
#     --data_configs.dataset.train_files '${COMP597_JOB_STORAGE_PARTITION}/example/c4/downloaded/multilingual/c4-en.tfrecord-00000-of-*.json.gz' \
#     --trainer_stats codecarbon \
#     --trainer_stats_configs.codecarbon.run_num 1 \
#     --trainer_stats_configs.codecarbon.project_name t5-energy \
#     --trainer_stats_configs.codecarbon.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/t5-benchmark/codecarbonlogs'
