#!/bin/bash

cd .. # go to parent dir to run

# This script is used to automate the collection of codecarbon data for different models and configurations just for the FMoE model.

# WHAT IT DOES:
# 1. automate codecarbon data collection for:
#     - switch-transformer - fmoe
#     - qwen-moe - fmoe
# 2. loop with different batch sizes and MORE training steps

# USAGE: 
# ./codecarbon_datacollect_fmoe.sh

# function that runs the model with the given parameters - SWITCH TRANSFORMERS
# input parameters: $1 number of GPUs, $2 batch size, $3 number of experts, $4 data set split, $5 run_num
run_model_switch() {
    local num_gpus=$1
    local batch_size=$2
    local num_experts=$3
    local dataset_split=$4
    local run_num=$5

    # Trainers: simple, fmoe, deepspeed, custom

    echo "Running switch-transformer models with $num_gpus GPUs, batch size $batch_size, $num_experts experts, dataset split '$dataset_split', run number $run_num"
    echo "$run_num,$num_gpus,$batch_size,$num_experts,$dataset_split" >> codecarbonlogs/fmoe_runs/run_log_switch.csv

    echo "----------------------------------------------------"
    echo "RUNNING SWITCH-TRANSFORMER FMOE"
    echo "----------------------------------------------------"
    # switch-transformer - fmoe
    torchrun --nproc-per-node="$num_gpus" --master-addr="localhost" --master-port=12355 launch.py \
        --model switch-base-n \
        --trainer fmoe \
        --batch_size "$batch_size" \
        --dataset_split "$dataset_split" \
        --switch_transformer_num_experts "$num_experts" \
        --train_stats codecarbon \
        --project_name "fmoe_runs/switch-128-fmoe" \
        --run_num "$run_num" 
}

# function that runs the model with the given parameters - QWEN-MOE
# input parameters: $1 number of GPUs, $2 batch size, $3 number of experts, $4 data set split, $5 run_num
run_model_qwen() {
    local num_gpus=$1
    local batch_size=$2
    local num_experts=$3
    local dataset_split=$4
    local run_num=$5

    # Trainers: deepspeed, fmoe, +simple?

    echo "Running qwen-moe models with $num_gpus GPUs, batch size $batch_size, $num_experts experts, dataset split '$dataset_split', run number $run_num"
    echo "$run_num,$num_gpus,$batch_size,$num_experts,$dataset_split" >> codecarbonlogs/fmoe_runs/run_log_qwen.csv

    echo "----------------------------------------------------"
    echo "RUNNING QWEN-MOE FMOE"
    echo "----------------------------------------------------"
    # qwen moe - fmoe
    torchrun --nproc-per-node="$num_gpus" --master-addr="localhost" --master-port=12355 launch.py \
        --model qwen-moe \
        --trainer fmoe \
        --batch_size "$batch_size" \
        --dataset_split "$dataset_split" \
        --qwen_num_experts "$num_experts" \
        --train_stats codecarbon \
        --project_name "fmoe_runs/qwen-moe-60-fmoe" \
        --run_num "$run_num" 
}

# create the directories for the codecarbon data files
mkdir -p codecarbonlogs
mkdir -p codecarbonlogs/fmoe_runs
mkdir -p codecarbonlogs/fmoe_runs/switch-128-fmoe
mkdir -p codecarbonlogs/fmoe_runs/qwen-moe-60-fmoe


# Loop through different configurations
echo "run_num,num_gpus,batch_size,num_experts,dataset_split" > codecarbonlogs/fmoe_runs/run_log_switch.csv
echo "run_num,num_gpus,batch_size,num_experts,dataset_split" > codecarbonlogs/fmoe_runs/run_log_qwen.csv

# SWITCH TRANSFORMER
run_num=1
batch_sizes=(9 10)
num_gpus_values=(4)
num_experts_values=(128)
num_iterations=(10 30 50 100)
for num_gpus in "${num_gpus_values[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do 
        for num_experts in "${num_experts_values[@]}"; do
            for iterations in "${num_iterations[@]}"; do
						dataset_split="train[:$((num_gpus * batch_size * iterations))]"
            echo "----------------------------------------"
            echo "Run $run_num config:"
            echo "  GPUs        : $num_gpus"
            echo "  Batch size  : $batch_size"
            echo "  Num experts : $num_experts"
            echo "  Dataset     : $dataset_split"
            echo "----------------------------------------"
            run_model_switch "$num_gpus" "$batch_size" "$num_experts" "$dataset_split" "$run_num"
            run_num=$((run_num + 1))
						done
        done
    done
done

# QWEN-MOE
run_num=1
batch_sizes=(6)
num_gpus_values=(4)
num_experts_values=(60)
num_iterations=(10 30 50 100)
for num_gpus in "${num_gpus_values[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do 
        for num_experts in "${num_experts_values[@]}"; do
				for iterations in "${num_iterations[@]}"; do
            dataset_split="train[:$((num_gpus * batch_size * iterations))]"
			echo "----------------------------------------"
            echo "Run $run_num config:"
            echo "  GPUs        : $num_gpus"
            echo "  Batch size  : $batch_size"
            echo "  Num experts : $num_experts"
            echo "  Dataset     : $dataset_split"
            echo "----------------------------------------"
            run_model_qwen "$num_gpus" "$batch_size" "$num_experts" "$dataset_split" "$run_num"
            run_num=$((run_num + 1))
        		done
				done
    done
done



# helper function for file naming
get_run_params_suffix() {
    local log_file=$1
    local run_num=$2
    params=$(awk -F, -v run="$run_num" 'NR>1 && $1 == run { print $4 "_" $3 "_" $2; exit }' "$log_file")
    # num_experts_batch_size_num_gpus
    echo "$params"
}

# delete the extra useless files
echo "Cleaning up unnecessary files..."
# go into codecarbonlogs directory
cd codecarbonlogs || exit 
cd fmoe_runs || exit

# Merge the codecarbon data files into a single file
echo "Merging codecarbon data files..."
# for each subdirectory, find all files with the same run number
# for each run number, merge the "codecarbon_full_training_{gpu_num}.csv" files into a single file and add a "gpu_rank" column to the merged file
for dir in */; do
    echo "Processing directory: $dir"
    # run numbers
    run_numbers=$(find "$dir" -type f -name "run_*_cc_*.csv" | \
        sed -E 's|.*/run_([0-9]+)_cc_.*|\1|' | sort -u)

    # For each run number, merge all corresponding files
    for run_num in $run_numbers; do
        echo "  Merging run number: $run_num"

        log_file="run_log_switch.csv"
        [[ "$dir" == *qwen* ]] && log_file="run_log_qwen.csv" 
        param_suffix=$(get_run_params_suffix "$log_file" "$run_num") 

        echo "    Parameters suffix: '$param_suffix'"
        echo "    Log file: $log_file"

        # Collect all matching files for this run for full training
        filesfull=$(find "$dir" -type f -name "run_${run_num}_cc_full_rank_*.csv" | sort)
        if [ -z "$filesfull" ]; then
            echo "    No files found for run $run_num, skipping."
            continue
        fi

        # Collect all matching files for this run for steps
        filesstep=$(find "$dir" -type f -name "run_${run_num}_cc_step_rank_*-steps.csv" | sort)
        if [ -z "$filesstep" ]; then
            echo "    No files found for run $run_num, skipping."
            continue
        fi

        # Collect all matching files for this run for substeps
        filessub=$(find "$dir" -type f -name "run_${run_num}_cc_substep_rank_*-substeps.csv" | sort)
        if [ -z "$filessub" ]; then
            echo "    No files found for run $run_num, skipping."
            continue
        fi

        # Call Python to merge
        python3 - <<EOF
import pandas as pd
import os

merged = []
for filepath in """$filesfull""".split():
    # Extract GPU rank from filename
    gpu_rank = filepath.split("_")[-1].replace(".csv","")
    df = pd.read_csv(filepath)
    df["gpu_rank"] = gpu_rank
    merged.append(df)

if merged:
    result = pd.concat(merged, ignore_index=True)
    output_file = "${dir}run_${run_num}_cc_full_${param_suffix}_merged.csv"
    result.to_csv(output_file, index=False)
    print(f"    Saved merged file: {output_file}")

mergedstep = []
for filepath in """$filesstep""".split():
    # Extract GPU rank from filename
    gpu_rank = filepath.split("_rank_")[1].split("-")[0]
    df = pd.read_csv(filepath)
    df["gpu_rank"] = gpu_rank
    mergedstep.append(df)

if mergedstep:
    result = pd.concat(mergedstep, ignore_index=True)
    output_file = "${dir}run_${run_num}_cc_step_${param_suffix}_merged.csv"
    result.to_csv(output_file, index=False)
    print(f"    Saved merged file: {output_file}")

mergedsub = []
for filepath in """$filessub""".split():
    # Extract GPU rank from filename
    gpu_rank = filepath.split("_rank_")[1].split("-")[0]
    df = pd.read_csv(filepath)
    df["gpu_rank"] = gpu_rank
    mergedsub.append(df)

if mergedsub:
    result = pd.concat(mergedsub, ignore_index=True)
    output_file = "${dir}run_${run_num}_cc_substep_${param_suffix}_merged.csv"
    result.to_csv(output_file, index=False)
    print(f"    Saved merged file: {output_file}")
EOF

    done
    echo "Done processing directory: $dir"
done

# Cleanup files
echo "Cleaning up individual files..."
# for every subdirectory remove all files that dont have "merged" in the name
for dir in */; do
    # skip the plots
    if [[ "$dir" == "plots/" ]]; then
        continue
    fi
   echo "Cleaning up directory: $dir"
   find "$dir" -type f ! -name "*merged*.csv" -not -path "$dir/plots/*" -exec rm {} \;
#    find "$dir" -type f ! -name "*merged*.csv"
done

# cd back out to the original directory
cd .. || exit
cd .. || exit


SCRIPT_DIR=$(dirname "$(realpath $0)")
PLOTS_DIR=$SCRIPT_DIR/codecarbonlogs/plots/fmoe
cd $SCRIPT_DIR/codecarbonlogs/fmoe_runs

DISTRIBUTED_TRAINERS="fmoe"

function get_model_from_dir() {
	local dir=$1
	echo $dir | sed -r -e 's/([[:alnum:]-]*)-([[:digit:]]*)-([[:alnum:]]*)/\1/g'
}

function get_num_experts_from_dir() {
	local dir=$1
	echo $dir | sed -r -e 's/([[:alnum:]-]*)-([[:digit:]]*)-([[:alnum:]]*)/\2/g'
}

function get_trainer_from_dir() {
	local dir=$1
	echo $dir | sed -r -e 's/([[:alnum:]-]*)-([[:digit:]]*)-([[:alnum:]]*)/\3/g'
}

function get_run_number_from_cc_file() {
	local file=$1
	echo $file | sed -r -e 's/run_([[:digit:]]*)_cc_full_([[:digit:]]*)_([[:digit:]]*)_([[:digit:]]*)_merged.csv/\1/g'
}

function get_number_experts_from_cc_file() {
	local file=$1
	echo $file | sed -r -e 's/run_([[:digit:]]*)_cc_full_([[:digit:]]*)_([[:digit:]]*)_([[:digit:]]*)_merged.csv/\2/g'
}

function get_batch_size_from_cc_file() {
	local file=$1
	echo $file | sed -r -e 's/run_([[:digit:]]*)_cc_full_([[:digit:]]*)_([[:digit:]]*)_([[:digit:]]*)_merged.csv/\3/g'
}

function get_number_gpus_from_cc_file() {
	local file=$1
	echo $file | sed -r -e 's/run_([[:digit:]]*)_cc_full_([[:digit:]]*)_([[:digit:]]*)_([[:digit:]]*)_merged.csv/\4/g'
}

function handle_distributed_run() {
	local model=$1
	local num_experts=$2
	local run=$3
	local granularity=$4

	local filepaths=""
	local labels=""

	for dir in $(ls); do
		if [[ $dir == "$model-$num_experts"* ]]; then
			local trainer=$(get_trainer_from_dir $dir)
			if [[ $DISTRIBUTED_TRAINERS =~ $trainer ]]; then
				for p in $(ls $dir/run_${run}_cc_full_*); do
					local f=$(basename $p)
					local batch_size=$(get_batch_size_from_cc_file $f)
					local num_gpus=$(get_number_gpus_from_cc_file $f)
					filepaths+=" $dir/run_${run}_cc_${granularity}_${num_experts}_${batch_size}_${num_gpus}_merged.csv"
					labels+=" ${model}_${trainer}_${num_experts}_${batch_size}_${num_gpus}"
				done
			fi
		fi
	done

	mkdir -p $PLOTS_DIR/$model/run_${run}_${granularity}
#	cat <<EOF
	python3 $SCRIPT_DIR/plot/main.py --filepaths $filepaths --labels ${labels} --granularity ${granularity} --mode multi-model --output $PLOTS_DIR/$model/run_${run}_${granularity}
#EOF

}

function handle_distributed() {
	local model=$1
	local num_experts=$2

	# TODO This is a little disgusting, I'm sorry. I'll come back to it later.
	local runs=()
	for dir in $(ls); do
		if [[ $dir == "$model-$num_experts"* ]]; then
			local trainer=$(get_trainer_from_dir $dir)
			if [[ $DISTRIBUTED_TRAINERS =~ $trainer ]]; then
				for p in $(ls $dir/*_cc_full_*); do
					local f=$(basename $p)
					runs+=($(get_run_number_from_cc_file $f))
				done
				break
			fi
		fi
	done

	for run in ${runs[@]}; do
		handle_distributed_run $model $num_experts $run "substep"
	done
}



function handle_model() {
	local model=$1
	local num_experts=$2
	handle_distributed $model $num_experts
}

models=(
	"qwen-moe-60"
	"switch-128"
)
mkdir -p $PLOTS_DIR/switch
mkdir -p $PLOTS_DIR/qwen-moe

# THE DUMB WAY
# mkdir -p $PLOTS_DIR/switch/run_1_substep
# mkdir -p $PLOTS_DIR/switch/run_2_substep
# mkdir -p $PLOTS_DIR/switch/run_3_substep
# mkdir -p $PLOTS_DIR/switch/run_4_substep
# mkdir -p $PLOTS_DIR/switch/run_5_substep
# mkdir -p $PLOTS_DIR/switch/run_6_substep
# mkdir -p $PLOTS_DIR/switch/run_7_substep
# mkdir -p $PLOTS_DIR/switch/run_8_substep
# mkdir -p $PLOTS_DIR/qwen-moe/run_1_substep
# mkdir -p $PLOTS_DIR/qwen-moe/run_2_substep
# mkdir -p $PLOTS_DIR/qwen-moe/run_3_substep
# mkdir -p $PLOTS_DIR/qwen-moe/run_4_substep
# mkdir -p $PLOTS_DIR/qwen-moe/run_5_substep
# mkdir -p $PLOTS_DIR/qwen-moe/run_6_substep
# mkdir -p $PLOTS_DIR/qwen-moe/run_7_substep
# mkdir -p $PLOTS_DIR/qwen-moe/run_8_substep

for model in ${models[@]}; do
	m=$(echo $model | sed -r -e 's/([[:alnum:]-]*)-([[:digit:]]*)/\1/g')
	n_experts=$(echo $model | sed -r -e 's/([[:alnum:]-]*)-([[:digit:]]*)/\2/g')
	handle_model $m $n_experts
done
