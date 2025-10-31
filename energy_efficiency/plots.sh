#!/bin/bash

cd ..

SCRIPT_DIR=$(dirname "$(realpath $0)")
PLOTS_DIR=$SCRIPT_DIR/codecarbonlogs/plots
cd $SCRIPT_DIR/codecarbonlogs

DISTRIBUTED_TRAINERS="fmoe custom deepspeed"

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

function handle_simple() {
	local model=$1
	local num_experts=$2
	
	for dir in $(find . -maxdepth 1 -name "$model-$num_experts-*" -type d); do
		local trainer=$(basename $(get_trainer_from_dir $dir))
		for p in $(ls $dir/*_cc_full_*); do
			local f=$(basename $p)
			local run_num=$(get_run_number_from_cc_file $f)
			local batch_size=$(get_batch_size_from_cc_file $f)
			local num_gpus=$(get_number_gpus_from_cc_file $f)
			local filepaths="${dir}/run_${run_num}_cc_substep_${num_experts}_${batch_size}_${num_gpus}_merged.csv"
			local labels="${model}_simple_${num_experts}_${batch_size}_${num_gpus}"
	#		cat <<EOF
			python3 $SCRIPT_DIR/plot/main.py --filepaths ${filepaths} --labels ${labels} --granularity substep --mode single --output $PLOTS_DIR/$model-$num_experts-$trainer
#EOF
		done
	done
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

#	cat <<EOF
	python3 $SCRIPT_DIR/plot/main.py --filepaths $filepaths --labels ${labels} --granularity ${granularity} --mode multi-model --output $PLOTS_DIR/$model
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
	handle_simple $model $num_experts
	handle_distributed $model $num_experts
}

models=(
	"qwen-moe-60"
	"qwen-moe-15"
	"qwen-moe-10"
	"switch-128"
	"switch-32"
	"switch-16"
)

mkdir -p $PLOTS_DIR/qwen-moe
mkdir -p $PLOTS_DIR/switch
mkdir -p $PLOTS_DIR/switch-128-custom
mkdir -p $PLOTS_DIR/switch-128-deepspeed
mkdir -p $PLOTS_DIR/switch-128-fmoe
mkdir -p $PLOTS_DIR/switch-16-simple
mkdir -p $PLOTS_DIR/switch-32-simple
mkdir -p $PLOTS_DIR/qwen-moe-60-fmoe
mkdir -p $PLOTS_DIR/qwen-moe-60-deepspeed
mkdir -p $PLOTS_DIR/qwen-moe-15-simple
mkdir -p $PLOTS_DIR/qwen-moe-10-simple

for model in ${models[@]}; do
	m=$(echo $model | sed -r -e 's/([[:alnum:]-]*)-([[:digit:]]*)/\1/g')
	n_experts=$(echo $model | sed -r -e 's/([[:alnum:]-]*)-([[:digit:]]*)/\2/g')
	handle_model $m $n_experts
done
