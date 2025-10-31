#!/bin/bash

# This script is used to plot the losses from the codecarbon data files.

# WHAT IT DOES:
# 1. goes to the codecarbonlogs directory
# 2. finds all loss files in the losses subdirectory
# 3. plots the losses for each run and saves the plots in the plots subdirectory
# USAGE:
# ./plot_losses.sh

cd ..

SCRIPT_DIR=$(dirname "$(realpath $0)")
echo "Running plotting script from $SCRIPT_DIR"
LOSSES_DIR=$SCRIPT_DIR/codecarbonlogs/losses
PLOTS_DIR=$SCRIPT_DIR/codecarbonlogs/plots/losses
PLOT_SCRIPT=$SCRIPT_DIR/codecarbon_scripts/plot_losses.py
mkdir -p $PLOTS_DIR
mkdir -p $PLOTS_DIR/switch-128-deepspeed
mkdir -p $PLOTS_DIR/switch-128-fmoe
mkdir -p $PLOTS_DIR/qwen-moe-64-fmoe
mkdir -p $PLOTS_DIR/qwen-moe-64-deepspeed

echo "Plotting losses for $LOSSES_DIR..."

for file in "$LOSSES_DIR"/*/*.csv; do
	if [[ -f "$file" ]]; then
		echo "Processing file: $file"
		# Extract project name and run number from the file path
		project_name=$(basename "$(dirname "$file")")
		run_num=$(basename "$file" | sed -E 's/run_([0-9]+)_cc_loss_rank_.*/\1/')

		# Define output plot file
		output_plot="$PLOTS_DIR/$project_name/${run_num}_loss_plot.png"

		# Run the plotting script
		echo "Plotting losses for $project_name, run $run_num..."
		python3 "$PLOT_SCRIPT" --input "$file" --output "$output_plot"
	fi
done
