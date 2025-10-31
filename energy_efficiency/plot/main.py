import argparse
import plot_utils
import plotting
import experiment
import re
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Plot training metrics from CSV.")
    parser.add_argument("--filepaths", nargs='+', required=True, help="Paths to one or more CSV files containing training data.")
    parser.add_argument("--labels", nargs='+', required=True, help="Labels for each file in the format '<model_name>_<trainer_type>_<num_experts>_<granularity>_<batch_size>_<num_gpus>' e.g. 'switch_custom_substep_128_9_8', 'qwen_moe_deepspeed_step_60_2_4'")
    parser.add_argument("--mode", choices=["single", "multi-trainer", "multi-param"], required=True, help="The type of data provided: single file, multiple trainers (same model and parameters), or multiple parameters (same model and trainer).")
    parser.add_argument("--metrics", nargs='+', default= ['duration'], help="List of metrics to plot: e.g., duration emissions energy_consumed")
    parser.add_argument("--output", required=True, help="Path to the output directory where the figures will be saved.")
    
    args = parser.parse_args()

    # Input validation
    label_pattern = r'^(.+)_([^_]+)_([^_]+)_(\d+)_(\d+)_(\d+)$'
    for label in args.labels:
        match = re.match(label_pattern, label)
        if not match:
            raise ValueError(f"Label '{label}' does not match expected format: <model_name>_<trainer_type>_<granularity>_<num_experts>_<batch_size>_<num_gpus>")

    if len(args.filepaths) > 1 & mode != "single": 
        raise ValueError(f"Mode 'single' only supports 1 filepath, but you inputed {len(args.filepaths)} filepaths.")
    
    if len(args.filepaths) = 1 & mode != "single": 
        raise ValueError(f"Mode '{args.mode}' requires multiple filepaths, but you inputted only 1.")

    return args

def generate_and_save_figs(experiments, mode, metric, output_dir):
    granularity = experiments[0].granularity

    # SUBSTEP GRAPHS
    if granularity == "substep":
        if mode == "single":
            exp = experiments[0]
            # Plot 1: group by rank
            fig1 = plotting.metric_per_iteration_grouped_by_rank(exp, metric)
            title1 = f"{metric}_per_iteration_compare_ranks_{exp.num_experts}_{exp.batch_size}_{exp.num_gpus}" 
            filepath1 = os.path.join(output_dir, title1)
            fig1.savefig(filepath1, dpi=400)

            # Plot 2: separate by rank
            fig2 = plotting.metric_per_iteration_separated_by_rank(experiments[0], metric)
            title2 = f"{metric}_per_iteration_split_ranks_{exp.num_experts}_{exp.batch_size}_{exp.num_gpus}"
            filepath2 = os.path.join(output_dir, title2)
            fig2.savefig(filepath2, dpi=400)

        elif mode == "multi-trainer":
            exp = experiments[0] if experiments[0].model != 'simple' else experiments[1]
            # Plot 1: group by trainer, seperate by rank
            fig1 = plotting.metric_per_iteration_grouped_by_trainer_separated_by_rank(experiments, metric)
            title1 = f"{metric}_per_iteration_compare_trainers_{exp.num_experts}_{exp.batch_size}_{exp.num_gpus}"
            filepath1 = os.path.join(output_dir, title1)
            fig1.savefig(filepath1, dpi=400)

    # STEP GRAPHS
    elif granularity == "step":
        if mode == "multi-trainer":
            exp = experiments[0]
            # Plot 1: plot
            fig1 = plotting.metric_against_loss_per_iteration(experiments, metric)
            title1 = f"{metric}_vs_loss_per_iteration_compare_trainers_{exp.num_experts}_{exp.batch_size}_{exp.num_gpus}"
            filepath1 = os.path.join(output_dir, title1)
            fig1.savefig(filepath1, dpi=400)

def main():
    # Parse command-line arguments
    args = parse_args()

    #  Each Experiment object has these attributes:
    #  'model': String, 
    #  'trainer': String
    #  'granularity': String,
    #  'learning_rate': String,
    #  'num_experts': int,
    #  'batch_size': int,
    #  'num_gpus': int,
    #  'num_iterations': int, 
    #  'ranks': [int], 
    #  'data': dictionary containing training data
    
    experiments = []
    for filepath, label in zip(args.filepaths, args.labels): 
        exp = experiment.Experiment(filepath, label)
        experiments.append(exp)

    # Generate and save graphs
    for metric in args.metrics:
        generate_and_save_figs(experiments, args.mode, metric, args.output)

if __name__ == "__main__":
    main()
