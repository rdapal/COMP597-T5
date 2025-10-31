#!/bin/bash

cd .. # go to parent dir to run

# This script is used to automate the cleanup of codecarbon data files after collection.

# WHAT IT DOES:
# 1. merges the codecarbon data files into a single file for each run
# 2. merges the losses into the step files
# 3. deletes individual files that are not needed anymore

# USAGE:
# ./codecarbon_cleanup.sh

# ------------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------------ #

# helper function for file naming
get_run_params_suffix() {
    local log_file=$1
    local run_num=$2
    params=$(awk -F, -v run="$run_num" 'NR>1 && $1 == run { print $4 "_" $3 "_" $2; exit }' "$log_file")
    # params: num_experts_batch_size_num_gpus
    echo "$params"
}

# go into codecarbonlogs directory
cd codecarbonlogs || exit 

# ------------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------------ #

# Merge the codecarbon data files into a single file
echo "Merging codecarbon data files..."
# for each subdirectory, find all files with the same run number
# for each run number, merge the "codecarbon_full_training_{gpu_num}.csv" files into a single file and add a "gpu_rank" column to the merged file
for dir in */; do
    echo "Processing directory: $dir"
    project_name="${dir%/}"
    # run numbers
    run_numbers=$(find "$dir" -type f -name "run_*_cc_*.csv" | \
        sed -E 's|.*/run_([0-9]+)_cc_.*|\1|' | sort -u)

    # each run number, merge all corresponding files
    for run_num in $run_numbers; do
        echo "  Merging run number: $run_num"

        # if plots dir, skip
        if [[ "$dir" == "plots/" ]]; then
            echo "    Skipping directory: $dir (plots directory)"
            continue
        fi

        # --- merge losses ---
        # if losses dir, merge losses
        if [[ "$dir" == "losses/" ]]; then
            for subdir in "$dir"*/; do
                echo "  Processing subdirectory: $subdir"
                # get the project name from the subdir name
                project_name=$(basename "$subdir")
                echo "    Project name: $project_name"
                # merge losses for this run number
                echo "    Merging losses for run number: $run_num"
                # find all loss files for this run number
				loss_files=$(find "$subdir" -type f -name "run_${run_num}_cc_loss_rank_*.csv" | sort)
                if [ -z "$loss_files" ]; then
                    echo "    No loss files found for run $run_num, skipping."
                    continue
                fi
                # merge losses
                python3 - <<EOF
import pandas as pd
import os
import re
import glob

log_dir = "$subdir"
run_id = "run_${run_num}"

def clean_loss(tensor_str):
    match = re.search(r"tensor\\(([\d\\.]+)", tensor_str)
    return float(match.group(1)) if match else None

# loss files for the given run number
loss_files = sorted(glob.glob(os.path.join(log_dir, f"{run_id}_cc_loss_rank_*.csv")))

dfs = []
for f in loss_files:
    df = pd.read_csv(f)
    df["loss"] = df["loss"].apply(clean_loss)
    dfs.append(df)

if dfs:
    combined_df = pd.concat(dfs).sort_values(by=["gpu_rank", "task_name"])
    merged_path = os.path.join(log_dir, f"{run_id}_cc_loss_merged.csv")
    combined_df.to_csv(merged_path, index=False)
    print(f"    > Merged loss CSV saved to: {merged_path}")
else:
    print("    No valid DataFrames to merge.")
EOF
            done
		fi

        # --- merge codecarbon data files ---
        # for all other dirs - codecarbon data files
        # get the right log file
        log_file="run_log_switch.csv"
        [[ "$dir" == *qwen* ]] && log_file="run_log_qwen.csv" 
        param_suffix=$(get_run_params_suffix "$log_file" "$run_num") 

        echo "    Parameters suffix: '$param_suffix'"
        echo "    Log file: $log_file"

        # files for full training
        filesfull=$(find "$dir" -type f -name "run_${run_num}_cc_full_rank_*.csv" | sort)
        if [ -z "$filesfull" ]; then
            echo "    No files found for run $run_num, skipping."
            continue
        fi

        # files for steps
        filesstep=$(find "$dir" -type f -name "run_${run_num}_cc_step_rank_*-steps.csv" | sort)
        if [ -z "$filesstep" ]; then
            echo "    No files found for run $run_num, skipping."
            continue
        fi

        # files for substeps
        filessub=$(find "$dir" -type f -name "run_${run_num}_cc_substep_rank_*-substeps.csv" | sort)
        if [ -z "$filessub" ]; then
            echo "    No files found for run $run_num, skipping."
            continue
        fi

        # merge the files
        python3 - <<EOF
import pandas as pd
import os

# merge files for full training
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

# merge files for steps 
mergedstep = []
for filepath in """$filesstep""".split():
    # Extract GPU rank from filename
    gpu_rank = filepath.split("_rank_")[1].split("-")[0]
    df = pd.read_csv(filepath)
    df["gpu_rank"] = gpu_rank
    mergedstep.append(df)

# + merge the losses into the step file
if mergedstep:
    result = pd.concat(mergedstep, ignore_index=True)
    output_file = "${dir}run_${run_num}_cc_step_${param_suffix}_merged.csv"
    result.to_csv(output_file, index=False)
    print(f"    Saved merged file: {output_file}")
    print("    Adding losses...")
    loss_path = "losses/${project_name}/run_${run_num}_cc_loss_merged.csv"
    if os.path.exists(loss_path):
        loss_df = pd.read_csv(loss_path)

        result["gpu_rank"] = result["gpu_rank"].astype(str)
        loss_df["gpu_rank"] = loss_df["gpu_rank"].astype(str)

        loss_result = pd.merge(result, loss_df, on=["gpu_rank", "task_name"], how="left")
        loss_result.to_csv(output_file, index=False)
        print(f"    Merged losses into: {output_file}")
    else:
        print("    No losses to merge.")

# merge files for substeps
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


# ------------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------------ #

# cleaning up files
echo "Cleaning up individual files..."
# for every subdirectory remove all files that dont have "merged" in the name
for dir in */; do
    # skip the plots 
    if [[ "$dir" == "plots/" ]]; then
        echo "Skipping directory: $dir"
        continue
    fi
   echo "Cleaning up directory: $dir"

    # check if the dir contains anything with merged files first
    if ! find "$dir" -maxdepth 1 -type f -name '*merged*' | grep -q .; then
        echo "No 'merged' files in $dir, skipping..."
        continue
    fi
    find "$dir" -type f ! -name "*merged*.csv" # prints the files that will be deleted for confirmation
    find "$dir" -type f ! -name "*merged*.csv" -not -path "$dir/plots/*" -exec rm {} \;
done

# ------------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------------ #

# for all subdirectories with *-simple, go in and edit the file names. 
for dir in */; do
    if [[ "$dir" == *-simple/ ]]; then
        echo "Renaming files in directory: $dir"
        # get the num_experts from the directory name - the right number
        correct_experts=$((echo "$dir" | sed -r -e 's/.*-([0-9]+)-simple/\1/'))
        echo "  Correct experts: $correct_experts"
        # rename the files in the directory
        for f in $(ls); do mv $f $(echo $f | sed -r -e "s/(run_[[:digit:]]*_cc_[[:alnum:]]*_)([[:digit:]]+)(\w*)/\1${correct_experts}\3/g"); done
    else
        echo "Skipping directory: $dir (not a simple run)"
    fi
done