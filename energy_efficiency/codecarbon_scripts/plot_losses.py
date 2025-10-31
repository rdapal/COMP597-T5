import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    """
    This script reads a CSV file containing loss data and generates plots.
    It plots the average loss across all GPUs and saves the plot to a specified output file.
    Usage:
        python plot_losses.py --input <path_to_loss_csv> --output <output_image_file>
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the loss CSV file")
    parser.add_argument("--output", required=True, help="Base output image file path (e.g. 'loss_plot.png')")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    # Parse step numbers from task_name
    df["step"] = df["task_name"].str.extract(r"#(\d+)").astype(int)

    # Sort by step
    df = df.sort_values(by=["gpu_rank", "step"])

    base_output = os.path.splitext(args.output)[0]
    ext = os.path.splitext(args.output)[1]

    # # Plot per-GPU loss curves
    # for rank in df["gpu_rank"].unique():
    #     subset = df[df["gpu_rank"] == rank]
    #     plt.figure()
    #     plt.plot(subset["step"], subset["loss"], marker="o")
    #     plt.xlabel("Step")
    #     plt.ylabel("Loss")
    #     plt.title(f"Loss Curve - GPU {rank}")
    #     plt.grid(True)
    #     plt.tight_layout()

    #     output_path = f"{base_output}_gpu{rank}{ext}"
    #     plt.savefig(output_path)
    #     plt.close()
    #     print(f"Saved plot for GPU {rank} to {output_path}")

    # Plot average loss across GPUs
    avg_df = df.groupby("step")["loss"].mean().reset_index()
    plt.figure()
    plt.plot(avg_df["step"], avg_df["loss"], marker="o", color="black")
    plt.xlabel("Step")
    plt.ylabel("Average Loss")
    plt.title("Average Loss Curve Across All GPUs")
    plt.grid(True)
    plt.tight_layout()

    avg_output_path = f"{base_output}_average{ext}"
    plt.savefig(avg_output_path)
    plt.close()
    print(f"Saved average loss plot to {avg_output_path}")

if __name__ == "__main__":
    main()
