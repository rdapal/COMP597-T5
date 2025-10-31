import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data(file : str) -> pd.DataFrame:
    return pd.read_csv(file)

def process_data(data : pd.DataFrame, ref_freq : int) -> pd.DataFrame:
    mean_data = data.groupby(["gpu_freq"])[["time_ns", "energy_uj"]].aggregate(["mean", "std"])
    mean_data = mean_data.reset_index()
    mean_data.columns = ["gpu_freq", "time_ns", "time_ns_std", "energy_uj", "energy_uj_std"]
    mean_data["norm_time"] = mean_data["time_ns"] / mean_data[mean_data["gpu_freq"] == ref_freq].iloc[0]["time_ns"]
    mean_data["norm_time_std"] = mean_data["time_ns_std"] / mean_data[mean_data["gpu_freq"] == ref_freq].iloc[0]["time_ns"]
    mean_data["norm_energy"] = mean_data["energy_uj"] / mean_data[mean_data["gpu_freq"] == ref_freq].iloc[0]["energy_uj"]
    mean_data["norm_energy_std"] = mean_data["energy_uj_std"] / mean_data[mean_data["gpu_freq"] == ref_freq].iloc[0]["energy_uj"]
    return mean_data

def compute_minor_ticks(ticks, jump):
    min = 1e12
    max = -1e12
    for tick in ticks:
        if tick < min:
            min = tick
        if tick > max:
            max = tick
    return np.arange(min, max, jump)
    # minor_ticks = []
    # prev = None
    # for tick in ticks:
    #     if prev is not None:
    #         i = prev
    #         while i < tick:
    #             minor_ticks.append(i)
    #             i += jump
    #     prev = tick
    # return minor_ticks

def plot_data(data : pd.DataFrame, title : str) -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.errorbar(data["gpu_freq"].values, data["norm_time"].values, yerr=data["norm_time_std"].values, marker="s", markersize=5, capsize=5, label="Computation time")
    ax.errorbar(data["gpu_freq"].values, data["norm_energy"].values, yerr=data["norm_energy_std"].values, marker="o", markersize=5, capsize=5, label="Energy consumption")
    # plt.hlines([0.8, 0.9, 0.95, 1.05, 1.1, 1.2, 1.4], xmin=0, xmax=1)
    # plt.axhline(0.9)
    ax.set_xlabel("GPU frequency (MHz)")
    ax.set_ylabel("Normalized to max frequency")
    ax.set_title(title)
    ax.legend()
    ax.set_yticks(compute_minor_ticks(ax.get_yticks(), jump=0.1), minor=True)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.7)
    

    return fig

def save(fig : plt.Figure, out : str, dpi : int) -> None:
    fig.savefig(out, dpi=dpi)

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--file", type=str, help="Name of the csv file containing the data")
    parser.add_argument("--ref_freq", type=int, help="The frequency against which to normalize")
    parser.add_argument("-o", "--output", type=str, help="Name of the output image file.", default="out.png")
    parser.add_argument("--dpi", type=int, help="DPI of the output image.", default=600)
    parser.add_argument("--title", type=str, help="Title of the plot. If not set, uses file name.", default=None)

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    print(args.title)

    title = args.title if args.title is not None else args.file

    data = load_data(args.file)
    data = process_data(data, args.ref_freq)
    fig = plot_data(data, title)
    save(fig, args.output, args.dpi)

if __name__ == "__main__":
    main()
