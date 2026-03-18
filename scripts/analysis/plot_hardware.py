"""
Plot hardware monitoring results for T5 energy analysis v.4

Averages data across 3 repetitions and plots phase bar charts per batch size,
- Reads the 500ms CSV output from hardware.py
- Removes invalid sub-millisecond per-phase energy/power visualisations,
  focusing on accurate macro-timelines as directed by Prof. Balmau

Plots Generated:
  1. Phase Durations         — if profile_phase was 'all' or specific phase
  2. Energy per Step         — dual-axis plot: per-step + cumulative
  3. CO2 Emissions           — dual-axis plot: per-step + cumulative
  4. GPU Memory              — line plot: allocated / reserved / peak
  5. GPU & CPU Utilization   — line plot with steady-state averages
  6. GPU Temperature         — line plot with thermal rise annotation

Usage:
    python scripts/analysis/plot_hardware.py --csv <path> --output <dir> --run_id <id>
"""

import argparse
import glob
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================
WARMUP_STEPS = 6

# ============================================================
# Helpers
# ============================================================
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df

def steady(df):
    return df.iloc[WARMUP_STEPS:]

def savefig(fig, output_dir, name, bs):
    path = os.path.join(output_dir, f"{name}_bs{bs}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {path}")

# ============================================================
# Aggregate and Plot
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing the CSVs')
    parser.add_argument('--output_dir', type=str, required=True, help='Output dir for plots')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_files = glob.glob(os.path.join(args.input_dir, "hardware_stats_bs*_rep*_*.csv"))
    
    # Extract unique batch sizes
    batch_sizes = set()
    for f in csv_files:
        match = re.search(r'bs(\d+)_rep\d+_(timeline|fwd|bwd|opt)', f)
        if match:
            batch_sizes.add(int(match.group(1)))

    for bs in sorted(batch_sizes, reverse=True):
        print(f"Processing Aggregated Plots for Batch Size {bs}...")
        
        # --- 1. Phase Durations (Bar Chart averaged over 3 reps) ---
        phases = {'fwd': 'forward_time_ms', 'bwd': 'backward_time_ms', 'opt': 'optimizer_time_ms'}
        phase_means, phase_stds = [], []
        phase_labels = ['Forward', 'Backward', 'Optimizer']

        for phase_abbr, col_name in phases.items():
            phase_files = glob.glob(os.path.join(args.input_dir, f"hardware_stats_bs{bs}_rep*_{phase_abbr}.csv"))
            all_vals = []
            for pf in phase_files:
                df = pd.read_csv(pf)
                if len(df) > WARMUP_STEPS:
                    steady = df.iloc[WARMUP_STEPS:]
                    all_vals.extend(steady[col_name].tolist())
            
            if all_vals:
                phase_means.append(np.mean(all_vals))
                phase_stds.append(np.std(all_vals))
            else:
                phase_means.append(0); phase_stds.append(0)

        fig, ax = plt.subplots(figsize=(8, 6))
        x_pos = np.arange(len(phase_labels))
        ax.bar(x_pos, phase_means, yerr=phase_stds, capsize=5, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(phase_labels)
        ax.set_ylabel('Time (ms)')
        ax.set_title(f'Average Phase Duration Across 3 Repetitions (BS {bs})')
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(phase_means):
            ax.text(i, v + (phase_stds[i] if phase_stds[i] else 0) + 1, f"{v:.1f}ms", ha='center', fontweight='bold')
        savefig(fig, args.output_dir, 'phase_durations_bar', bs)

        # --- 2. Timelines (Averaged across 3 reps) ---
        timeline_files = glob.glob(os.path.join(args.input_dir, f"hardware_stats_bs{bs}_rep*_timeline.csv"))
        if not timeline_files:
            continue
        
        dfs = [pd.read_csv(tf) for tf in timeline_files]
        min_len = min(len(df) for df in dfs)
        dfs_trunc = [df.iloc[:min_len].reset_index(drop=True) for df in dfs]
        df_avg = sum(dfs_trunc) / len(dfs_trunc)

        # Energy Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_avg['step_num'], df_avg['energy_step_j'] * 1000, color='#e74c3c', lw=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Energy (mJ)')
        ax.set_title(f'Energy Per Step Timeline (Averaged over 3 Reps, BS={bs})')
        ax.grid(True, alpha=0.3)
        savefig(fig, args.output_dir, 'energy_timeline', bs)

        # GPU Memory Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_avg['step_num'], df_avg['gpu_memory_allocated_mb'], label='Allocated', color='#2ecc71', lw=2)
        ax.plot(df_avg['step_num'], df_avg['gpu_memory_reserved_mb'], label='Reserved', color='#3498db', lw=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Memory (MB)')
        ax.set_title(f'GPU Memory Timeline (Averaged over 3 Reps, BS={bs})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        savefig(fig, args.output_dir, 'gpu_memory_timeline', bs)

        # Utilization Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_avg['step_num'], df_avg['gpu_utilization'], label='GPU Util (%)', color='#e67e22', lw=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Utilization (%)')
        ax.set_ylim(0, 105)
        ax.set_title(f'GPU Utilization Timeline (Averaged over 3 Reps, BS={bs})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        savefig(fig, args.output_dir, 'utilization_timeline', bs)

if __name__ == '__main__':
    main()