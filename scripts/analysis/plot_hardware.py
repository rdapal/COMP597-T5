"""
Plot hardware monitoring results for T5 energy analysis

Generates plots from hardware_stats CSV files:
  Timing:
    1. Phase Duration vs Training Step
    2. Phase Breakdown Pie Chart
    3. Warmup Comparison Bar Chart
  Hardware:
    4. GPU Memory Usage vs Training Step
    5. GPU Utilization vs Training Step
  Energy:
    6. GPU Power Draw Per Phase vs Training Step
    7. Energy Breakdown Pie Chart
    8. Per-Step Energy vs Training Step
    9. GPU Temperature vs Training Step
    10. Time vs Energy Comparison
    11. Warmup Energy Comparison

Usage:
    python scripts/analysis/plot_hardware.py --csv <path_to_csv> --output <output_dir>
    python scripts/analysis/plot_hardware.py --csv <path_to_csv> --output <output_dir> --run_id t5_energy
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# Globals
# ============================================================
WARMUP_STEPS = 1


def load_data(csv_path):
    """Load hardware stats CSV"""
    df = pd.read_csv(csv_path)
    print(f"Columns found: {list(df.columns)}")
    return df


def has_power_data(df):
    """Check if DataFrame contains power/energy columns"""
    return ('energy_step_j' in df.columns
            and 'gpu_power_forward_start_w' in df.columns)


def steady_state(df):
    """Return DataFrame excluding warmup steps"""
    return df.iloc[WARMUP_STEPS:]


# ============================================================
# TIMING PLOTS
# ============================================================

def plot_phase_durations(df, output_dir, run_id):
    """Plot phase durations over training steps"""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df['step_num'], df['forward_time_ms'],
            label='Forward', linewidth=2, color='#3498db')
    ax.plot(df['step_num'], df['backward_time_ms'],
            label='Backward', linewidth=2, color='#e74c3c')
    ax.plot(df['step_num'], df['optimizer_time_ms'],
            label='Optimizer', linewidth=2, color='#2ecc71')
    ax.plot(df['step_num'], df['step_time_ms'],
            label='Total', linewidth=2, color='#9b59b6',
            linestyle='--', alpha=0.7)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Duration (ms)', fontsize=12)
    ax.set_title('Phase Durations During T5 Training', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"phase_durations_{run_id}.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  Saved: {filepath}")


def plot_phase_breakdown(df, output_dir, run_id):
    """Plot phase breakdown pie chart"""
    steady = steady_state(df)

    avg_forward = steady['forward_time_ms'].mean()
    avg_backward = steady['backward_time_ms'].mean()
    avg_optimizer = steady['optimizer_time_ms'].mean()

    sizes = [avg_forward, avg_backward, avg_optimizer]
    labels = [
        f'Forward\n{avg_forward:.1f}ms',
        f'Backward\n{avg_backward:.1f}ms',
        f'Optimizer\n{avg_optimizer:.1f}ms',
    ]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    explode = (0, 0.05, 0)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(sizes, labels=labels, colors=colors, explode=explode,
           autopct='%1.1f%%', startangle=90,
           textprops={'fontsize': 11})
    ax.set_title(
        'T5 Training Phase Breakdown — Time (Steady State)', fontsize=14
    )

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"phase_breakdown_{run_id}.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  Saved: {filepath}")


def plot_warmup_comparison(df, output_dir, run_id):
    """Compare warmup vs steady state"""
    warmup = df.iloc[0]
    steady = steady_state(df)

    metrics = [
        'step_time_ms', 'forward_time_ms',
        'backward_time_ms', 'optimizer_time_ms',
    ]
    labels = ['Total', 'Forward', 'Backward', 'Optimizer']

    warmup_vals = [warmup[m] for m in metrics]
    steady_vals = [steady[m].mean() for m in metrics]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, warmup_vals, width,
           label='Warmup (Step 0)', color='#e74c3c')
    ax.bar(x + width / 2, steady_vals, width,
           label='Steady State (Avg)', color='#2ecc71')

    ax.set_ylabel('Duration (ms)', fontsize=12)
    ax.set_title('T5 Warmup vs Steady State Performance', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(bottom=0)

    for i, (w, s) in enumerate(zip(warmup_vals, steady_vals)):
        overhead = w / s if s > 0 else 0
        ax.annotate(
            f'{overhead:.2f}x',
            xy=(i, max(w, s) + 15),
            ha='center', fontsize=10, color='gray',
        )

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"warmup_comparison_{run_id}.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================================
# HARDWARE PLOTS
# ============================================================

def plot_gpu_memory(df, output_dir, run_id):
    """Plot GPU memory usage over training steps"""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df['step_num'], df['gpu_memory_allocated_mb'],
            label='Allocated', linewidth=2, color='#2ecc71')
    ax.plot(df['step_num'], df['gpu_memory_reserved_mb'],
            label='Reserved', linewidth=2, color='#3498db', linestyle='--')
    ax.plot(df['step_num'], df['gpu_memory_peak_mb'],
            label='Peak', linewidth=2, color='#e74c3c', linestyle=':')

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('GPU Memory (MB)', fontsize=12)
    ax.set_title('GPU Memory Usage During T5 Training', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"gpu_memory_{run_id}.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  Saved: {filepath}")


def plot_gpu_utilization(df, output_dir, run_id):
    """Plot GPU utilization over training steps"""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df['step_num'], df['gpu_utilization'],
            linewidth=2, color='#e67e22', marker='o', markersize=2)

    steady = steady_state(df)
    avg_util = steady['gpu_utilization'].mean()
    ax.axhline(y=avg_util, color='red', linestyle='--',
               label=f'Avg (steady state): {avg_util:.1f}%')

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('GPU Utilization (%)', fontsize=12)
    ax.set_title('GPU Utilization During T5 Training', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"gpu_utilization_{run_id}.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================================
# ENERGY & POWER PLOTs
# ============================================================

def plot_gpu_power(df, output_dir, run_id):
    """Plot GPU power draw per phase over training steps"""
    df = df.copy()
    df['power_forward_w'] = (
        df['gpu_power_forward_start_w'] + df['gpu_power_forward_end_w']
    ) / 2.0
    df['power_backward_w'] = (
        df['gpu_power_backward_start_w'] + df['gpu_power_backward_end_w']
    ) / 2.0
    df['power_optimizer_w'] = (
        df['gpu_power_optimizer_start_w'] + df['gpu_power_optimizer_end_w']
    ) / 2.0

    steady = steady_state(df)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        df['step_num'], df['power_forward_w'],
        label=f"Forward (avg {steady['power_forward_w'].mean():.1f}W)",
        linewidth=1.5, color='#3498db', alpha=0.8,
    )
    ax.plot(
        df['step_num'], df['power_backward_w'],
        label=f"Backward (avg {steady['power_backward_w'].mean():.1f}W)",
        linewidth=1.5, color='#e74c3c', alpha=0.8,
    )
    ax.plot(
        df['step_num'], df['power_optimizer_w'],
        label=f"Optimizer (avg {steady['power_optimizer_w'].mean():.1f}W)",
        linewidth=1.5, color='#2ecc71', alpha=0.8,
    )

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Power (W)', fontsize=12)
    ax.set_title('GPU Power Draw Per Phase During T5 Training', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"gpu_power_{run_id}.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  Saved: {filepath}")


def plot_energy_breakdown(df, output_dir, run_id):
    """Plot energy breakdown pie chart"""
    steady = steady_state(df)

    avg_energy_fwd = steady['energy_forward_j'].mean() * 1000
    avg_energy_bwd = steady['energy_backward_j'].mean() * 1000
    avg_energy_opt = steady['energy_optimizer_j'].mean() * 1000

    sizes = [avg_energy_fwd, avg_energy_bwd, avg_energy_opt]
    labels = [
        f'Forward\n{avg_energy_fwd:.1f}mJ',
        f'Backward\n{avg_energy_bwd:.1f}mJ',
        f'Optimizer\n{avg_energy_opt:.1f}mJ',
    ]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    explode = (0, 0.05, 0)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(sizes, labels=labels, colors=colors, explode=explode,
           autopct='%1.1f%%', startangle=90,
           textprops={'fontsize': 11})
    ax.set_title(
        'T5 Training Phase Breakdown — Energy (Steady State)', fontsize=14
    )

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"energy_breakdown_{run_id}.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  Saved: {filepath}")


def plot_energy_per_step(df, output_dir, run_id):
    """Plot per-step energy with cumulative overlay"""
    fig, ax1 = plt.subplots(figsize=(12, 6))

    energy_mj = df['energy_step_j'] * 1000
    cumulative_j = df['energy_step_j'].cumsum()

    color1 = '#e74c3c'
    ax1.plot(df['step_num'], energy_mj, color=color1, linewidth=1.5,
             alpha=0.8, label='Per-step energy')
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Energy per Step (mJ)', fontsize=12, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    steady = steady_state(df)
    avg_energy_mj = steady['energy_step_j'].mean() * 1000
    ax1.axhline(y=avg_energy_mj, color=color1, linestyle='--', alpha=0.5,
                label=f'Steady avg: {avg_energy_mj:.1f} mJ/step')

    ax2 = ax1.twinx()
    color2 = '#3498db'
    ax2.plot(df['step_num'], cumulative_j, color=color2, linewidth=2,
             linestyle='--', alpha=0.7, label='Cumulative energy')
    ax2.set_ylabel('Cumulative Energy (J)', fontsize=12, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    ax1.set_title('Energy Consumption During T5 Training', fontsize=14)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"energy_per_step_{run_id}.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  Saved: {filepath}")


def plot_temperature(df, output_dir, run_id):
    """Plot GPU temperature over training steps"""
    if 'gpu_temperature_c' not in df.columns:
        print("  Skipping temperature plot (column not found)")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df['step_num'], df['gpu_temperature_c'],
            linewidth=2, color='#e67e22')

    steady = steady_state(df)
    avg_temp = steady['gpu_temperature_c'].mean()
    ax.axhline(y=avg_temp, color='red', linestyle='--', alpha=0.5,
               label=f'Avg (steady): {avg_temp:.1f}°C')

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title('GPU Temperature During T5 Training', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"gpu_temperature_{run_id}.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  Saved: {filepath}")


def plot_time_vs_energy_comparison(df, output_dir, run_id):
    """Side-by-side time% vs energy% plot"""
    steady = steady_state(df)

    t_fwd = steady['forward_time_ms'].mean()
    t_bwd = steady['backward_time_ms'].mean()
    t_opt = steady['optimizer_time_ms'].mean()

    e_fwd = steady['energy_forward_j'].mean() * 1000
    e_bwd = steady['energy_backward_j'].mean() * 1000
    e_opt = steady['energy_optimizer_j'].mean() * 1000

    t_total = t_fwd + t_bwd + t_opt
    e_total = e_fwd + e_bwd + e_opt

    phases = ['Forward', 'Backward', 'Optimizer']
    time_pcts = [
        t_fwd / t_total * 100,
        t_bwd / t_total * 100,
        t_opt / t_total * 100,
    ]
    energy_pcts = [
        e_fwd / e_total * 100,
        e_bwd / e_total * 100,
        e_opt / e_total * 100,
    ]

    x = np.arange(len(phases))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, time_pcts, width,
                   label='Time (%)', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width / 2, energy_pcts, width,
                   label='Energy (%)', color='#e74c3c', alpha=0.8)

    for bar, val in zip(bars1, time_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=10, color='#2c3e50')
    for bar, val in zip(bars2, energy_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=10, color='#2c3e50')

    ax.set_ylabel('Proportion (%)', fontsize=12)
    ax.set_title(
        'T5 Phase Breakdown: Time vs Energy (Steady State)', fontsize=14
    )
    ax.set_xticks(x)
    ax.set_xticklabels(phases, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 70)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"time_vs_energy_{run_id}.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  Saved: {filepath}")


def plot_warmup_energy_comparison(df, output_dir, run_id):
    """Compare warmup vs steady state in both time and energy"""
    warmup = df.iloc[0]
    steady = steady_state(df)

    labels = ['Total', 'Forward', 'Backward', 'Optimizer']

    time_metrics = [
        'step_time_ms', 'forward_time_ms',
        'backward_time_ms', 'optimizer_time_ms',
    ]
    energy_metrics = [
        'energy_step_j', 'energy_forward_j',
        'energy_backward_j', 'energy_optimizer_j',
    ]

    warmup_time = [warmup[m] for m in time_metrics]
    steady_time = [steady[m].mean() for m in time_metrics]
    warmup_energy = [warmup[m] * 1000 for m in energy_metrics]
    steady_energy = [steady[m].mean() * 1000 for m in energy_metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(len(labels))
    width = 0.35

    # Time comparison
    ax1.bar(x - width / 2, warmup_time, width,
            label='Warmup', color='#e74c3c')
    ax1.bar(x + width / 2, steady_time, width,
            label='Steady State', color='#2ecc71')
    ax1.set_ylabel('Duration (ms)', fontsize=12)
    ax1.set_title('Time: Warmup vs Steady State', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(bottom=0)
    for i, (w, s) in enumerate(zip(warmup_time, steady_time)):
        ratio = w / s if s > 0 else 0
        ax1.annotate(f'{ratio:.2f}x',
                     xy=(i, max(w, s) + 10),
                     ha='center', fontsize=10, color='gray')

    # Energy comparison
    ax2.bar(x - width / 2, warmup_energy, width,
            label='Warmup', color='#e74c3c')
    ax2.bar(x + width / 2, steady_energy, width,
            label='Steady State', color='#2ecc71')
    ax2.set_ylabel('Energy (mJ)', fontsize=12)
    ax2.set_title('Energy: Warmup vs Steady State', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(bottom=0)
    for i, (w, s) in enumerate(zip(warmup_energy, steady_energy)):
        ratio = w / s if s > 0 else 0
        ax2.annotate(f'{ratio:.2f}x',
                     xy=(i, max(w, s) + 1),
                     ha='center', fontsize=10, color='gray')

    plt.suptitle('T5 Warmup Overhead: Time vs Energy',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    filepath = os.path.join(
        output_dir, f"warmup_energy_comparison_{run_id}.png"
    )
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================================
# TEXT SUMMARY
# ============================================================

def print_summary(df):
    """print text summary of key findings"""
    steady = steady_state(df)
    power = has_power_data(df)

    print("\n" + "=" * 60)
    print("SUMMARY OF KEY FINDINGS")
    print("=" * 60)

    print(f"\nTotal steps: {len(df)} "
          f"({WARMUP_STEPS} warmup, {len(steady)} steady-state)")

    # ---- Timing ----
    print(f"\n--- Timing (steady state) ---")
    for phase, col in [
        ('Forward', 'forward_time_ms'),
        ('Backward', 'backward_time_ms'),
        ('Optimizer', 'optimizer_time_ms'),
        ('Total step', 'step_time_ms'),
    ]:
        avg = steady[col].mean()
        std = steady[col].std()
        print(f"  {phase:12s}: {avg:7.2f} +/- {std:.2f} ms")

    phase_sum = (
        steady['forward_time_ms']
        + steady['backward_time_ms']
        + steady['optimizer_time_ms']
    ).mean()
    step_avg = steady['step_time_ms'].mean()
    gap = step_avg - phase_sum
    print(f"  {'Data xfer':12s}: {gap:7.2f} ms (step - phases)")

    # ---- Memory ----
    print(f"\n--- GPU Memory (steady state) ---")
    print(f"  Allocated: {steady['gpu_memory_allocated_mb'].mean():.0f} MB")
    print(f"  Reserved:  {steady['gpu_memory_reserved_mb'].mean():.0f} MB")
    print(f"  Peak:      {steady['gpu_memory_peak_mb'].mean():.0f} MB")

    # ---- Utilization ----
    print(f"\n--- GPU Utilization (steady state) ---")
    print(f"  Compute: {steady['gpu_utilization'].mean():.1f}%")
    if 'gpu_memory_utilization' in df.columns:
        mem_util = steady['gpu_memory_utilization'].mean()
        print(f"  Memory bandwidth: {mem_util:.1f}%")

    # ---- Power ----
    if power:
        print(f"\n--- Power (steady state) ---")
        for phase, sc, ec in [
            ('Forward',
             'gpu_power_forward_start_w', 'gpu_power_forward_end_w'),
            ('Backward',
             'gpu_power_backward_start_w', 'gpu_power_backward_end_w'),
            ('Optimizer',
             'gpu_power_optimizer_start_w', 'gpu_power_optimizer_end_w'),
        ]:
            avg_p = ((steady[sc] + steady[ec]) / 2).mean()
            std_p = ((steady[sc] + steady[ec]) / 2).std()
            print(f"  {phase:12s}: {avg_p:6.1f} +/- {std_p:.1f} W")

    # ---- Energy ----
    if power:
        print(f"\n--- Energy ---")
        total_e = df['energy_step_j'].sum()
        avg_e = steady['energy_step_j'].mean() * 1000
        warmup_e = df.iloc[0]['energy_step_j'] * 1000
        print(f"  Total training:  {total_e:.3f} J "
              f"({total_e / 3600 * 1000:.4f} mWh)")
        print(f"  Avg per step:    {avg_e:.2f} mJ")
        print(f"  Warmup step:     {warmup_e:.2f} mJ")

    # ---- Temperature ----
    if power and 'gpu_temperature_c' in df.columns:
        print(f"\n--- Temperature ---")
        print(f"  Avg: {steady['gpu_temperature_c'].mean():.1f} C")
        print(f"  Min: {steady['gpu_temperature_c'].min():.1f} C")
        print(f"  Max: {steady['gpu_temperature_c'].max():.1f} C")

    # ---- Warmup Overhead ----
    print(f"\n--- Warmup Overhead (Step 0 vs Steady State) ---")
    warmup = df.iloc[0]
    for phase, col in [
        ('Forward', 'forward_time_ms'),
        ('Backward', 'backward_time_ms'),
        ('Optimizer', 'optimizer_time_ms'),
        ('Total step', 'step_time_ms'),
    ]:
        w_val = warmup[col]
        s_val = steady[col].mean()
        ratio = w_val / s_val if s_val > 0 else 0
        print(f"  {phase:12s}: {w_val:7.2f} ms vs {s_val:.2f} ms "
              f"({ratio:.2f}x)")

    if power:
        print(f"\n  Energy overhead:")
        for phase, col in [
            ('Forward', 'energy_forward_j'),
            ('Backward', 'energy_backward_j'),
            ('Optimizer', 'energy_optimizer_j'),
            ('Total step', 'energy_step_j'),
        ]:
            w_val = warmup[col] * 1000
            s_val = steady[col].mean() * 1000
            ratio = w_val / s_val if s_val > 0 else 0
            print(f"  {phase:12s}: {w_val:7.2f} mJ vs {s_val:.2f} mJ "
                  f"({ratio:.2f}x)")

    print("\n" + "=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Plot hardware monitoring results'
    )
    parser.add_argument(
        '--csv', type=str, required=True,
        help='Path to hardware_stats CSV',
    )
    parser.add_argument(
        '--output', type=str, default='./plots',
        help='Output directory for plots',
    )
    parser.add_argument(
        '--run_id', type=str, default='t5_hardware',
        help='Run identifier for filenames',
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading data from: {args.csv}")
    df = load_data(args.csv)
    print(f"Loaded {len(df)} steps")

    power = has_power_data(df)
    if power:
        print("Power/energy columns detected — generating full plot suite")
    else:
        print("No power/energy columns — generating timing + hardware plots only")

    print(f"\nGenerating plots in: {args.output}\n")

    # ---- Always generate these ----
    print("Timing plots:")
    plot_phase_durations(df, args.output, args.run_id)
    plot_phase_breakdown(df, args.output, args.run_id)
    plot_warmup_comparison(df, args.output, args.run_id)

    print("\nHardware plots:")
    plot_gpu_memory(df, args.output, args.run_id)
    plot_gpu_utilization(df, args.output, args.run_id)

    # ---- Generate if power data exists ----
    if power:
        print("\nEnergy plots:")
        plot_gpu_power(df, args.output, args.run_id)
        plot_energy_breakdown(df, args.output, args.run_id)
        plot_energy_per_step(df, args.output, args.run_id)
        plot_temperature(df, args.output, args.run_id)
        plot_time_vs_energy_comparison(df, args.output, args.run_id)
        plot_warmup_energy_comparison(df, args.output, args.run_id)

    # ---- Text summary ----
    print_summary(df)

    print(f"\n{'=' * 40}")
    print(f"All plots generated in: {args.output}")
    total_plots = 5 + (6 if power else 0)
    print(f"Total plots: {total_plots}")
    print(f"{'=' * 40}")


if __name__ == '__main__':
    main()
