"""
Plot hardware monitoring results for T5 energy analysis v.2

Generates 9 plots from hardware_stats v2 CSV:

  1. Phase Durations         — line plot, all phases over training steps
  2. Time vs Energy          — grouped bar, phase proportions side-by-side
  3. GPU Power per Phase     — line plot, shows warmup ramp + steady state
  4. Warmup Analysis         — dual-panel: time (slower) vs energy (cheaper)
  5. Energy per Step         — dual-axis: per-step + cumulative
  6. CO2 Emissions           — dual-axis: per-step + cumulative
  7. GPU Memory              — line plot: allocated / reserved / peak
  8. GPU Utilization         — line plot with steady-state average
  9. GPU Temperature         — line plot with thermal rise annotation

Usage:
    python scripts/analysis/plot_hardware.py --csv <path> --output <dir>
    python scripts/analysis/plot_hardware.py --csv <path> --output <dir> --run_id t5_v2
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# Configuration
# ============================================================
WARMUP_STEPS = 6

# Consistent colour scheme across all plots
C_DATA_TRANSFER = '#f39c12'  # amber
C_FORWARD       = '#3498db'  # blue
C_BACKWARD      = '#e74c3c'  # red
C_OPTIMIZER     = '#2ecc71'  # green
C_OVERHEAD      = '#95a5a6'  # grey
C_TOTAL         = '#9b59b6'  # purple
C_WARMUP        = '#e74c3c'
C_STEADY        = '#2ecc71'
C_ENERGY        = '#e74c3c'
C_CUMULATIVE    = '#3498db'
C_CO2           = '#27ae60'
C_CO2_CUM       = '#2c3e50'
C_TEMPERATURE   = '#e67e22'
C_UTILIZATION   = '#e67e22'


# ============================================================
# Helpers
# ============================================================

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    print(f"  Columns: {len(df.columns)}, Rows: {len(df)}")
    return df


def steady(df):
    return df.iloc[WARMUP_STEPS:]


def has_v2_columns(df):
    """Check CSV format (data_transfer, overhead, co2)"""
    return all(c in df.columns for c in [
        'data_transfer_time_ms', 'overhead_time_ms',
        'energy_overhead_j', 'co2_step_mg',
    ])


def savefig(fig, output_dir, name, run_id):
    path = os.path.join(output_dir, f"{name}_{run_id}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {path}")


# ============================================================
# Plot 1: Phase Durations over Training Steps
# ============================================================

def plot_phase_durations(df, output_dir, run_id):
    """All phase durations as time series. Shows CUDA warmup spike
    at step 0 and steady-state stability"""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df['step_num'], df['forward_time_ms'],
            label='Forward', lw=2, color=C_FORWARD)
    ax.plot(df['step_num'], df['backward_time_ms'],
            label='Backward', lw=2, color=C_BACKWARD)
    ax.plot(df['step_num'], df['optimizer_time_ms'],
            label='Optimizer', lw=2, color=C_OPTIMIZER)

    if 'overhead_time_ms' in df.columns:
        ax.plot(df['step_num'], df['overhead_time_ms'],
                label='Overhead', lw=1.5, color=C_OVERHEAD, linestyle=':')

    ax.plot(df['step_num'], df['step_time_ms'],
            label='Total Step', lw=2, color=C_TOTAL, linestyle='--', alpha=0.7)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Duration (ms)', fontsize=12)
    ax.set_title('Phase Durations During T5 Training', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Annotate step 0 spike
    step0_total = df['step_time_ms'].iloc[0]
    ss_total = steady(df)['step_time_ms'].mean()
    ax.annotate(
        f'Step 0: {step0_total:.0f}ms\n({step0_total/ss_total:.1f}× steady)',
        xy=(0, step0_total), xytext=(15, step0_total - 30),
        fontsize=9, color='#7f8c8d',
        arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=0.8),
    )

    plt.tight_layout()
    savefig(fig, output_dir, 'phase_durations', run_id)


# ============================================================
# Plot 2: Time vs Energy — Grouped Bar (replaces both pie charts)
# ============================================================

def plot_time_vs_energy(df, output_dir, run_id):
    """Side-by-side time% and energy% for all phases.
    Key insight: near-identical proportions because GPU power is
    constant across phases (~214W)"""
    ss = steady(df)

    # Compute averages
    t_dt  = ss['data_transfer_time_ms'].mean() if 'data_transfer_time_ms' in df.columns else 0
    t_fwd = ss['forward_time_ms'].mean()
    t_bwd = ss['backward_time_ms'].mean()
    t_opt = ss['optimizer_time_ms'].mean()
    t_oh  = ss['overhead_time_ms'].mean() if 'overhead_time_ms' in df.columns else 0

    e_dt  = ss['energy_data_transfer_j'].mean() * 1000 if 'energy_data_transfer_j' in df.columns else 0
    e_fwd = ss['energy_forward_j'].mean() * 1000
    e_bwd = ss['energy_backward_j'].mean() * 1000
    e_opt = ss['energy_optimizer_j'].mean() * 1000
    e_oh  = ss['energy_overhead_j'].mean() * 1000 if 'energy_overhead_j' in df.columns else 0

    # Only show phases with >0.5% contribution to avoid clutter
    t_total = t_dt + t_fwd + t_bwd + t_opt + t_oh
    e_total = e_dt + e_fwd + e_bwd + e_opt + e_oh

    phases, time_pcts, energy_pcts, colors = [], [], [], []
    for label, t_val, e_val, color in [
        ('Data\nTransfer', t_dt, e_dt, C_DATA_TRANSFER),
        ('Forward',        t_fwd, e_fwd, C_FORWARD),
        ('Backward',       t_bwd, e_bwd, C_BACKWARD),
        ('Optimizer',      t_opt, e_opt, C_OPTIMIZER),
        ('Overhead',       t_oh, e_oh, C_OVERHEAD),
    ]:
        t_pct = t_val / t_total * 100 if t_total > 0 else 0
        e_pct = e_val / e_total * 100 if e_total > 0 else 0
        if t_pct >= 0.5 or e_pct >= 0.5:
            phases.append(label)
            time_pcts.append(t_pct)
            energy_pcts.append(e_pct)
            colors.append(color)

    x = np.arange(len(phases))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_t = ax.bar(x - width/2, time_pcts, width,
                    label='Time (%)', color=C_FORWARD, alpha=0.8)
    bars_e = ax.bar(x + width/2, energy_pcts, width,
                    label='Energy (%)', color=C_ENERGY, alpha=0.8)

    # Value labels
    for bar, val in zip(bars_t, time_pcts):
        if val >= 1.0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars_e, energy_pcts):
        if val >= 1.0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Proportion (%)', fontsize=12)
    ax.set_title('T5 Phase Breakdown: Time vs Energy (Steady State)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(phases, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(max(time_pcts), max(energy_pcts)) + 10)

    # Annotation: key insight
    ax.text(0.98, 0.95,
            'Time ≈ Energy because GPU power\n'
            'is constant across phases (~214W)',
            transform=ax.transAxes, fontsize=9, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    savefig(fig, output_dir, 'time_vs_energy', run_id)


# ============================================================
# Plot 3: GPU Power per Phase over Training Steps
# ============================================================

def plot_gpu_power(df, output_dir, run_id):
    """Per-phase average power. Shows power warmup ramp (steps 0-5)
    decoupled from timing warmup (step 0 only)"""
    df = df.copy()
    df['power_forward_w'] = (
        df['gpu_power_forward_start_w'] + df['gpu_power_forward_end_w']) / 2
    df['power_backward_w'] = (
        df['gpu_power_backward_start_w'] + df['gpu_power_backward_end_w']) / 2
    df['power_optimizer_w'] = (
        df['gpu_power_optimizer_start_w'] + df['gpu_power_optimizer_end_w']) / 2

    ss = steady(df)

    fig, ax = plt.subplots(figsize=(12, 6))
    for col, label, color in [
        ('power_forward_w',   f"Forward (avg {ss['power_forward_w'].mean():.1f}W)",   C_FORWARD),
        ('power_backward_w',  f"Backward (avg {ss['power_backward_w'].mean():.1f}W)",  C_BACKWARD),
        ('power_optimizer_w', f"Optimizer (avg {ss['power_optimizer_w'].mean():.1f}W)", C_OPTIMIZER),
    ]:
        ax.plot(df['step_num'], df[col], label=label, lw=1.5, color=color, alpha=0.8)

    # Warmup region shading
    ax.axvspan(-0.5, WARMUP_STEPS - 0.5, alpha=0.08, color='red', label='Power warmup')

    # NVML caveat
    ax.text(0.98, 0.05,
            'Note: NVML refresh ~50–100ms;\n'
            'phases <50ms have aliased readings',
            transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
            color='#7f8c8d', style='italic')

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Power (W)', fontsize=12)
    ax.set_title('GPU Power Draw Per Phase During T5 Training', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    savefig(fig, output_dir, 'gpu_power', run_id)


# ============================================================
# Plot 4: Warmup Analysis — Dual Panel
# ============================================================

def plot_warmup_analysis(df, output_dir, run_id):
    """Time (warmup slower) vs energy (warmup cheaper).
    """
    warmup = df.iloc[0]
    ss = steady(df)

    labels = ['Total', 'Forward', 'Backward', 'Optimizer']
    time_cols = ['step_time_ms', 'forward_time_ms',
                 'backward_time_ms', 'optimizer_time_ms']
    energy_cols = ['energy_step_j', 'energy_forward_j',
                   'energy_backward_j', 'energy_optimizer_j']

    w_time = [warmup[c] for c in time_cols]
    s_time = [ss[c].mean() for c in time_cols]
    w_energy = [warmup[c] * 1000 for c in energy_cols]
    s_energy = [ss[c].mean() * 1000 for c in energy_cols]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    x = np.arange(len(labels))
    width = 0.35

    # Left: Time
    ax1.bar(x - width/2, w_time, width, label='Warmup (Step 0)', color=C_WARMUP)
    ax1.bar(x + width/2, s_time, width, label='Steady State (Avg)', color=C_STEADY)
    ax1.set_ylabel('Duration (ms)', fontsize=12)
    ax1.set_title('Time: Warmup is Slower (CUDA JIT)', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(bottom=0)
    for i, (w, s) in enumerate(zip(w_time, s_time)):
        ratio = w / s if s > 0 else 0
        ax1.annotate(f'{ratio:.2f}×', xy=(i, max(w, s) + 10),
                     ha='center', fontsize=9, color='#c0392b')

    # Right: Energy
    ax2.bar(x - width/2, w_energy, width, label='Warmup (Step 0)', color=C_WARMUP)
    ax2.bar(x + width/2, s_energy, width, label='Steady State (Avg)', color=C_STEADY)
    ax2.set_ylabel('Energy (mJ)', fontsize=12)
    ax2.set_title('Energy: Warmup Uses Less (GPU at idle power)', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(bottom=0)
    for i, (w, s) in enumerate(zip(w_energy, s_energy)):
        pct = (1 - w / s) * 100 if s > 0 else 0
        ax2.annotate(f'{pct:.0f}% less', xy=(i, max(w, s) + 800),
                     ha='center', fontsize=9, color='#27ae60')

    fig.suptitle('T5 Warmup: Slower but More Energy-Efficient',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    savefig(fig, output_dir, 'warmup_analysis', run_id)


# ============================================================
# Plot 5: Energy per Step + Cumulative
# ============================================================

def plot_energy_per_step(df, output_dir, run_id):
    """Per-step energy with cumulative overlay and steady-state average"""
    fig, ax1 = plt.subplots(figsize=(12, 6))

    energy_mj = df['energy_step_j'] * 1000
    cumulative_j = df['energy_step_j'].cumsum()
    ss = steady(df)
    avg_mj = ss['energy_step_j'].mean() * 1000

    ax1.plot(df['step_num'], energy_mj, color=C_ENERGY, lw=1.5,
             alpha=0.8, label='Per-step energy')
    ax1.axhline(y=avg_mj, color=C_ENERGY, linestyle='--', alpha=0.5,
                label=f'Steady avg: {avg_mj:.1f} mJ/step')

    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Energy per Step (mJ)', fontsize=12, color=C_ENERGY)
    ax1.tick_params(axis='y', labelcolor=C_ENERGY)

    ax2 = ax1.twinx()
    ax2.plot(df['step_num'], cumulative_j, color=C_CUMULATIVE,
             lw=2, linestyle='--', alpha=0.7, label='Cumulative energy')
    ax2.set_ylabel('Cumulative Energy (J)', fontsize=12, color=C_CUMULATIVE)
    ax2.tick_params(axis='y', labelcolor=C_CUMULATIVE)

    # Total annotation
    total_j = cumulative_j.iloc[-1]
    ax2.annotate(f'Total: {total_j:.0f} J ({total_j/3600:.2f} Wh)',
                 xy=(df['step_num'].iloc[-1], total_j),
                 xytext=(-120, -20), textcoords='offset points',
                 fontsize=10, color=C_CUMULATIVE,
                 arrowprops=dict(arrowstyle='->', color=C_CUMULATIVE))

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)

    ax1.set_title('Energy Consumption During T5 Training', fontsize=14)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    savefig(fig, output_dir, 'energy_per_step', run_id)


# ============================================================
# Plot 6: CO2 Emissions per Step + Cumulative
# ============================================================

def plot_co2_emissions(df, output_dir, run_id):
    """CO2 emissions per step with cumulative overlay"""
    if 'co2_step_mg' not in df.columns:
        print("    Skipping CO2 plot (column not found)")
        return

    fig, ax1 = plt.subplots(figsize=(12, 6))

    co2_mg = df['co2_step_mg']
    cumulative_mg = co2_mg.cumsum()
    ss = steady(df)
    avg_co2 = ss['co2_step_mg'].mean()

    ax1.plot(df['step_num'], co2_mg, color=C_CO2, lw=1.5,
             alpha=0.8, label='Per-step CO₂')
    ax1.axhline(y=avg_co2, color=C_CO2, linestyle='--', alpha=0.5,
                label=f'Steady avg: {avg_co2:.4f} mg/step')
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('CO₂ per Step (mg)', fontsize=12, color=C_CO2)
    ax1.tick_params(axis='y', labelcolor=C_CO2)

    ax2 = ax1.twinx()
    ax2.plot(df['step_num'], cumulative_mg, color=C_CO2_CUM,
             lw=2, linestyle='--', alpha=0.7, label='Cumulative CO₂')
    ax2.set_ylabel('Cumulative CO₂ (mg)', fontsize=12, color=C_CO2_CUM)
    ax2.tick_params(axis='y', labelcolor=C_CO2_CUM)

    # Total annotation
    total_mg = cumulative_mg.iloc[-1]
    ax2.annotate(f'Total: {total_mg:.2f} mg\n({total_mg/1000:.4f} g CO₂eq)',
                 xy=(df['step_num'].iloc[-1], total_mg),
                 xytext=(-140, -25), textcoords='offset points',
                 fontsize=10, color=C_CO2_CUM,
                 arrowprops=dict(arrowstyle='->', color=C_CO2_CUM))

    # Carbon intensity note
    ax1.text(0.02, 0.95,
             'Carbon intensity: Quebec (Hydro-Québec)\n≈ 30 gCO₂eq/kWh',
             transform=ax1.transAxes, fontsize=9, va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='honeydew', alpha=0.8))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)

    ax1.set_title('CO₂ Emissions During T5 Training', fontsize=14)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    savefig(fig, output_dir, 'co2_emissions', run_id)


# ============================================================
# Plot 7: GPU Memory
# ============================================================

def plot_gpu_memory(df, output_dir, run_id):
    """GPU memory: allocated vs reserved vs peak.
    Distinguishes true usage from caching allocator overhead"""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df['step_num'], df['gpu_memory_allocated_mb'],
            label='Allocated (actual tensors)', lw=2, color=C_OPTIMIZER)
    ax.plot(df['step_num'], df['gpu_memory_reserved_mb'],
            label='Reserved (caching allocator)', lw=2,
            color=C_FORWARD, linestyle='--')
    ax.plot(df['step_num'], df['gpu_memory_peak_mb'],
            label='Peak (max during step)', lw=2,
            color=C_BACKWARD, linestyle=':')

    # VRAM capacity line
    ax.axhline(y=32768, color='black', linestyle='--', alpha=0.3, lw=1)
    ax.text(df['step_num'].iloc[-1], 32768 + 300, '32 GB VRAM',
            ha='right', fontsize=9, color='#7f8c8d')

    ss = steady(df)
    alloc = ss['gpu_memory_allocated_mb'].mean()
    peak = ss['gpu_memory_peak_mb'].mean()
    reserved = ss['gpu_memory_reserved_mb'].mean()
    ax.text(0.02, 0.55,
            f'Steady state:\n'
            f'  Allocated: {alloc/1024:.1f} GB ({alloc/32768*100:.0f}% of VRAM)\n'
            f'  Peak:      {peak/1024:.1f} GB ({peak/32768*100:.0f}% of VRAM)\n'
            f'  Reserved:  {reserved/1024:.1f} GB ({reserved/32768*100:.0f}% of VRAM)',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8),
            family='monospace')

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('GPU Memory (MB)', fontsize=12)
    ax.set_title('GPU Memory Usage During T5 Training', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    savefig(fig, output_dir, 'gpu_memory', run_id)


# ============================================================
# Plot 8: GPU Utilization
# ============================================================

def plot_gpu_utilization(df, output_dir, run_id):
    """GPU compute utilization with steady-state average"""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df['step_num'], df['gpu_utilization'],
            lw=2, color=C_UTILIZATION, marker='o', markersize=2)

    ss = steady(df)
    avg = ss['gpu_utilization'].mean()
    ax.axhline(y=avg, color='red', linestyle='--',
               label=f'Steady avg: {avg:.1f}%')

    ax.text(0.98, 0.15,
            'Synthetic data → zero I/O bottleneck\n→ near-100% GPU utilization',
            transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('GPU Utilization (%)', fontsize=12)
    ax.set_title('GPU Compute Utilization During T5 Training', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    savefig(fig, output_dir, 'gpu_utilization', run_id)


# ============================================================
# Plot 9: GPU Temperature
# ============================================================

def plot_gpu_temperature(df, output_dir, run_id):
    """GPU temperature over training. Monotonic rise = thermal
    equilibrium not reached in the short run"""
    if 'gpu_temperature_c' not in df.columns:
        print("    Skipping temperature plot (column not found)")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df['step_num'], df['gpu_temperature_c'],
            lw=2, color=C_TEMPERATURE)

    t_start = df['gpu_temperature_c'].iloc[0]
    t_end = df['gpu_temperature_c'].iloc[-1]

    ax.annotate(f'{t_start:.0f}°C',
                xy=(0, t_start), xytext=(15, -20),
                textcoords='offset points', fontsize=11,
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax.annotate(f'{t_end:.0f}°C',
                xy=(df['step_num'].iloc[-1], t_end),
                xytext=(-40, -20), textcoords='offset points', fontsize=11,
                arrowprops=dict(arrowstyle='->', color='gray'))

    total_time_s = df['step_time_ms'].sum() / 1000
    rate = (t_end - t_start) / total_time_s if total_time_s > 0 else 0
    ax.text(0.02, 0.95,
            f'Rise: {t_start:.0f}°C → {t_end:.0f}°C '
            f'(+{t_end - t_start:.0f}°C in ~{total_time_s:.0f}s, '
            f'{rate:.1f}°C/s)\n'
            f'Thermal equilibrium not reached',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title('GPU Temperature During T5 Training', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    savefig(fig, output_dir, 'gpu_temperature', run_id)


# ============================================================
# Text Summary
# ============================================================

def print_summary(df):
    """Console summary matching v2 CSV format"""
    ss = steady(df)
    v2 = has_v2_columns(df)

    print("\n" + "=" * 65)
    print("PLOT SUMMARY — KEY NUMBERS")
    print("=" * 65)
    print(f"Steps: {len(df)} total, {WARMUP_STEPS} warmup, {len(ss)} steady")

    # Timing
    print(f"\n--- Timing (steady state) ---")
    cols = [('Forward', 'forward_time_ms'), ('Backward', 'backward_time_ms'),
            ('Optimizer', 'optimizer_time_ms')]
    if v2:
        cols = [('Data xfer', 'data_transfer_time_ms')] + cols + [('Overhead', 'overhead_time_ms')]
    cols.append(('Total', 'step_time_ms'))
    for label, col in cols:
        avg = ss[col].mean()
        pct = avg / ss['step_time_ms'].mean() * 100
        print(f"  {label:12s}: {avg:7.2f} ms  ({pct:5.1f}%)")

    # Energy
    if 'energy_step_j' in df.columns:
        total_j = df['energy_step_j'].sum()
        avg_mj = ss['energy_step_j'].mean() * 1000
        print(f"\n--- Energy ---")
        print(f"  Total:   {total_j:.1f} J ({total_j/3600:.4f} Wh)")
        print(f"  Avg/step: {avg_mj:.1f} mJ (steady)")

    # CO2
    if 'co2_step_mg' in df.columns:
        total_co2 = df['co2_step_mg'].sum()
        print(f"\n--- CO₂ (30 gCO₂eq/kWh) ---")
        print(f"  Total:   {total_co2:.3f} mg ({total_co2/1000:.6f} g)")

    # Memory
    print(f"\n--- GPU Memory (steady state) ---")
    print(f"  Allocated: {ss['gpu_memory_allocated_mb'].mean()/1024:.1f} GB")
    print(f"  Peak:      {ss['gpu_memory_peak_mb'].mean()/1024:.1f} GB")
    print(f"  Reserved:  {ss['gpu_memory_reserved_mb'].mean()/1024:.1f} GB")

    print("=" * 65)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Plot hardware monitoring results (v2)')
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to hardware_stats CSV')
    parser.add_argument('--output', type=str, default='./plots',
                        help='Output directory for plots')
    parser.add_argument('--run_id', type=str, default='t5_v2',
                        help='Run identifier for filenames')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f"\nLoading: {args.csv}")
    df = load_data(args.csv)

    v2 = has_v2_columns(df)
    has_power = 'energy_step_j' in df.columns
    print(f"  Format: {'v2' if v2 else 'v1'}, "
          f"Power data: {'yes' if has_power else 'no'}")
    print(f"  Output: {args.output}\n")

    print("Generating plots:")

    print("  [1/9] Phase Durations")
    plot_phase_durations(df, args.output, args.run_id)

    if has_power:
        print("  [2/9] Time vs Energy")
        plot_time_vs_energy(df, args.output, args.run_id)

        print("  [3/9] GPU Power")
        plot_gpu_power(df, args.output, args.run_id)

        print("  [4/9] Warmup Analysis")
        plot_warmup_analysis(df, args.output, args.run_id)

        print("  [5/9] Energy per Step")
        plot_energy_per_step(df, args.output, args.run_id)

        print("  [6/9] CO₂ Emissions")
        plot_co2_emissions(df, args.output, args.run_id)

    print("  [7/9] GPU Memory")
    plot_gpu_memory(df, args.output, args.run_id)

    print("  [8/9] GPU Utilization")
    plot_gpu_utilization(df, args.output, args.run_id)

    if has_power:
        print("  [9/9] GPU Temperature")
        plot_gpu_temperature(df, args.output, args.run_id)

    print_summary(df)

    n_plots = 4 + (5 if has_power else 0)
    print(f"\nDone — {n_plots} plots saved to {args.output}/")


if __name__ == '__main__':
    main()
