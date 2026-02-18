"""
Plot our hardware monitoring results

Generates plots from hardware_stats CSV files, currently:
1. GPU Memory Usage vs Training Step
2. Phase Duration vs Training Step  
3. GPU Utilization vs Training Step
4. Phase Breakdown Pie Chart
5. Warmup Comparison Bar Chart

Usage:
    python scripts/analysis/plot_hardware.py --csv <path_to_csv> --output <output_dir>
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load_data(csv_path):
    """Load hardware stats CSV"""
    return pd.read_csv(csv_path)

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
    print(f"Saved: {filepath}")

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
            label='Total', linewidth=2, color='#9b59b6', linestyle='--', alpha=0.7)
    
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
    print(f"Saved: {filepath}")

def plot_gpu_utilization(df, output_dir, run_id):
    """Plot GPU utilization over training steps"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df['step_num'], df['gpu_utilization'], 
            linewidth=2, color='#e67e22', marker='o', markersize=2)
    
    avg_util = df['gpu_utilization'].iloc[1:].mean()  # Exclude warmup
    ax.axhline(y=avg_util, color='red', linestyle='--', 
               label=f'Avg (steady state): {avg_util:.1f}%')
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('GPU Utilization (%)', fontsize=12)
    ax.set_title('GPU Utilization During T5 Training', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, f"gpu_utilization_{run_id}.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Saved: {filepath}")

def plot_phase_breakdown(df, output_dir, run_id):
    """Plot phase breakdown pie chart"""
    # Exclude warmup step
    steady = df.iloc[1:]
    
    avg_forward = steady['forward_time_ms'].mean()
    avg_backward = steady['backward_time_ms'].mean()
    avg_optimizer = steady['optimizer_time_ms'].mean()
    
    sizes = [avg_forward, avg_backward, avg_optimizer]
    labels = [f'Forward\n{avg_forward:.1f}ms', 
              f'Backward\n{avg_backward:.1f}ms', 
              f'Optimizer\n{avg_optimizer:.1f}ms']
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    explode = (0, 0.05, 0)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, explode=explode,
        autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11}
    )
    
    ax.set_title('T5 Training Phase Breakdown (Steady State)', fontsize=14)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, f"phase_breakdown_{run_id}.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Saved: {filepath}")

def plot_warmup_comparison(df, output_dir, run_id):
    """Compare warmup vs steady state"""
    warmup = df.iloc[0]
    steady = df.iloc[1:]
    
    metrics = ['step_time_ms', 'forward_time_ms', 'backward_time_ms', 'optimizer_time_ms']
    labels = ['Total', 'Forward', 'Backward', 'Optimizer']
    
    warmup_vals = [warmup[m] for m in metrics]
    steady_vals = [steady[m].mean() for m in metrics]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, warmup_vals, width, label='Warmup (Step 0)', color='#e74c3c')
    bars2 = ax.bar(x + width/2, steady_vals, width, label='Steady State (Avg)', color='#2ecc71')
    
    ax.set_ylabel('Duration (ms)', fontsize=12)
    ax.set_title('T5 Warmup vs Steady State Performance', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(bottom=0)
    
    # Add overhead labels
    for i, (w, s) in enumerate(zip(warmup_vals, steady_vals)):
        overhead = w / s if s > 0 else 0
        ax.annotate(f'{overhead:.2f}x', xy=(i, max(w, s) + 15),
                   ha='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, f"warmup_comparison_{run_id}.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Saved: {filepath}")

def main():
    parser = argparse.ArgumentParser(description='Plot hardware monitoring results')
    parser.add_argument('--csv', type=str, required=True, help='Path to hardware_stats CSV')
    parser.add_argument('--output', type=str, default='./plots', help='Output directory')
    parser.add_argument('--run_id', type=str, default='t5', help='Run identifier for filenames')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Loading data from: {args.csv}")
    df = load_data(args.csv)
    print(f"Loaded {len(df)} steps")
    
    print(f"\nGenerating plots in: {args.output}\n")
    
    plot_gpu_memory(df, args.output, args.run_id)
    plot_phase_durations(df, args.output, args.run_id)
    plot_gpu_utilization(df, args.output, args.run_id)
    plot_phase_breakdown(df, args.output, args.run_id)
    plot_warmup_comparison(df, args.output, args.run_id)
    
    print(f"\n---> All plots generated! <---")

if __name__ == '__main__':
    main()
