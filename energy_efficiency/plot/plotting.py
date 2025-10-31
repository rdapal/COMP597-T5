import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import plot_utils
import experiment
import re
import math

def metric_per_iteration_grouped_by_rank(exp, metric):
    '''
    Granularity: substep
    Mode: single 
    Type of chart: grouped and stacked bar chart
    Y-axis: metric
    X-axis: iteration number
    Description: One bar per gpu rank for each iteration. Each bar is stacked: Forward / Backward / Optimiser
    '''

    # Initialise graphing parameters
    bar_width = 0.2
    bar_spacing = 0.02
    group_width = len(exp.ranks) * (bar_width + bar_spacing)
    group_spacing = 0.3
    hatches = ['/', '\\', '']  # Forward: '/', Backward: '\\', Optimiser: solid
    colours = ['#1f77b4', '#ff7f0e', '#2ca02c']  # In order: Forward, Backward, Optimiser

    fig, ax = plt.subplots(figsize=(14, 6))

    for i in range(1, exp.num_iterations): # Start from 1 to skip the first iteration (warm up phase)
        for j in exp.ranks: # One bar per gpu rank 
            # Get [Forward, Backward, Optimiser] values for current iteration and rank
            values = exp.data[metric][j][i]

            # Calculate bar offset on graph
            offset = i * (group_width + group_spacing) + j * (bar_width + bar_spacing)

            # Plot stacked bar for this rank
            bottom = 0
            for x, value in enumerate(values):
                ax.bar(offset, value, width = bar_width, edgecolor = 'black', hatch = hatches[x], color = colours[x], bottom = bottom)
                bottom += value

            # Label bar with its rank number (0, 1, 2, or 3)
            vertical_pos = sum(values) + (sum(values) * 0.05)
            ax.text(offset, vertical_pos, j, ha='center', va='top', fontsize=8) #verical position is just above bars

    # Set x-axis ticks
    tick_positions = [
            i * (group_width + group_spacing) + (group_width - bar_spacing - bar_width) / 2
            for i in range(1, exp.num_iterations)
            ]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{i+1}" for i in range(1, exp.num_iterations)])

    # Titles, legends, axes
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.05) # Add 5% padding to y-axis for visual appeal
    ax.legend(['Forward', 'Backward', 'Optimiser'])
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(f"{metric} ({plot_utils.get_unit(metric)})")
    ax.set_title(f"{metric} ({plot_utils.get_unit(metric)}) per iteration depending on rank ({exp.model}_{exp.trainer}, {exp.num_experts} experts, batch size {exp.batch_size}, {exp.num_gpus} gpus)", wrap=True)
    
    fig.tight_layout()
    return fig 

def metric_per_iteration_separated_by_rank(exp, metric):
    '''
    Granularity: substep
    Mode: single
    Type of chart: stacked bar chart, multiple plots
    Y-axis: metric
    X-axis: iteration
    Description: one plot per GPU rank. Each bar is stacked: Forward / Backward / Optimiser
    '''

    # Initialise graphing parameters
    bar_width = 0.8
    hatches = ['/', '\\', '']
    colours = ['#1f77b4', '#ff7f0e', '#2ca02c']
    num_ranks = len(exp.ranks)
    nrows = 4
    ncols = math.ceil(num_ranks / nrows)
    bottom_row_indices = [i for i in range(num_ranks) if i // ncols == nrows - 1]
    
    # Plotting
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.5 * ncols, 3.5 * nrows), sharex=True) #assumes num_ranks > 1
    axs = axs.flatten() #previous line returns 2D array; we want a 1D array for uniformity

    for i, rank in enumerate(exp.ranks): 
        ax = axs[i]
        for j in range(1, exp.num_iterations): # Start from 1 to skip the first iteration (warm up phase)
            values = exp.data[metric][rank][j]
            bottom = 0
            for k, value in enumerate(values): 
                ax.bar(j, value, width=bar_width, bottom=bottom, color=colours[k], hatch=hatches[k], edgecolor='black')
                bottom += value
        
        # Subplot axes and title
        ax.set_ylabel(f"{metric} ({plot_utils.get_unit(metric)})")
        ymin, ymax = ax.get_ylim() # Add padding to y-axis for rank subtitle
        ax.set_ylim(ymin, ymax * 1.15)
        ax.text(0.5, 0.95, f"Rank {rank}", transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='center')
        
        if i in bottom_row_indices:
            ax.set_xlabel("Training iteration")
            ax.set_xticks(range(1, exp.num_iterations))
            ax.set_xticklabels([f"{j+1}" for j in range(1, exp.num_iterations)])

    # Legend, title for full plot
    fig.legend(['Forward', 'Backward', 'Optimiser'])
    plt.suptitle(f"{metric} ({plot_utils.get_unit(metric)}) per iteration ({exp.model}_{exp.trainer}, {exp.num_experts} experts, batch size {exp.batch_size}, {exp.num_gpus} gpus)", wrap=True)
   
    fig.tight_layout()
    return fig

def metric_per_iteration_grouped_by_trainer_separated_by_rank(experiments, metric):
    '''
    Granularity: substep
    Mode: multiple trainers, same parameter set and model
    Type of chart: grouped and stacked bar chart, multiple plots
    Y-axis: metric
    X-axis: iteration number
    Description: One plot ger gpu rank. One bar per trainer per group. Each bar is stacked: Forward / Backward / Optimiser
    '''
    # Initialise graphing parameters
    num_trainers = len(experiments)
    rank_list = experiments[0].ranks #parameters are the same for all datasets, so arbitrarily read dataset 0
    num_ranks = len(rank_list)
    num_iterations = experiments[0].num_iterations
    contains_simple_trainer = any(exp.trainer == "simple" for exp in experiments)

    bar_width = 0.2
    bar_spacing = 0.05
    group_width = num_trainers * (bar_width + bar_spacing)
    group_spacing = 0.3
    hatches = ['/', '\\', '']  # Forward: '/', Backward: '\\', Optimiser: solid
    colours = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Forward, Backward, Optimiser
    
    nrows = 4 #Always set 4 rows and vary columns depending on number of ranks
    ncols = math.ceil(num_ranks / nrows)
    bottom_row_indices = [i for i in range(num_ranks) if i // ncols == nrows - 1] #indices of cells in the bottow row of the figure

    # Plot figure (several subplots)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.5 * ncols, 4.5 * nrows), sharex=True)
    axs = axs.flatten()

    # One subplot per rank
    for i, rank in enumerate(rank_list):
        ax = axs[i]
        for j in range(1, num_iterations): # Start from 1 to skip the first iteration (warm up phase) 
            for k, exp in enumerate(experiments): # Each experiment corresponds to a different trainer
                if exp.trainer == 'simple' and rank not in exp.data[metric].keys(): 
                    continue

                # Get [Forward, Backward, Optimiser] values for current rank, iteration, and trainer
                values = exp.data[metric][rank][j]

                # Calculate bar offset on graph
                offset = j * (group_width + group_spacing) + k * (bar_width + bar_spacing)

                # Create the stacked bar
                bottom = 0
                for x, value in enumerate(values):
                    ax.bar(offset, value, width = bar_width, edgecolor = 'black', hatch = hatches[x], color = colours[x], bottom = bottom)
                    bottom += value

                # Add trainer's name as label above the bar
                trainer_name = plot_utils.get_abbreviated_trainer_name(exp.trainer)
                vertical_pos = sum(values) + (sum(values) * 0.07) #small percentage above total bar height
                ax.text(offset, vertical_pos, trainer_name, ha='center', va='top', fontsize=12) #verical position is just above bars
        
        # Subplot labels, title
        ax.set_ylabel(f"{metric} ({plot_utils.get_unit(metric)})")
        ymin, ymax = ax.get_ylim() 
        ax.set_ylim(ymin, ymax * 1.15) # Add padding to y-axis to fit subplot title
        ax.text(0.5, 0.95, f"Rank {rank}", transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='center') # Subplot title
        
        if i in bottom_row_indices:
            ax.set_xlabel("Training iteration")
            tick_positions = [
                    i * (group_width + group_spacing) + (group_width - bar_spacing - bar_width) / 2
                    for i in range(1, exp.num_iterations)
                    ]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([f"{j+1}" for j in range(1, num_iterations)])

    # Title, legends for full plot
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.legend(['Forward', 'Backward', 'Optimiser'], loc='upper center', bbox_to_anchor=(0.5, 0.955), ncol=3)
    
    trainer_types = set(exp.trainer for exp in experiments)
    labels = [f"{plot_utils.get_abbreviated_trainer_name(t)} - {t}" for t in trainer_types]
    legend_text = '    '.join(labels)
    fig.text(
            0.5,
            0.92,
            legend_text,
            ha='center',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="lightgrey")
            )

    fig.suptitle(f"{metric} ({plot_utils.get_unit(metric)}) per iteration for each trainer ({experiments[0].model} model, {experiments[0].num_experts} experts, batch size {experiments[0].batch_size}, {experiments[0].num_gpus} gpus)", y=0.98, wrap=True)
   
    return fig


# Plot: dual y axis -- cumulative metric, loss. x axis: iteration. Lines: one model, fmoe and deepspeed trainers, and same set of parameters. One subplot 
# per rank.
def metric_against_loss_per_iteration(experiments, metric):
    
    ranks = experiments[0].ranks 
    num_iterations = experiments[0].num_iterations
    
    nrows = 4 #Always set 4 rows and vary columns depending on number of ranks
    ncols = math.ceil(len(ranks) / nrows)
    bottom_row_indices = [i for i in range(len(ranks)) if i // ncols == nrows - 1] #indices of cells in the bottow row of the figure
    colours = ['red', 'blue', 'purple', 'green']
    
    # Figure contains one subplot per rank
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.5 * ncols, 4.5 * nrows), sharex=True)
    axs = axs.flatten()

    # Build figure subplot-by-subplot
    for i, rank in enumerate(ranks):
        ax1 = axs[i] # Left y-axis: metric
        ax2 = ax1.twinx() # Right y-axis: loss
        
        for j, exp in enumerate(experiments): # Each experiment corresponds to a different trainer
            df = exp.df.sort_values(['gpu_rank', 'iteration']) # Normally, the csv's are already sorted by rank then iteration, but we assure it
            df[f"cumulative_{metric}"] = df.groupby('gpu_rank')[metric].cumsum() # Sum cumulated over iterations, independent for each rank 
            rank_df = df[(df['gpu_rank'] == rank) & (df['iteration'] != 1)] # Skip the first iteration (warm-up phase)
            
            colour = colours[j % len(colours)]
            
            # Plot cumulative metric on left y-axis
            ax1.plot(rank_df['iteration'],
            rank_df[f"cumulative_{metric}"],
            color=colour, 
            linestyle='dashed',
            label=f"{exp.model}_{exp.trainer} {metric}"
            )
            
            # Plot loss on right y-axis
            rank_df_sub = rank_df.iloc[::50]
            ax2.plot(rank_df_sub['iteration'],
            rank_df_sub['loss'],
            color=colour,
            linestyle='solid',
            label=f"{exp.model}_{exp.trainer} loss"
            )
         
        # Subtitles, axis ticks
        ymin, ymax = ax1.get_ylim()
        ax1.set_ylim(ymin, ymax * 1.15) # Add padding to y-axis to fit title within subplot
        ax1.set_title(f'Rank {rank}')

        ax1.set_ylabel(f"Cumulative {metric} ({plot_utils.get_unit(metric)})")
        ax1.tick_params(axis='y')

        ax2.set_ylabel('Loss')
        ax2.tick_params(axis='y')

        ax1.set_xlabel("Iteration")
        
    # Legend, title
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    legend_text = []
    for k, exp in enumerate(experiments):
        colour = colours[k % len(colours)]
        legend_text.append(Line2D([0], [0], color=colour, linestyle='dashed',
                                      label=f"{exp.model}-{exp.trainer} {metric}"))
        legend_text.append(Line2D([0], [0], color=colour, linestyle='solid',
                                      label=f"{exp.model}-{exp.trainer} loss"))

    fig.legend(handles=legend_text, loc='center', ncol=2, bbox_to_anchor=(0.5, 0.93))
    fig.suptitle(f"Cumulative {metric} ({plot_utils.get_unit(metric)}) vs loss per iteration, depending on trainer ({experiments[0].model} model, learning rate {experiments[0].learning_rate}, {experiments[0].num_experts} experts, batch size {experiments[0].batch_size}, {experiments[0].num_gpus} gpus)", y=0.98, wrap=True)

    return fig



