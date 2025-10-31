import style
import experiment
import titles
import base
from base import BaseChart

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import math

class SubstepBarsSplitByRankMultipleExperiments(BaseChart): 
    
    def __init__(self, metric, ranks, grouping_attribute): 
        super().__init__(metric, ranks)
        self.grouping_attribute = grouping_attribute

    def _build_figure(self):
        fig, axs = plt.subplots(nrows=self.nrows, ncols=self.ncols, figsize = self.fig_size, sharex=True) #assumes num ranks >= 2
        axs = axs.flatten() #previous line returns 2D array; we want a 1D array
        return fig, axs

    def _plot(self, experiments, axs):
        sample_exp = experiments[0] if experiments[0].trainer != 'simple' else experiments[1]
        num_iterations = sample_exp.num_iterations
        group_width = style.get_group_width(len(self.ranks))

        for i, rank in enumerate(self.ranks):
            ax = axs[i] # One subplot per rank
            for j in range(1, num_iterations): # Skip first iteration (warm up phase)
                for exp in experiments:
                    if exp.trainer == 'simple' and rank not in exp.ranks: # Exps with simple trainer skip some ranks
                        continue
                    bar_values = exp.get_stacked_components(rank, j, self.metric) # [Forward, Backward, Optimiser] for current rank and iteration
                    bar_x = style.get_grouped_bar_x_coord(rank, j, group_width)
                    BaseChart.build_stacked_bar(ax, bar_values, bar_x)
                    
                    # Label bar with the experiment's grouping attribute (e.g. "d" for "deepspeed" when grouping_attribute = Trainer)
                    attribute = getattr(exp, self.grouping_attribute.__name__.lower()) # e.g. Trainer -> self.trainer
                    label = attribute.get_abbrev() 
                    label_y = sum(values) * style.TEXT_LABEL_PERCENT_ABOVE_BAR #Label y coordinate is jsut above bar
                    ax.text(bar_x, label_y, label, ha='center', va='top', fontsize=12)

    def _decorate(self, experiments, fig, axs):
        sample_exp = experiments[0]
        
        # Tick marks are shared across all axes
        axs[0].set_xticks(range(1, sample_exp.num_iterations))

        for i, ax in enumerate(axs):
            self.set_subplot_title(ax, f"Rank {i}")
            if i in self.bottom_row_indices:
                ax.set_xlabel("Training iteration")
            if i in self.left_col_indices: 
                ax.set_ylabel(f"{self.metric.get_label()} ({self.metric.get_unit()})")
        
        fig.suptitle(titles.title_metric_per_iteration(experiments, self.metric, self.grouping_attribute))
        legend_text = titles.legend_for_grouping_attribute(experiments, self.grouping_attribute)
        fig.text(style.LEGEND_X, style.LEGEND_Y, legend_text, ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="lightgrey"))

