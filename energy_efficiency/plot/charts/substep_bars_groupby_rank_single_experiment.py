import style
import experiment
import titles
import base

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import math

class SubstepBarsGroupbyRankSingleExperiment(BaseChart): 
    
    def __init__(self, metric, ranks): 
        super().__init__(metric, ranks)
        self.fig_size = (14, 6) #override

    def _build_figure(self):
        fig, ax = plt.subplots(figsize = self.fig_size)
        return fig, ax

    def _plot(self, experiments, axs):
        ax = axs[0]
        exp = experiments[0]

        for i in range(1, exp.num_iterations): # Start from 1 to skip the first iteration (warm up phase)
            for rank in self.ranks: 
                bar_values = exp.get_stacked_components(rank, i, self.metric) # [Forward, Backward, Optimiser] for current iteration and rank
                group_width = style.get_group_width(len(self.ranks))
                bar_x = style.get_grouped_bar_x_coord(rank, i, group_width) # Bar offset on x-axis
                BaseChart.build_stacked_bar(ax, bar_values, bar_x) # Plot stacked bar
                
                # Label bar with its rank number (0, 1, 2, or 3)
                label_y = sum(values) * style.RANK_LABEL_PERCENT_ABOVE_BAR
                ax.text(bar_x, label_y, rank, ha='center', va='top', fontsize=8)

        
    def _decorate(self, experiments, fig, axs, metric):
        ax = axs[0]
        exp = experiments[0]
        
        group_width = style.get_group_width(self.ranks)
        x_tick_positions = style.get_bar_group_centers(group_width, exp.num_iterations)
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels([f"{i}" for i in range(2, exp.num_iterations + 1)])
        ax.set_xlabel("Training iteration")

        ax.set_ylabel(f"{self.metric.get_label()} ({self.metric.get_unit()})")
        fig.legend(titles.legend_for_substeps())
        fig.suptitle(titles.title_metric_per_iteration(experiments, self.metric))

