import style
import experiment
import titles
import base

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import math

class SubstepBarsSplitByRankSingleExperiment(BaseChart): 
    
    def _build_figure(self): 
        fig, axs = plt.subplots(nrows=self.nrows, ncols=self.ncols, figsize=self.fig_size, sharex=True) #assumes num ranks >= 2
        axs = axs.flatten() #previous line returns 2D array; we want a 1D array for uniformity
        return fig, axs

    def _plot(self, experiments, axs): 
        exp = experiments[0]
        for i, rank in enumerate(self.ranks): 
            ax = axs[i]
            for j in range(1, exp.num_iterations):
                bar_values = exp.get_stacked_components(rank, j, self.metric)
                super().build_stacked_bar(ax, bar_values, j)

    def _decorate(self, experiments, fig, axs): 
        exp = experiments[0]
        
        # Ticks shared between all axes 
        axs[0].set_xticks(range(1, exp.num_iterations))
        
        for i, ax in enumerate(axs):
            BaseChart.set_subplot_title(ax, f"Rank {i}")
            ax.set_ylabel(f"{self.metric.get_label()} ({self.metric.get_unit()})")
            if i in self.bottom_row_indices:
                ax.set_xlabel("Training iteration")
                ax.set_xticklabels([f"{j+1}" for j in range(1, exp.num_iterations)])

        fig.suptitle(titles.title_metric_per_iteration(experiments, self.metric)
        fig.legend(titles.legend_for_substeps())


