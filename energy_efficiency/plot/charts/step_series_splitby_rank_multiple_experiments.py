mport style
import experiment
import titles
import base
from base import BaseChart

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import re
import math

class StepSeriesSplitByRankMultipleExperiments(BaseChart):

    def __init__(self, metric, ranks, 
                y_left, 
                y_right = None, 
                x_metric = None, 
                cumulative_left = False, cumulative_right = False,
                smooth_left = False, smooth_right = False,
                grouping_attribute = None):
        
        super().__init__(metric, ranks)
        self.y_left = y_left
        self.y_right = y_right
        self.x_metric = x_metric
        self.cumulative_left = cumulative_left
        self.cumulative_right = cumulative_right
        self.smooth_left = smooth_left
        self.smooth_right = smooth_right
        self.grouping_attribute = grouping_attribute
        
        self.right_axes = None  # will be a list after _build_figure

    def _build_figure(self):
        fig, axs = plt.subplots(nrows=self.nrows, ncols=self.ncols, figsize = self.fig_size, sharex=True) #assumes num ranks >= 2
        axs = axs.flatten() #previous line returns 2D array; we want a 1D array
        self.right_axes = [None] * len(axs)
        return fig, axs

    def _plot(self, experiments, axs):
        # Representative experiment (skips 'simple' since it lacks some ranks)
        sample_exp = experiments[0] if experiments[0].trainer != 'simple' else experiments[1]
        
        for i, rank in enumerate(self.ranks):
            ax_left = axs[i]        
            if self.y_right is not None: 
                ax_right = ax_left.twinx()
                self.right_axes[i] = ax_right

            for j, exp in enumerate(experiments):
                if exp.trainer == 'simple' and rank not in exp.ranks:
                    continue
                colour = style.LINE_COLOURS[j % len(style.LINE_COLOURS)]
              
                # x-axis: iterations or a metric 
                if self.x_metric is not None: 
                    x = exp.get_series(rank, self.x_metric) 
                else: 
                    x = range(2, exp.num_iterations + 1)

                # left y-axis
                yL = exp.get_series(rank, self.y_left, self.cumulative_left, self.smooth_left)
                ax_left.plot(x, yL, color=colour, linestyle="solid" if not self.y_right else "dashed")

                # right y-axis
                if self.y_right is not None:
                    yR = exp.get_series(rank, self.y_right, self.cumulative_right, self.smooth_right)
                    ax_right.plot(x, yR, color=colour, linestyle="solid")

    #TODO: currently, this legend is only helpful if we are varying model type or trainer type. If the set of series corresponds to a different 
    # varying variable, such as batch size, this legend will not be helpful. Make this legend more flexible.
    def _add_legend(self, experiments, fig):
        legend_text = []
        for j, exp in enumerate(experiments):
            colour = style.LINE_COLOURS[j % len(style.LINE_COLOURS)]
            base = f"{exp.model.get_label()} {exp.trainer.get_label()}"

            if self.y_right is not None:
                # Dual y-axis per iteration: two styles, same color
                legend_text.append(Line2D([0], [0], color=colour, linestyle="dashed",
                                      label=f"{base}: {self.y_left.get_label()}"))
                legend_text.append(Line2D([0], [0], color=colour, linestyle="solid",
                                      label=f"{base}: {self.y_right.get_label()}"))
            else:
                # Single y-axis per iteration OR metric-vs-metric
                legend_text.append(Line2D([0], [0], color=colour, linestyle="solid", label = base))
         
         fig.text(style.LEGEND_X, style.LEGEND_Y, legend_text, ha='center', fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="lightgrey"))


    def _decorate(self, experiments, fig, axs):
        sample_exp = experiments[0] if experiments[0].trainer != "simple" else experiments[1]

        # Only set iteration ticks manually 
        if not self.x_metric:
            axs[0].set_xticks(range(1, sample_exp.num_iterations))
        
        for i, ax in enumerate(axs):
            BaseChart.set_subplot_title(ax, f"Rank {i}")
            if i in self.bottom_row_indices:
                xlabel = f"{self.x_metric.get_label()} ({self.x_metric.get_unit()})" if self.x_metric else "Training iterations"
                ax.set_xlabel(xlabel)
            if i in self.left_col_indices:
                left_ylabel = f"{self.y_left.get_label()} ({self.y_left.get_unit()})" #TODO: could be unitless
                ax.set_ylabel(left_ylabel)
            if i in self.right_col_indices:
                if self.y_right: 
                    right_ylabel = f"{self.y_right.get_label()} ({self.y_right.get_unit()})" #TODO: could be unitless
                    self.right_axes[i].set_ylabel(right_ylabel)

        # Figure title
        if self.x_metric is not None:  # metric vs metric
            fig.suptitle(titles.title_metric_vs_metric(experiments, self.y_left, self.x_metric))
        elif self.y_right is not None:  # dual y-axes per iteration
            fig.suptitle(titles.title_dual_metrics_per_iteration(experiments, self.y_left, self.y_right))
        else:  # single y-axis per iteration
            fig.suptitle(titles.title_metric_per_iteration(experiments, self.y_left, self.groupbing_attribute))
        
        # Figure legend
        self._add_legend(experiments, fig)

