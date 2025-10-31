from abc import ABC, abstractmethod
import style
import experiment

class BaseChart(ABC): 

    def __init__(self, metric, ranks): 
        self.metric = metric 
        self.ranks = ranks 
        self.nrows, self.ncols = style.get_subplot_grid(len(self.ranks))
        self.bottom_row_indices = style.get_bottom_row_indices(len(self.ranks), self.nrows, self.ncols)
        self.fig_size = (6.5 * self.ncols, 4.5 * self.nrows)

    # ------------------------------------- TEMPLATE METHOD ------------------------------------
    def render(self, experiments):
        fig, axs = self._build_figure()
        self._plot(experiments, axs)
        self._decorate(experiments, fig, axs)
        return fig 

    # ------------------------------------- ABSTRACT STEPS --------------------------------------
    @abstractmethod
    def _build_figure(self): 
        pass

    @abstractmethod 
    def _plot(self, experiments, axs): 
        pass

    @abstractmethod
    def _decorate(self, experiments, fig, axs): 
        pass

    # ------------------------------------- HELPERS ----------------------------------------------
    @staticmethod
    def build_stacked_bar(ax, stacked_values, bar_offset): 
        bottom = 0
        for x, value in enumerate(stacked_values):
            ax.bar(
            bar_offset, 
            value, 
            width = style.BAR_WIDTH_GROUPED,
            edgecolor = 'black',
            hatch = style.BAR_HATCHES[x],
            color = style.BAR_COLOURS[x],
            bottom = bottom
            )
            
            bottom += value
    
   @staticmethod 
   def set_subplot_title(ax, title):
        ymin, ymax = ax.get_ylim() # Add padding to y-axis to make space for the subplot title
        ax.set_ylim(ymin, ymax * style.Y_AXIS_SUBTITLE_PADDING_PERCENT)
        ax.text(style.SUBPLOT_TITLE_X, style.SUBPLOT_TITLE_Y, title, transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='center')

