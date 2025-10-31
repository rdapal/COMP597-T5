
'''
Visual style/geometry contants and helpers for plots.
'''

# ----------------------- BAR GRAPHS --------------------------------
BAR_WIDTH_SINGLE = 0.8 # width of bar in plots without groups
BAR_WIDTH_GROUPED = 0.2 # width of bar in plots with grouped bars

BAR_SPACING_SINGLE = 0.02 # spacing between bars in plots without groups
BAR_SPACING_GROUPED = 0.05 # spacing between bars within a group
GROUP_SPACING = 0.3 # spacing between groups

BAR_HATCHES = ['/', '\\', ''] # Forward: '/', Backward: '\\', Optimiser: solid
BAR_COLOURS = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Forward: 'Blue', Backward: 'Orange', Optimiser: 'Green'


# ----------------------- LINE GRAPHS --------------------------------
LINE_COLOURS = ['red', 'blue', 'purple', 'green'] # CAREFUL: code will cycle (repeat) through colours if #series > #colours


# ----------------------- TEXT GEOMETRY --------------------------------
Y_AXIS_PADDING_PERCENT = 1.05 # Add padding to y-axis for visual appeal 
Y_AXIS_SUBTITLE_PADDING_PERCENT = 1.15 # Add padding to y-axis to make room for the subplot's title
RANK_LABEL_PERCENT_ABOVE_BAR = 1.05 # Rank number (0, 1, 2, ...) displayed above each bar
TEXT_LABEL_PERCENT_ABOVE_BAR = 1.07 # Text labels (e.g. for trainer: "c", "s", etc) displayed above each bar

SUPER_TITLE_Y = 0.98 # y coordinate of the fig supertitle
SUBPLOT_TITLE_X = 0.5 # x coordinate of suplot title 
SUBPLOT_TITLE_Y = 0.95 # y coordinate of sublot title

LEGEND_X = 0.5 # x coordinate of legend
LEGEND_Y = 0.93 # y coordinate of legend

TIGHT_LAYOUT_RECT = [0, 0, 1, 0.91]


# ----------------------- HELPER FUNCTIONS --------------------------------
# Returns x-coordinate of a bar in a grouped bar chart (i.e. chart with clusters of bars at each x coordinate)
def get_grouped_bar_x_coord(rank, iteration, group_width): 
    return iteration * (group_width + GROUP_SPACING) + rank * (BAR_WIDTH_GROUPED + BAR_SPACING_GROUPED)

# Returns total x-axis width of a group of bars in a grouped bar chart
def get_group_width(num_bars_per_group): 
    return num_bars_per_group * (BAR_WIDTH_GROUPED + BAR_SPACING_GROUPED)

# Returns the x-coordinate of the centers of all groups in a grouped bar chart
def get_bar_group_centers(group_width, num_iterations):
    positions = [
        i * (group_width + GROUP_SPACING) + (group_width - BAR_SPACING_GROUPED - BAR_WIDTH_GROUPED) / 2 
        for i in range(1, num_iterations)
        ]

    return positions 

# Returns the indices of the bottom-most plots in a figure containing multiple subplots
def get_bottom_row_indices(num_subplots, nrows, ncols):
    return [i for i in range(num_subplots) if i // ncols == nrows - 1]

#TODO: to allow y-axis sharing 
def get_leftmost_column_indices():
    pass


# Returns (nrows, ncols) of a figure given the number of desired subplots in the figure
def get_subplot_grid(num_subplots): 
    if num_subplots in (1, 2, 3): 
        return (num_subplots, 1)
    return (math.ceil(num_subplots/2), 2) # Max 2 columns


