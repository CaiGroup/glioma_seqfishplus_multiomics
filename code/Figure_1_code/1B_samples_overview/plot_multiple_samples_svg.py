import glob
import pandas as pd
import os
import pickle
import numpy as np
from collections import defaultdict
os.environ["PATH"] = os.path.expanduser("~/miniconda3/envs/fox_env24/bin") + ":" + os.environ["PATH"]
pd.options.mode.chained_assignment = None
# plotting packages
from bokeh.plotting import figure, show
from bokeh.io import export_svg
from bokeh.models import ColumnDataSource, ColorBar, Span, Label, Range1d
from bokeh.transform import factor_cmap
from bokeh.models import Legend
from bokeh.layouts import gridplot
from scipy.spatial import ConvexHull, Delaunay
from scipy.interpolate import splprep, splev

palette_hex = [
    "#73b9a5", "#a33f3e", "#2261b2", "#fea17b", "#c5aedf",
    "#d4af9e", "#fea8d4", "#7e5679", "#6ea9fd", "#bd8985",
    "#d36d54", "#e4d177", "#89aeb3", "#fec4a7", "#977d89",
    "#637fd0", "#93581c", "#4bcbfe", "#82e2dd", "#dea257",
    "#70715d", "#009aa5", "#c27869", "#a893c6", "#00eafa",
    "#a96296", "#769be0", "#aa5839", "#7dcab8", "#847ebb",
    "#f28679", "#286958", "#00838d", "#b9989c", "#de8a5c",
    "#e3c7fe", "#97792d", "#6d6cb7", "#8e586f", "#b9d08a",
    "#a8a6e3", "#c35653", "#50729f", "#70548b", "#575a99",
    "#b26270", "#a58d39", "#9cc2c8", "#c0af85", "#72999a",
    "#5ebafe", "#59d6ac", "#984649", "#1bb1ca", "#c99b89",
    "#8fe7a1", "#568972", "#6690e7", "#e5b76b", "#798258",
    "#9984aa", "#a9dfc8", "#567319", "#cd8b68", "#979bb7",
    "#b17981", "#bacde7", "#8c6d80", "#576424", "#cfbe91",
    "#b2abca", "#e6bcb3", "#9783cf", "#2e9096", "#bd6c41",
]

def clean_axes(p):
    p.xaxis.axis_label = None
    p.yaxis.axis_label = None
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.xaxis.major_label_text_font_size = "0pt"
    p.yaxis.major_label_text_font_size = "0pt"
    p.xaxis.major_tick_line_color = None
    p.xaxis.minor_tick_line_color = None
    p.yaxis.major_tick_line_color = None
    p.yaxis.minor_tick_line_color = None
    p.xaxis.axis_line_color = None
    p.yaxis.axis_line_color = None
    p.title.text_color = "black"
    p.title.text_font_size = "9pt"

def add_scale_bar(p, bar_x_start, bar_x_end, bar_y):
    p.line([bar_x_start, bar_x_end], [bar_y, bar_y], line_width=3, line_color="black")
    p.add_layout(Label(
        x=(bar_x_start + bar_x_end) / 2, y=bar_y,
        text="2 \u03bcm", text_align="center", text_baseline="bottom",
        text_color="black", text_font_size="10pt", y_offset=2))

# load cell type data
celltype_df = pd.read_csv('/resnick/groups/mthomson/ci2_data/CaiLab/gbm/dna_distance_data/V2_24_patients_Obj_mapped.csv', low_memory=False)
celltype_df.columns = [col.replace('.', '_') for col in celltype_df.columns]
celltype_df.rename(columns={'Unnamed: 0': 'cell_id'}, inplace=True)

# order cell types (tumor first, then normal)
unique_celltypes = [
    'AC.like',
    'OPC',
    'OPC.like Prolif.',
    'AC',
    'Endothelial',
    'Microglia',
    'Neuron GABA.',
    'Neuron Glut.',
    'Olig. normal',
    'Pericytes',
    'T'
]

# assign colors
palette = palette_hex[:len(unique_celltypes)]

# get list of samples
sample_list = celltype_df.ID.unique()

# plot each sample, collect into list
plot_size = 200
plots = []

for sample in sample_list:
    sample_df = celltype_df.loc[celltype_df['ID'] == sample]

    p = figure(title=str(sample),
               width=plot_size, height=plot_size,
               background_fill_color="white",
               border_fill_color="white",
               toolbar_location=None,
               outline_line_color=None)

    for celltype_value, color in zip(unique_celltypes, palette):
        celltype_source = ColumnDataSource(sample_df[sample_df['predicted_celltype'] == celltype_value])
        p.scatter('center_x', 'center_y', source=celltype_source, size=2, fill_alpha=0.8, color=color)

    clean_axes(p)
    plots.append(p)

# pad to fill grid if fewer than 40 samples
n_cols = 5
n_rows = 8
total_cells = n_cols * n_rows
while len(plots) < total_cells:
    plots.append(None)

# reshape into rows
grid = [plots[i * n_cols:(i + 1) * n_cols] for i in range(n_rows)]

grid_plot = gridplot(grid, merge_tools=False)

# export as SVG
output_dir = "../../../figures/Figure_1/Figure_1B/"
os.makedirs(output_dir, exist_ok=True)

for row in grid:
    for p in row:
        if p is not None:
            p.output_backend = "svg"

export_svg(grid_plot, filename=os.path.join(output_dir, "Figure_1B.svg"))