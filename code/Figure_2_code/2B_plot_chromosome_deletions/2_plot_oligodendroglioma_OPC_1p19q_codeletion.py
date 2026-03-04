import glob
import pandas as pd
import os
import pickle

os.environ["PATH"] = os.path.expanduser("~/miniconda3/envs/fox_env24/bin") + ":" + os.environ["PATH"]
pd.options.mode.chained_assignment = None

# plotting packages
from bokeh.plotting import figure
from bokeh.io import export_svgs
from bokeh.models import ColumnDataSource, ColorBar, Span, Label, Range1d
from bokeh.transform import factor_cmap
from bokeh.models import Legend

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

path = '/central/groups/mthomson/sgdata/CaiLab/michalp/Glioma_analysis/data/DNA_data'

# get all files
files = glob.glob(f"{path}/*assigned.csv")

# only keep 1 patient sample
patient_id = '2543'
files = [i for i in files if patient_id in i]

# load data
df_1 = pd.read_csv(files[0], low_memory = False)

# fix chromosomes
# make dictionary
probe_df = pd.read_csv("/central/groups/CaiLab/personal/jjfox/DNA_seqFISH/human/24-12-19_human_genome_DNA_seqFISH_final_probes.csv")
df_unique = probe_df.drop_duplicates(subset='bin_name', keep='last')
chr_dict = dict(zip(df_unique['bin_name'], df_unique['chr']))

# map
df_1["blockbin"] = (
    df_1["block_assigned"].str.replace("block_", "block", regex=False)
    + "_" +
    df_1["name"].str.replace("bin_", "bin", regex=False)
)
df_1['chromosome_true'] = df_1.blockbin.map(chr_dict)

# map chromosome arms
# load dictionary mapping chromosome arms to blockbins
with open('chrom_arm_dict.pkl', 'rb') as f:
    chrom_arm_dict = pickle.load(f)

# map chromosome arms for blockbins
df_1['chrom_arm'] = df_1.blockbin.map(chrom_arm_dict)

# load cell type data
celltype_df = pd.read_csv('/resnick/groups/mthomson/ci2_data/CaiLab/gbm/dna_distance_data/V2_24_patients_Obj_mapped.csv', low_memory=False)
# replace . with _ in column names
celltype_df.columns = [
    col.replace('.', '_') for col in celltype_df.columns
]
# Rename column for cell ID
celltype_df.rename(columns={'Unnamed: 0': 'cell_id'}, inplace=True)
# get cell types in this sample
gbm_cells = celltype_df.loc[celltype_df['ID'] == patient_id].copy()
# remove underscore from IDs
gbm_cells['cell_id'] = gbm_cells['cell_id'].apply(lambda x: int(x.split('_')[0]))
# make dictionary of cell ID and cell type
gbm_cell_dict = dict(zip(gbm_cells['cell_id'], gbm_cells['predicted_celltype']))                
# add cell types to df_1
df_1['predicted_celltype'] = df_1['matched_label'].map(gbm_cell_dict)
celltypes = df_1['predicted_celltype'].unique()

##### Plot Tumor Cell ####

# look at 1 cell
cell = 7763

df_cell = df_1.loc[df_1['matched_label'] == cell].copy()

# drop na for serial and unassigned bins
df_cell = df_cell[df_cell['chromosome_true'].notna()]

# ---- Shared range computation ----

x_min, x_max = df_cell['x'].min(), df_cell['x'].max()
y_min, y_max = df_cell['y'].min(), df_cell['y'].max()
x_span = x_max - x_min
y_span = y_max - y_min
max_span = max(x_span, y_span)
x_center = (x_min + x_max) / 2
y_center = (y_min + y_max) / 2
pad = 0.05 * max_span

shared_x_range = (x_center - max_span/2 - pad, x_center + max_span/2 + pad)
shared_y_range = (y_center - max_span/2 - pad, y_center + max_span/2 + pad)

plot_size = 670

# ---- Plot 1: Chromosome scatter ----
source = ColumnDataSource(df_cell)
unique_chrs = df_cell['chromosome_true'].unique()
palette = palette_hex[:len(unique_chrs)]
unique_chrs = list(unique_chrs)
unique_chrs.sort(key=lambda x: x.split('chr')[1])
unique_chrs_num = unique_chrs[:-2]
unique_chrs_num.sort(key=lambda x: int(x.split('chr')[1]))
unique_chrs = unique_chrs_num + [unique_chrs[-2]] + [unique_chrs[-1]]

p = figure(title=f"Scatterplot by Chromosome Cell {cell}",
           width=plot_size, height=plot_size,
           background_fill_color="white",
           border_fill_color="white",
           x_range=Range1d(start=shared_x_range[0], end=shared_x_range[1]),
           y_range=Range1d(start=shared_y_range[0], end=shared_y_range[1]))

for chr_value, color in zip(unique_chrs, palette):
    chr_source = ColumnDataSource(df_cell[df_cell['chromosome_true'] == chr_value])
    p.scatter('x', 'y', source=chr_source, size=13, fill_alpha=0.8, color=color)

#p.legend.visible = False
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
p.title.text_color = "white"

scale_bar_length = 2
bar_x_end = x_max - 0.05 * x_span
bar_x_start = bar_x_end - scale_bar_length
bar_y = y_min + 0.05 * y_span

p.line([bar_x_start, bar_x_end], [bar_y, bar_y], line_width=3, line_color="black")
p.add_layout(Label(
    x=(bar_x_start + bar_x_end) / 2, y=bar_y,
    text="2 \u03bcm", text_align="center", text_baseline="bottom",
    text_color="black", text_font_size="10pt", y_offset=2))

p.output_backend = "svg"
export_svgs(p, filename="../../../figures/Figure_2/2543_OPC_chromosomes.svg") 

# ---- Plot 2: Chromosome arm scatter chr1 ----
df_cell_chr = df_cell.sort_values('chrom_arm', key=lambda x: x.map({'p': 0, 'q': 1}))
source = ColumnDataSource(df_cell_chr)

chrom_factors = ['p', 'q']
color_mapper = factor_cmap('chrom_arm', palette=[palette_hex[1], palette_hex[2]], factors=chrom_factors)

p2 = figure(title="Scatter Plot Colored by Chromosome Arm",
           background_fill_color="white",
           border_fill_color="white",
           x_range=Range1d(start=shared_x_range[0], end=shared_x_range[1]),
           y_range=Range1d(start=shared_y_range[0], end=shared_y_range[1]),
           width=plot_size, height=plot_size)

# Plot all non-chr1 first (grey, background)
for chr_value, color in zip(unique_chrs, palette):
    if chr_value == 'chr1':
        continue
    chr_source = ColumnDataSource(df_cell_chr[df_cell_chr['chromosome_true'] == chr_value])
    p2.scatter('x', 'y', source=chr_source, size=13, fill_alpha=0.8, color='#CCCCCC')

# Plot chr1 last (on top, colored by arm)
chr1_df = df_cell_chr[df_cell_chr['chromosome_true'] == 'chr1']
chr1_source = ColumnDataSource(chr1_df)
p2.scatter('x', 'y', source=chr1_source, size=13, fill_alpha=0.8, color=color_mapper)


#p2.legend.visible = False
p2.xaxis.axis_label = None
p2.yaxis.axis_label = None
p2.xgrid.grid_line_color = None
p2.ygrid.grid_line_color = None
p2.xaxis.major_label_text_font_size = "0pt"
p2.yaxis.major_label_text_font_size = "0pt"
p2.xaxis.major_tick_line_color = None
p2.xaxis.minor_tick_line_color = None
p2.yaxis.major_tick_line_color = None
p2.yaxis.minor_tick_line_color = None
p2.xaxis.axis_line_color = None
p2.yaxis.axis_line_color = None
p2.title.text_color = "white"

p2.line([bar_x_start, bar_x_end], [bar_y, bar_y], line_width=3, line_color="black")
p2.add_layout(Label(
    x=(bar_x_start + bar_x_end) / 2, y=bar_y,
    text="2 \u03bcm", text_align="center", text_baseline="bottom",
    text_color="black", text_font_size="10pt", y_offset=2))

p2.output_backend = "svg"
export_svgs(p2, filename="../../../figures/Figure_2/2543_OPC_chr1.svg")

# ---- Plot 3: Chromosome arm scatter chr19 ----
df_cell_chr = df_cell.sort_values('chrom_arm', key=lambda x: x.map({'p': 0, 'q': 1}))
source = ColumnDataSource(df_cell_chr)

chrom_factors = ['p', 'q']
color_mapper = factor_cmap('chrom_arm', palette=[palette_hex[1], palette_hex[2]], factors=chrom_factors)

p3 = figure(title="Scatter Plot Colored by Chromosome Arm",
           background_fill_color="white",
           border_fill_color="white",
           x_range=Range1d(start=shared_x_range[0], end=shared_x_range[1]),
           y_range=Range1d(start=shared_y_range[0], end=shared_y_range[1]),
           width=plot_size, height=plot_size)

# Plot all non-chr1 first (grey, background)
for chr_value, color in zip(unique_chrs, palette):
    if chr_value == 'chr19':
        continue
    chr_source = ColumnDataSource(df_cell_chr[df_cell_chr['chromosome_true'] == chr_value])
    p3.scatter('x', 'y', source=chr_source, size=13, fill_alpha=0.8, color='#CCCCCC')

# Plot chr1 last (on top, colored by arm)
chr19_df = df_cell_chr[df_cell_chr['chromosome_true'] == 'chr19']
chr19_source = ColumnDataSource(chr19_df)
p3.scatter('x', 'y', source=chr19_source, size=13, fill_alpha=0.8, color=color_mapper)


p3.xaxis.axis_label = None
p3.yaxis.axis_label = None
p3.xgrid.grid_line_color = None
p3.ygrid.grid_line_color = None
p3.xaxis.major_label_text_font_size = "0pt"
p3.yaxis.major_label_text_font_size = "0pt"
p3.xaxis.major_tick_line_color = None
p3.xaxis.minor_tick_line_color = None
p3.yaxis.major_tick_line_color = None
p3.yaxis.minor_tick_line_color = None
p3.xaxis.axis_line_color = None
p3.yaxis.axis_line_color = None
p3.title.text_color = "white"

p3.line([bar_x_start, bar_x_end], [bar_y, bar_y], line_width=3, line_color="black")
p3.add_layout(Label(
    x=(bar_x_start + bar_x_end) / 2, y=bar_y,
    text="2 \u03bcm", text_align="center", text_baseline="bottom",
    text_color="black", text_font_size="10pt", y_offset=2))

p3.output_backend = "svg"
export_svgs(p3, filename="../../../figures/Figure_2/2543_OPC_chr19.svg")

##### Plot Normal Cell ####

# look at 1 cell
cell = 2057

df_cell = df_1.loc[df_1['matched_label'] == cell].copy()

# drop na for serial and unassigned bins
df_cell = df_cell[df_cell['chromosome_true'].notna()]

# ---- Shared range computation ----

x_min, x_max = df_cell['x'].min(), df_cell['x'].max()
y_min, y_max = df_cell['y'].min(), df_cell['y'].max()
x_span = x_max - x_min
y_span = y_max - y_min
max_span = max(x_span, y_span)
x_center = (x_min + x_max) / 2
y_center = (y_min + y_max) / 2
pad = 0.05 * max_span

shared_x_range = (x_center - max_span/2 - pad, x_center + max_span/2 + pad)
shared_y_range = (y_center - max_span/2 - pad, y_center + max_span/2 + pad)

plot_size = 670

# ---- Plot 1: Chromosome scatter ----
source = ColumnDataSource(df_cell)
unique_chrs = df_cell['chromosome_true'].unique()
palette = palette_hex[:len(unique_chrs)]
unique_chrs = list(unique_chrs)
unique_chrs.sort(key=lambda x: x.split('chr')[1])
unique_chrs_num = unique_chrs[:-2]
unique_chrs_num.sort(key=lambda x: int(x.split('chr')[1]))
unique_chrs = unique_chrs_num + [unique_chrs[-2]] + [unique_chrs[-1]]

p = figure(title=f"Scatterplot by Chromosome Cell {cell}",
           width=plot_size, height=plot_size,
           background_fill_color="white",
           border_fill_color="white",
           x_range=Range1d(start=shared_x_range[0], end=shared_x_range[1]),
           y_range=Range1d(start=shared_y_range[0], end=shared_y_range[1]))

for chr_value, color in zip(unique_chrs, palette):
    chr_source = ColumnDataSource(df_cell[df_cell['chromosome_true'] == chr_value])
    p.scatter('x', 'y', source=chr_source, size=13, fill_alpha=0.8, color=color)

#p.legend.visible = False
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
p.title.text_color = "white"

scale_bar_length = 2
bar_x_end = x_max - 0.05 * x_span
bar_x_start = bar_x_end - scale_bar_length
bar_y = y_min + 0.05 * y_span

p.line([bar_x_start, bar_x_end], [bar_y, bar_y], line_width=3, line_color="black")
p.add_layout(Label(
    x=(bar_x_start + bar_x_end) / 2, y=bar_y,
    text="2 \u03bcm", text_align="center", text_baseline="bottom",
    text_color="black", text_font_size="10pt", y_offset=2))

p.output_backend = "svg"
export_svgs(p, filename="../../../figures/Figure_2/2543_oligNormal_chromosomes.svg")

# ---- Plot 2: Chromosome arm scatter chr1 ----
df_cell_chr = df_cell.sort_values('chrom_arm', key=lambda x: x.map({'p': 0, 'q': 1}))
source = ColumnDataSource(df_cell_chr)

chrom_factors = ['p', 'q']
color_mapper = factor_cmap('chrom_arm', palette=[palette_hex[1], palette_hex[2]], factors=chrom_factors)

p2 = figure(title="Scatter Plot Colored by Chromosome Arm",
           background_fill_color="white",
           border_fill_color="white",
           x_range=Range1d(start=shared_x_range[0], end=shared_x_range[1]),
           y_range=Range1d(start=shared_y_range[0], end=shared_y_range[1]),
           width=plot_size, height=plot_size)

# Plot all non-chr1 first (grey, background)
for chr_value, color in zip(unique_chrs, palette):
    if chr_value == 'chr1':
        continue
    chr_source = ColumnDataSource(df_cell_chr[df_cell_chr['chromosome_true'] == chr_value])
    p2.scatter('x', 'y', source=chr_source, size=13, fill_alpha=0.8, color='#CCCCCC')

# Plot chr1 last (on top, colored by arm)
chr1_df = df_cell_chr[df_cell_chr['chromosome_true'] == 'chr1']
chr1_source = ColumnDataSource(chr1_df)
p2.scatter('x', 'y', source=chr1_source, size=13, fill_alpha=0.8, color=color_mapper)


#p2.legend.visible = False
p2.xaxis.axis_label = None
p2.yaxis.axis_label = None
p2.xgrid.grid_line_color = None
p2.ygrid.grid_line_color = None
p2.xaxis.major_label_text_font_size = "0pt"
p2.yaxis.major_label_text_font_size = "0pt"
p2.xaxis.major_tick_line_color = None
p2.xaxis.minor_tick_line_color = None
p2.yaxis.major_tick_line_color = None
p2.yaxis.minor_tick_line_color = None
p2.xaxis.axis_line_color = None
p2.yaxis.axis_line_color = None
p2.title.text_color = "white"

p2.line([bar_x_start, bar_x_end], [bar_y, bar_y], line_width=3, line_color="black")
p2.add_layout(Label(
    x=(bar_x_start + bar_x_end) / 2, y=bar_y,
    text="2 \u03bcm", text_align="center", text_baseline="bottom",
    text_color="black", text_font_size="10pt", y_offset=2))

p2.output_backend = "svg"
export_svgs(p2, filename="../../../figures/Figure_2/2543_oligNormal_chr1.svg")

# ---- Plot 3: Chromosome arm scatter chr19 ----
df_cell_chr = df_cell.sort_values('chrom_arm', key=lambda x: x.map({'p': 0, 'q': 1}))
source = ColumnDataSource(df_cell_chr)

chrom_factors = ['p', 'q']
color_mapper = factor_cmap('chrom_arm', palette=[palette_hex[1], palette_hex[2]], factors=chrom_factors)

p3 = figure(title="Scatter Plot Colored by Chromosome Arm",
           background_fill_color="white",
           border_fill_color="white",
           x_range=Range1d(start=shared_x_range[0], end=shared_x_range[1]),
           y_range=Range1d(start=shared_y_range[0], end=shared_y_range[1]),
           width=plot_size, height=plot_size)

# Plot all non-chr1 first (grey, background)
for chr_value, color in zip(unique_chrs, palette):
    if chr_value == 'chr19':
        continue
    chr_source = ColumnDataSource(df_cell_chr[df_cell_chr['chromosome_true'] == chr_value])
    p3.scatter('x', 'y', source=chr_source, size=13, fill_alpha=0.8, color='#CCCCCC')

# Plot chr1 last (on top, colored by arm)
chr19_df = df_cell_chr[df_cell_chr['chromosome_true'] == 'chr19']
chr19_source = ColumnDataSource(chr19_df)
p3.scatter('x', 'y', source=chr19_source, size=13, fill_alpha=0.8, color=color_mapper)


p3.xaxis.axis_label = None
p3.yaxis.axis_label = None
p3.xgrid.grid_line_color = None
p3.ygrid.grid_line_color = None
p3.xaxis.major_label_text_font_size = "0pt"
p3.yaxis.major_label_text_font_size = "0pt"
p3.xaxis.major_tick_line_color = None
p3.xaxis.minor_tick_line_color = None
p3.yaxis.major_tick_line_color = None
p3.yaxis.minor_tick_line_color = None
p3.xaxis.axis_line_color = None
p3.yaxis.axis_line_color = None
p3.title.text_color = "white"

p3.line([bar_x_start, bar_x_end], [bar_y, bar_y], line_width=3, line_color="black")
p3.add_layout(Label(
    x=(bar_x_start + bar_x_end) / 2, y=bar_y,
    text="2 \u03bcm", text_align="center", text_baseline="bottom",
    text_color="black", text_font_size="10pt", y_offset=2))

p3.output_backend = "svg"
export_svgs(p3, filename="../../../figures/Figure_2/2543_oligNormal_chr19.svg")

##### Make Legends #####

## Chromosome Legend

p_legend = figure(width=200, height=600, outline_line_color=None,
                  background_fill_color="white", border_fill_color="white",
                  toolbar_location=None)

# Fix the range and place dummy points outside it
p_legend.x_range = Range1d(0, 1)
p_legend.y_range = Range1d(0, 1)

p_legend.xaxis.visible = False
p_legend.yaxis.visible = False
p_legend.xgrid.grid_line_color = None
p_legend.ygrid.grid_line_color = None

for chr_value, color in zip(unique_chrs, palette):
    p_legend.scatter([-999], [-999], size=13, fill_alpha=0.8, color=color,
                     legend_label=str(chr_value))

p_legend.legend.title = "Chromosome"
p_legend.legend.title_text_color = "black"
p_legend.legend.label_text_color = "black"
p_legend.legend.location = "top_left"
p_legend.legend.border_line_color = None
p_legend.legend.background_fill_alpha = 0.0

p_legend.output_backend = "svg"
export_svgs(p_legend, filename="../../../figures/Figure_2/chromosome_legend.svg")

## Chromosome Arm Legend

p_legend2 = figure(width=200, height=150, outline_line_color=None,
                   background_fill_color="white", border_fill_color="white",
                   toolbar_location=None)

p_legend2.x_range = Range1d(0, 1)
p_legend2.y_range = Range1d(0, 1)

p_legend2.xaxis.visible = False
p_legend2.yaxis.visible = False
p_legend2.xgrid.grid_line_color = None
p_legend2.ygrid.grid_line_color = None

chrom_arm_colors = {'p': palette_hex[1], 'q': palette_hex[2]}
for arm, color in chrom_arm_colors.items():
    p_legend2.scatter([-999], [-999], size=8, fill_alpha=0.6, color=color,
                      legend_label=arm)

p_legend2.legend.title = "Arm"
p_legend2.legend.title_text_color = "black"
p_legend2.legend.label_text_color = "black"
p_legend2.legend.location = "top_left"
p_legend2.legend.border_line_color = None
p_legend2.legend.background_fill_alpha = 0.0

p_legend2.output_backend = "svg"
export_svgs(p_legend2, filename="../../../figures/Figure_2/chrom_arm_legend.svg")