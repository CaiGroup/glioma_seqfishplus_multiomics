import glob
import pandas as pd
import os
import pickle
import numpy as np

os.environ["PATH"] = os.path.expanduser("~/miniconda3/envs/fox_env24/bin") + ":" + os.environ["PATH"]
pd.options.mode.chained_assignment = None

# plotting packages
from bokeh.plotting import figure
from bokeh.io import export_svgs
from bokeh.models import ColumnDataSource, ColorBar, Span, Label, Range1d
from bokeh.transform import factor_cmap
from bokeh.models import Legend
from scipy.spatial import ConvexHull
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

plot_hull = True
plot_other_chr = False

path = '/central/groups/mthomson/sgdata/CaiLab/michalp/Glioma_analysis/data/DNA_data'

# get all files
files = glob.glob(f"{path}/*assigned.csv")

# only keep 1 patient sample
patient_id = '2543'
files = [i for i in files if patient_id in i]

# load data
df_1 = pd.read_csv(files[0], low_memory=False)

# fix chromosomes
probe_df = pd.read_csv("/central/groups/CaiLab/personal/jjfox/DNA_seqFISH/human/24-12-19_human_genome_DNA_seqFISH_final_probes.csv")
df_unique = probe_df.drop_duplicates(subset='bin_name', keep='last')
chr_dict = dict(zip(df_unique['bin_name'], df_unique['chr']))

df_1["blockbin"] = (
    df_1["block_assigned"].str.replace("block_", "block", regex=False)
    + "_" +
    df_1["name"].str.replace("bin_", "bin", regex=False)
)
df_1['chromosome_true'] = df_1.blockbin.map(chr_dict)

# map chromosome arms
with open('chrom_arm_dict.pkl', 'rb') as f:
    chrom_arm_dict = pickle.load(f)

df_1['chrom_arm'] = df_1.blockbin.map(chrom_arm_dict)

# load cell type data
celltype_df = pd.read_csv('/resnick/groups/mthomson/ci2_data/CaiLab/gbm/dna_distance_data/V2_24_patients_Obj_mapped.csv', low_memory=False)
celltype_df.columns = [col.replace('.', '_') for col in celltype_df.columns]
celltype_df.rename(columns={'Unnamed: 0': 'cell_id'}, inplace=True)
gbm_cells = celltype_df.loc[celltype_df['ID'] == patient_id].copy()
gbm_cells['cell_id'] = gbm_cells['cell_id'].apply(lambda x: int(x.split('_')[0]))
gbm_cell_dict = dict(zip(gbm_cells['cell_id'], gbm_cells['predicted_celltype']))
df_1['predicted_celltype'] = df_1['matched_label'].map(gbm_cell_dict)
celltypes = df_1['predicted_celltype'].unique()


def compute_scale_bar(shared_x_range, shared_y_range, scale_bar_length=2, inset_frac=0.04):
    plot_x_min, plot_x_max = shared_x_range
    plot_y_min, plot_y_max = shared_y_range
    plot_x_span = plot_x_max - plot_x_min
    plot_y_span = plot_y_max - plot_y_min
    bar_inset_x = inset_frac * plot_x_span
    bar_inset_y = inset_frac * plot_y_span
    bar_x_end = plot_x_max - bar_inset_x
    bar_x_start = bar_x_end - scale_bar_length
    bar_y = plot_y_min + bar_inset_y
    return bar_x_start, bar_x_end, bar_y


def add_scale_bar(p, bar_x_start, bar_x_end, bar_y):
    p.line([bar_x_start, bar_x_end], [bar_y, bar_y], line_width=3, line_color="black")
    p.add_layout(Label(
        x=(bar_x_start + bar_x_end) / 2, y=bar_y,
        text="2 \u03bcm", text_align="center", text_baseline="bottom",
        text_color="black", text_font_size="10pt", y_offset=2))


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
    p.title.text_color = "white"


def sort_chromosomes(unique_chrs):
    unique_chrs = list(unique_chrs)
    unique_chrs.sort(key=lambda x: x.split('chr')[1])
    unique_chrs_num = unique_chrs[:-2]
    unique_chrs_num.sort(key=lambda x: int(x.split('chr')[1]))
    return unique_chrs_num + [unique_chrs[-2]] + [unique_chrs[-1]]


def compute_hull(df_cell, hull_gap=0.15):
    all_points = np.column_stack([df_cell['x'].values, df_cell['y'].values])
    hull = ConvexHull(all_points)
    hull_verts = all_points[hull.vertices]
    centroid = hull_verts.mean(axis=0)
    directions = hull_verts - centroid
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    hull_verts_expanded = hull_verts + directions * hull_gap
    hull_verts_closed = np.vstack([hull_verts_expanded, hull_verts_expanded[0]])
    tck, u = splprep([hull_verts_closed[:, 0], hull_verts_closed[:, 1]], s=0, per=True, k=3)
    u_fine = np.linspace(0, 1, 300)
    hull_xs, hull_ys = splev(u_fine, tck)
    return list(hull_xs), list(hull_ys)


def compute_shared_range(df_cell, pad_frac=0.05):
    x_min, x_max = df_cell['x'].min(), df_cell['x'].max()
    y_min, y_max = df_cell['y'].min(), df_cell['y'].max()
    x_span = x_max - x_min
    y_span = y_max - y_min
    max_span = max(x_span, y_span)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    pad = pad_frac * max_span
    shared_x_range = (x_center - max_span/2 - pad, x_center + max_span/2 + pad)
    shared_y_range = (y_center - max_span/2 - pad, y_center + max_span/2 + pad)
    return shared_x_range, shared_y_range


def plot_cell(df_1, cell, label, out_dir, plot_hull=True, plot_other_chr=True, plot_size=670, scale_bar_length=2):
    df_cell = df_1.loc[df_1['matched_label'] == cell].copy()
    df_cell = df_cell[df_cell['chromosome_true'].notna()]

    shared_x_range, shared_y_range = compute_shared_range(df_cell)
    hull_xs, hull_ys = compute_hull(df_cell)
    bar_x_start, bar_x_end, bar_y = compute_scale_bar(shared_x_range, shared_y_range, scale_bar_length)

    unique_chrs = sort_chromosomes(df_cell['chromosome_true'].unique())
    palette = palette_hex[:len(unique_chrs)]

    # Plot 1: Chromosome scatter
    p = figure(title=f"Scatterplot by Chromosome Cell {cell}",
               width=plot_size, height=plot_size,
               background_fill_color="white",
               border_fill_color="white",
               outline_line_color=None,
               x_range=Range1d(start=shared_x_range[0], end=shared_x_range[1]),
               y_range=Range1d(start=shared_y_range[0], end=shared_y_range[1]))

    for chr_value, color in zip(unique_chrs, palette):
        chr_source = ColumnDataSource(df_cell[df_cell['chromosome_true'] == chr_value])
        p.scatter('x', 'y', source=chr_source, size=13, fill_alpha=0.8, color=color)

    if plot_hull:
        p.line(hull_xs, hull_ys, line_width=1.5, line_color="black", line_dash="dashed")
    
    clean_axes(p)
    add_scale_bar(p, bar_x_start, bar_x_end, bar_y)
    p.output_backend = "svg"
    export_svgs(p, filename=f"{out_dir}/{label}_chromosomes.svg")

    # Plot 2: chr arm scatter
    chrom_factors = ['p', 'q']
    color_mapper_arms = factor_cmap('chrom_arm', palette=[palette_hex[1], palette_hex[2]], factors=chrom_factors)

    for highlight_chr, plot_label in [('chr1', 'chr1'), ('chr19', 'chr19')]:
        df_cell_chr = df_cell.sort_values('chrom_arm', key=lambda x: x.map({'p': 0, 'q': 1}))

        p_arm = figure(title=f"Scatter Plot Colored by Chromosome Arm",
                       background_fill_color="white",
                       border_fill_color="white",
                       outline_line_color=None,
                       x_range=Range1d(start=shared_x_range[0], end=shared_x_range[1]),
                       y_range=Range1d(start=shared_y_range[0], end=shared_y_range[1]),
                       width=plot_size, height=plot_size)

        for chr_value in unique_chrs:
            if chr_value == highlight_chr:
                continue
            chr_source = ColumnDataSource(df_cell_chr[df_cell_chr['chromosome_true'] == chr_value])
            bg_color = '#CCCCCC' if plot_other_chr else '#FFFFFF'
            p_arm.scatter('x', 'y', source=chr_source, size=13, fill_alpha=0.8, color=bg_color)

        highlight_df = df_cell_chr[df_cell_chr['chromosome_true'] == highlight_chr]
        highlight_source = ColumnDataSource(highlight_df)
        color_mapper = factor_cmap('chrom_arm', palette=[palette_hex[1], palette_hex[2]], factors=chrom_factors)
        p_arm.scatter('x', 'y', source=highlight_source, size=13, fill_alpha=0.8, color=color_mapper)

        if plot_hull:
            p_arm.line(hull_xs, hull_ys, line_width=1.5, line_color="black", line_dash="dashed")

        clean_axes(p_arm)
        add_scale_bar(p_arm, bar_x_start, bar_x_end, bar_y)
        p_arm.output_backend = "svg"
        export_svgs(p_arm, filename=f"{out_dir}/{label}_{plot_label}.svg")


os.makedirs("../../../figures/Figure_2/", exist_ok=True)
out_dir = "../../../figures/Figure_2"

plot_cell(df_1, cell=7763, label="2543_OPC",        out_dir=out_dir, plot_hull=plot_hull, plot_other_chr=plot_other_chr)
plot_cell(df_1, cell=2057, label="2543_oligNormal", out_dir=out_dir, plot_hull=plot_hull, plot_other_chr=plot_other_chr)


##### Make Legends #####

unique_chrs_legend = sort_chromosomes(df_1['chromosome_true'].dropna().unique())
palette_legend = palette_hex[:len(unique_chrs_legend)]

## Chromosome Legend
p_legend = figure(width=200, height=600, outline_line_color=None,
                  background_fill_color="white", border_fill_color="white",
                  toolbar_location=None)

p_legend.x_range = Range1d(0, 1)
p_legend.y_range = Range1d(0, 1)
p_legend.xaxis.visible = False
p_legend.yaxis.visible = False
p_legend.xgrid.grid_line_color = None
p_legend.ygrid.grid_line_color = None

for chr_value, color in zip(unique_chrs_legend, palette_legend):
    p_legend.scatter([-999], [-999], size=13, fill_alpha=0.8, color=color,
                     legend_label=str(chr_value))

p_legend.legend.title = "Chromosome"
p_legend.legend.title_text_color = "black"
p_legend.legend.label_text_color = "black"
p_legend.legend.location = "top_left"
p_legend.legend.border_line_color = None
p_legend.legend.background_fill_alpha = 0.0
p_legend.output_backend = "svg"
export_svgs(p_legend, filename=f"{out_dir}/chromosome_legend.svg")

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
export_svgs(p_legend2, filename=f"{out_dir}/chrom_arm_legend.svg")