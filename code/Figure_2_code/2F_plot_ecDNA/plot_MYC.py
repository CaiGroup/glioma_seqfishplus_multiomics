import glob
import os
import pickle

import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.io import export_svgs
from bokeh.models import ColumnDataSource, Label, Range1d
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev

os.environ["PATH"] = os.path.expanduser("~/miniconda3/envs/fox_env24/bin") + ":" + os.environ["PATH"]
pd.options.mode.chained_assignment = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PALETTE_HEX = [
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

DATA_PATH = "/home/jjfox/resnick_ci2_gbm_/Data/DNA_data/"
PROBE_CSV = "/resnick/groups/CaiLab/jjfox/copied_from_central/DNA_seqFISH/human/24-12-19_human_genome_DNA_seqFISH_final_probes.csv"
CELLTYPE_CSV = "/resnick/groups/mthomson/ci2_data/CaiLab/gbm/dna_distance_data/V2_24_patients_Obj_mapped.csv"
CHROM_ARM_DICT_PKL = "chrom_arm_dict.pkl"
OUT_DIR = "../../../figures/Figure_2/Figure_2E"

PLOT_HULL = False
PLOT_OTHER_CHR = True
PLOT_SIZE = 670
SCALE_BAR_LENGTH = 2

HIGHLIGHT_CHRS = ["chr8"]
ECDNA_NAME = "block34_bin7"
ECDNA_COLOR = PALETTE_HEX[41]
CHR7_COLOR = PALETTE_HEX[6]
BG_COLOR = "#CCCCCC"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def sort_chromosomes(unique_chrs):
    """Sort chromosome names numerically, with chrX and chrY at the end."""
    numeric, other = [], []
    for c in unique_chrs:
        suffix = c.split("chr")[1]
        if suffix.isdigit():
            numeric.append(c)
        else:
            other.append(c)
    numeric.sort(key=lambda x: int(x.split("chr")[1]))
    other.sort()
    return numeric + other


def compute_shared_range(df_cell, pad_frac=0.05):
    """Return square (x_range, y_range) centered on the data."""
    x_min, x_max = df_cell["x"].min(), df_cell["x"].max()
    y_min, y_max = df_cell["y"].min(), df_cell["y"].max()
    max_span = max(x_max - x_min, y_max - y_min)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    pad = pad_frac * max_span
    half = max_span / 2 + pad
    return (x_center - half, x_center + half), (y_center - half, y_center + half)


def compute_hull(df_cell, hull_gap=0.15):
    """Return smoothed, slightly expanded convex hull coordinates."""
    points = np.column_stack([df_cell["x"].values, df_cell["y"].values])
    hull = ConvexHull(points)
    verts = points[hull.vertices]
    centroid = verts.mean(axis=0)
    directions = verts - centroid
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    verts_expanded = verts + directions * hull_gap
    verts_closed = np.vstack([verts_expanded, verts_expanded[0]])
    tck, _ = splprep([verts_closed[:, 0], verts_closed[:, 1]], s=0, per=True, k=3)
    hull_xs, hull_ys = splev(np.linspace(0, 1, 300), tck)
    return list(hull_xs), list(hull_ys)


def compute_scale_bar(x_range, y_range, bar_length=2, inset_frac=0.04):
    """Return (x_start, x_end, y) for a scale bar in the lower-right corner."""
    x_span = x_range[1] - x_range[0]
    y_span = y_range[1] - y_range[0]
    x_end = x_range[1] - inset_frac * x_span
    x_start = x_end - bar_length
    y = y_range[0] + inset_frac * y_span
    return x_start, x_end, y


def _make_base_figure(title, x_range, y_range, size=PLOT_SIZE):
    """Create a clean Bokeh figure with hidden axes and white background."""
    p = figure(
        title=title, width=size, height=size,
        background_fill_color="white", border_fill_color="white",
        outline_line_color=None,
        x_range=Range1d(start=x_range[0], end=x_range[1]),
        y_range=Range1d(start=y_range[0], end=y_range[1]),
    )
    p.xaxis.axis_label = None
    p.yaxis.axis_label = None
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    for axis in (p.xaxis, p.yaxis):
        axis.major_label_text_font_size = "0pt"
        axis.major_tick_line_color = None
        axis.minor_tick_line_color = None
        axis.axis_line_color = None
    p.title.text_color = "white"
    return p


def _add_scale_bar(p, x_range, y_range, bar_length=SCALE_BAR_LENGTH):
    """Draw a scale bar with label on the figure."""
    x_start, x_end, y = compute_scale_bar(x_range, y_range, bar_length)
    p.line([x_start, x_end], [y, y], line_width=3, line_color="black")
    p.add_layout(Label(
        x=(x_start + x_end) / 2, y=y,
        text=f"{bar_length} \u03bcm",
        text_align="center", text_baseline="bottom",
        text_color="black", text_font_size="10pt", y_offset=2,
    ))


def _add_hull(p, hull_xs, hull_ys):
    p.line(hull_xs, hull_ys, line_width=1.5, line_color="black", line_dash="dashed")


def _export_svg(p, path):
    p.output_backend = "svg"
    export_svgs(p, filename=path)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_cell(df, cell, label, out_dir,
              plot_hull=True, plot_other_chr=True):
    """Generate chromosome and ecDNA scatter SVGs for a single cell."""
    df_cell = df.loc[df["matched_label"] == cell].copy()
    df_notchr = df_cell[df_cell["chromosome_true"].isna()]
    df_cell = df_cell[df_cell["chromosome_true"].notna()]

    x_range, y_range = compute_shared_range(df_cell)
    hull_xs, hull_ys = compute_hull(df_cell)
    unique_chrs = sort_chromosomes(df_cell["chromosome_true"].unique())
    palette = PALETTE_HEX[: len(unique_chrs)]

    # --- Chromosome scatter ---
    p = _make_base_figure(f"Scatterplot by Chromosome Cell {cell}", x_range, y_range)
    for chr_val, color in zip(unique_chrs, palette):
        src = ColumnDataSource(df_cell[df_cell["chromosome_true"] == chr_val])
        p.scatter("x", "y", source=src, size=13, fill_alpha=0.8, color=color)
    if plot_hull:
        _add_hull(p, hull_xs, hull_ys)
    _add_scale_bar(p, x_range, y_range)
    _export_svg(p, f"{out_dir}/{label}_chromosomes.svg")

    # --- ecDNA scatter ---
    p2 = _make_base_figure(f"ecDNA Scatterplot {cell}", x_range, y_range)
    bg = BG_COLOR if plot_other_chr else "#FFFFFF"
    for chr_val in unique_chrs:
        if chr_val in HIGHLIGHT_CHRS:
            continue
        src = ColumnDataSource(df_cell[df_cell["chromosome_true"] == chr_val])
        p2.scatter("x", "y", source=src, size=13, fill_alpha=0.8, color=bg)
    for chr_val, color in zip(unique_chrs, palette):
        if chr_val not in HIGHLIGHT_CHRS:
            continue
        src = ColumnDataSource(df_cell[df_cell["chromosome_true"] == chr_val])
        p2.scatter("x", "y", source=src, size=13, fill_alpha=0.8, color=color)
    ecdna_src = ColumnDataSource(df_cell[df_cell["blockbin"] == ECDNA_NAME])
    p2.scatter("x", "y", source=ecdna_src, size=13, fill_alpha=0.8, color=ECDNA_COLOR)
    if plot_hull:
        _add_hull(p2, hull_xs, hull_ys)
    _add_scale_bar(p2, x_range, y_range)
    _export_svg(p2, f"{out_dir}/{label}_ecDNA.svg")


# ---------------------------------------------------------------------------
# Legend helpers
# ---------------------------------------------------------------------------
def _make_legend_figure(width, height):
    p = figure(
        width=width, height=height, outline_line_color=None,
        background_fill_color="white", border_fill_color="white",
        toolbar_location=None,
    )
    p.x_range = Range1d(0, 1)
    p.y_range = Range1d(0, 1)
    p.xaxis.visible = False
    p.yaxis.visible = False
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    return p


def export_chromosome_legend(unique_chrs, palette, out_dir):
    p = _make_legend_figure(200, 600)
    for chr_val, color in zip(unique_chrs, palette):
        p.scatter([-999], [-999], size=13, fill_alpha=0.8, color=color,
                  legend_label=str(chr_val))
    p.legend.title = "Chromosome"
    p.legend.title_text_color = "black"
    p.legend.label_text_color = "black"
    p.legend.location = "top_left"
    p.legend.border_line_color = None
    p.legend.background_fill_alpha = 0.0
    _export_svg(p, f"{out_dir}/chromosome_legend.svg")


def export_ecdna_legend(out_dir):
    p = _make_legend_figure(200, 150)
    for label, color in {"chr7": CHR7_COLOR, "EGFR": ECDNA_COLOR}.items():
        p.scatter([-999], [-999], size=8, fill_alpha=0.6, color=color,
                  legend_label=label)
    p.legend.title = "Chromosome"
    p.legend.title_text_color = "black"
    p.legend.label_text_color = "black"
    p.legend.location = "top_left"
    p.legend.border_line_color = None
    p.legend.background_fill_alpha = 0.0
    _export_svg(p, f"{out_dir}/ecDNA_legend.svg")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def build_chr_dict():
    probe_df = pd.read_csv(PROBE_CSV)
    df_unique = probe_df.drop_duplicates(subset="bin_name", keep="last")
    return dict(zip(df_unique["bin_name"], df_unique["chr"]))


def load_chrom_arm_dict():
    with open(CHROM_ARM_DICT_PKL, "rb") as f:
        return pickle.load(f)


def load_patient(patient_id, chr_dict, chrom_arm_dict):
    """Load and annotate DNA seqFISH data for one patient."""
    files = [f for f in glob.glob(f"{DATA_PATH}/*assigned.csv") if patient_id in f]
    df = pd.read_csv(files[0], low_memory=False)

    df["blockbin"] = (
        df["block_assigned"].str.replace("block_", "block", regex=False)
        + "_"
        + df["name"].str.replace("bin_", "bin", regex=False)
    )
    df["chromosome_true"] = df["blockbin"].map(chr_dict)
    df["chrom_arm"] = df["blockbin"].map(chrom_arm_dict)

    celltype_df = pd.read_csv(CELLTYPE_CSV, low_memory=False)
    celltype_df.columns = [col.replace(".", "_") for col in celltype_df.columns]
    celltype_df.rename(columns={"Unnamed: 0": "cell_id"}, inplace=True)
    gbm_cells = celltype_df.loc[celltype_df["ID"] == patient_id].copy()
    gbm_cells["cell_id"] = gbm_cells["cell_id"].apply(lambda x: int(x.split("_")[0]))
    cell_dict = dict(zip(gbm_cells["cell_id"], gbm_cells["predicted_celltype"]))
    df["predicted_celltype"] = df["matched_label"].map(cell_dict)

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    chr_dict = build_chr_dict()
    chrom_arm_dict = load_chrom_arm_dict()

    # Patient 3883 (primary)
    df_3883 = load_patient("3557", chr_dict, chrom_arm_dict)
    plot_cell(df_3883, cell=5091, label="MYC_3557_OPC_1", out_dir=OUT_DIR,
              plot_hull=PLOT_HULL, plot_other_chr=PLOT_OTHER_CHR)
    plot_cell(df_3883, cell=15082, label="MYC_3557_OPC_2", out_dir=OUT_DIR,
              plot_hull=PLOT_HULL, plot_other_chr=PLOT_OTHER_CHR)

    # Patient 5597 (recurrence)
    df_5597 = load_patient("2683", chr_dict, chrom_arm_dict)
    plot_cell(df_5597, cell=9636, label="MYC_2683_OPC_1", out_dir=OUT_DIR,
              plot_hull=PLOT_HULL, plot_other_chr=PLOT_OTHER_CHR)
    plot_cell(df_5597, cell=8276, label="MYC_2683_OPC_2", out_dir=OUT_DIR,
              plot_hull=PLOT_HULL, plot_other_chr=PLOT_OTHER_CHR)

    # Legends
    unique_chrs = sort_chromosomes(df_5597["chromosome_true"].dropna().unique())
    export_chromosome_legend(unique_chrs, PALETTE_HEX[: len(unique_chrs)], OUT_DIR)
    export_ecdna_legend(OUT_DIR)


if __name__ == "__main__":
    main()