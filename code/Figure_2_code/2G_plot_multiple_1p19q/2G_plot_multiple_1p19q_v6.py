import glob
import json
import os
import pickle
import tempfile
import zipfile
from functools import lru_cache

import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.io import export_svgs
from bokeh.models import ColumnDataSource, Range1d
from matplotlib.path import Path
from scipy.optimize import minimize
from tqdm import tqdm

os.environ["PATH"] = os.path.expanduser("~/miniconda3/envs/fox_env24/bin") + ":" + os.environ["PATH"]
pd.options.mode.chained_assignment = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PALETTE_HEX = [
    "#73b9a5", "#a33f3e", "#fea8d4", "#fea17b", "#c5aedf",
    "#d4af9e", "#2261b2", "#7e5679", "#6ea9fd", "#bd8985",
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
]      # this is an edited  palette to replace integer positions 2 with 6. makes chr3 pink and chr7 blue

TUMOR_CELLTYPES = {"OPC", "AC.like", "OPC.like Prolif."}
TUMOR_COLOR = "#FF0000"
NONTUMOR_COLOR = "#808080"

DATA_PATH = "/home/jjfox/resnick_ci2_gbm_/Data/DNA_data/"
PROBE_CSV = "/resnick/groups/CaiLab/jjfox/copied_from_central/DNA_seqFISH/human/24-12-19_human_genome_DNA_seqFISH_final_probes.csv"
CELLTYPE_CSV = "/resnick/groups/mthomson/ci2_data/CaiLab/gbm/dna_distance_data/V2_24_patients_Obj_mapped.csv"
CHROM_ARM_DICT_PKL = "chrom_arm_dict.pkl"
ANALYSIS_DIR = "/resnick/groups/mthomson/ci2_data/CaiLab/gbm/V2_runs/analysis"
OUT_DIR = "../../../figures/Figure_2/Figure_2C"

PLOT_HULL = True
PLOT_OTHER_CHR = True
# Toggle the large-region (REGIONS) plots on/off. Useful when iterating on
# just the per-cell zoom-ins, since the large regions are slow.
PLOT_LARGE_REGIONS = False

# Final figure size in points (1 pt = 1/72 inch). Each scatter is rendered as
# a PNG at TARGET_PT * DPI / 72 pixels on the long edge, so at 99 pt and
# 1200 DPI that is ~1650 px. The vector overlay (hulls + scale bar) is
# rendered at the native point size as SVG.
TARGET_PT = 99
DPI = 1200
SCATTER_LONG_EDGE_PX = int(round(TARGET_PT * DPI / 72))  # ~1650 px at 1200 DPI

SCALE_BAR_LENGTH = 2
# Default scatter dot size in points (at the final TARGET_PT figure size).
# This value reproduces the old large-region dot size: the previous pipeline
# rendered a ~4000 px scatter with size=11.2 px (POINT_SIZE=0.7 * RENDER_SCALE=16),
# which is ~0.28% of the long edge -> ~0.277 pt at a 99 pt figure.
POINT_SIZE_PT = 0.277
# Cell zoom-ins use a larger dot since each region is only 20 um wide; this
# multiplier is applied on top of POINT_SIZE_PT.
CELL_POINT_SIZE_MULT = 8.0

BG_COLOR = "#CCCCCC"

# Arms to highlight, with colors. Each entry is (chromosome, [(arm, color), ...])
# Using the same red/blue pair as your other figures so p and q are consistent.
ARM_HIGHLIGHTS = [
    ("chr1",  [("p", "#864429"), ("q", "#de8a5c")]),
    ("chr19", [("p", "#864429"), ("q", "#de8a5c")]),
]

# Whole-chromosome highlights (no arm split). Each entry is (chromosome, color).
CHR_HIGHLIGHTS = [
    ("chr7", "#2261b2"),
]

PIXELS_PER_MICRON = 9.62

PATIENT_ID = "2122"

REGIONS = [
    ("region1", (2150, 2450), (650, 950)),
    ("region2", (2800, 3100), (750, 1050)),
]

# Per-cell zoom-in regions. Each entry is (label, x_range, y_range).
# Each cell gets the same set of plots as the main region (all chromosomes,
# chr1 arms, chr19 arms, chr7), with filenames prefixed by the cell label.
CELL_REGIONS = [
    ("tumor1",  (2273, 2293), (752, 772)),
    ("tumor2",  (2811, 2831), (788, 808)),
    ("tumor3",  (3037, 3057), (992, 1012)),
    ("normal1", (2214, 2234), (797, 817)),
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def should_rotate(x_range, y_range):
    """Rotate iff the region is taller than wide (portrait), so the rendered
    plot is always landscape (wider than tall)."""
    return (y_range[1] - y_range[0]) > (x_range[1] - x_range[0])


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


def load_geojson_from_zip(zip_path):
    """Extract and parse the first GeoJSON file found inside a zip archive."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, "r") as z:
            geojson_files = [f for f in z.namelist() if f.endswith(".geojson")]
            if not geojson_files:
                raise FileNotFoundError("No .geojson file found in the archive.")
            z.extract(geojson_files[0], tmpdir)
            with open(os.path.join(tmpdir, geojson_files[0])) as f:
                data = json.load(f)
    return data


@lru_cache(maxsize=None)
def get_segmentation_geojson(patient_id):
    """Return the parsed segmentation GeoJSON for a patient.

    Cached: the (large) zip is loaded and parsed only once per patient even
    when called from many regions.
    """
    if patient_id == '4494_1':
        geofiles = glob.glob('/resnick/groups/mthomson/ci2_data/CaiLab/gbm/V2_runs/analysis/25-04-04_3097_4494_5538_dna_analysis/roi2_4494/segmentations/segmentation_nuclei.zip')
    else:
        geofiles = glob.glob(
            f"{ANALYSIS_DIR}/*{patient_id}*/roi*{patient_id}/segmentations/segmentation_nuclei.zip"
        )
    if not geofiles:
        raise FileNotFoundError(f"No segmentation zip found for patient {patient_id}")
    return load_geojson_from_zip(geofiles[-1])


def load_nucleus_hull_from_geojson(geojson, dna_cell_id, points_x, points_y):
    """
    Look up the nucleus polygon in a preloaded GeoJSON, convert to microns,
    and register it to the point cloud by finding the (dx, dy) translation
    that maximizes spots inside the polygon.
    """
    cell_mask = next(
        (f for f in geojson["features"] if f["id"] == str(dna_cell_id)),
        None,
    )
    if cell_mask is None:
        raise KeyError(f"Cell ID {dna_cell_id} not found in segmentation GeoJSON")

    coords = cell_mask["geometry"]["coordinates"][0]
    xs, ys = zip(*coords)

    xs_um = np.array(xs) / PIXELS_PER_MICRON
    ys_um = np.array(ys) / PIXELS_PER_MICRON

    points = np.column_stack([np.array(points_x), np.array(points_y)])

    def n_outside(offset):
        dx, dy = offset
        shifted_verts = np.column_stack([xs_um + dx, ys_um + dy])
        path = Path(shifted_verts)
        inside = path.contains_points(points)
        return -inside.sum()

    x0_dx = np.median(points_x) - xs_um.mean()
    x0_dy = np.median(points_y) - ys_um.mean()

    result = minimize(
        n_outside,
        x0=[x0_dx, x0_dy],
        method="Nelder-Mead",
        options={"xatol": 0.01, "fatol": 0.5, "maxiter": 2000},
    )

    dx, dy = result.x
    return tuple(xs_um + dx), tuple(ys_um + dy)


def select_target_cell(df_region, geojson):
    """For each cell whose spots fall in df_region, register its nucleus
    polygon to the spots and count how many spots fall inside the registered
    polygon. Return (target_label, hull_xs, hull_ys) for the winner -- the
    cell with the most spots inside its own nucleus polygon.

    Cells whose nucleus can't be located in the GeoJSON are skipped.
    """
    best = None  # (count, label, hull_xs, hull_ys)
    for cell_label in df_region["matched_label"].unique():
        df_cell = df_region[df_region["matched_label"] == cell_label]
        dna_cell_id = df_cell["cell"].iloc[0]
        points_x = df_cell["x"].values
        points_y = df_cell["y"].values
        try:
            hull_xs, hull_ys = load_nucleus_hull_from_geojson(
                geojson, dna_cell_id,
                points_x=points_x, points_y=points_y,
            )
        except (KeyError, FileNotFoundError):
            continue

        path = Path(np.column_stack([hull_xs, hull_ys]))
        n_inside = int(path.contains_points(
            np.column_stack([points_x, points_y])
        ).sum())

        if best is None or n_inside > best[0]:
            best = (n_inside, cell_label, hull_xs, hull_ys)

    if best is None:
        raise RuntimeError("No cell with a usable nucleus polygon found in region")

    _, target_label, hull_xs, hull_ys = best
    return target_label, hull_xs, hull_ys


def compute_scale_bar(x_range, y_range, bar_length=2, inset_frac=0.04):
    """Return (x_start, x_end, y) for a scale bar in the lower-right corner."""
    x_span = x_range[1] - x_range[0]
    y_span = y_range[1] - y_range[0]
    x_end = x_range[1] - inset_frac * x_span
    x_start = x_end - bar_length
    y = y_range[0] + inset_frac * y_span
    return x_start, x_end, y


def _make_base_figure(x_range, y_range, long_edge_px=TARGET_PT, rotate=False):
    """Create a figure whose long edge is `long_edge_px` and whose data aspect
    is preserved. If `rotate=True`, original-x is plotted on the figure's
    vertical axis and original-y on the figure's horizontal axis (90 deg CCW).
    """
    x_span = x_range[1] - x_range[0]
    y_span = y_range[1] - y_range[0]

    if rotate:
        fig_x_range = y_range
        fig_y_range = x_range
        fig_x_span, fig_y_span = y_span, x_span
    else:
        fig_x_range = x_range
        fig_y_range = y_range
        fig_x_span, fig_y_span = x_span, y_span

    width = int(round(long_edge_px))
    height = int(round(width * (fig_y_span / fig_x_span)))

    p = figure(
        width=width, height=height,
        background_fill_color="white", border_fill_color="white",
        outline_line_color=None,
        x_range=Range1d(start=fig_x_range[0], end=fig_x_range[1]),
        y_range=Range1d(start=fig_y_range[0], end=fig_y_range[1]),
        toolbar_location=None,
        match_aspect=True,
        min_border=0,
    )
    p.xaxis.visible = False
    p.yaxis.visible = False
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.title.text = ""
    return p


def _add_scale_bar(p, x_range, y_range, rotate=False, bar_length=SCALE_BAR_LENGTH):
    """Draw a scale bar in the bottom-right of the figure. Inputs are the
    original (pre-rotation) data ranges."""
    if rotate:
        fig_x_range = y_range
        fig_y_range = x_range
    else:
        fig_x_range = x_range
        fig_y_range = y_range
    x_start, x_end, y = compute_scale_bar(fig_x_range, fig_y_range, bar_length)
    p.line([x_start, x_end], [y, y], line_width=3, line_color="black")


def _add_hull(p, hull_xs, hull_ys, line_color="black"):
    p.line(hull_xs, hull_ys, line_width=1.5, line_color=line_color, line_dash="dashed")


def _export_svg(p, path):
    p.output_backend = "svg"
    export_svgs(p, filename=path)


def _rasterize_svg(svg_path, png_path, dpi=DPI):
    """Rasterize an SVG file to PNG using cairosvg."""
    import cairosvg
    cairosvg.svg2png(url=svg_path, write_to=png_path, dpi=dpi)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_region(df, x_range, y_range, out_dir, patient_id,
                celltype_to_color,
                plot_hull=True, plot_other_chr=True,
                plot_highlights=True,
                point_size_pt=POINT_SIZE_PT,
                target_cell_label=None,
                target_cell_hull=None,
                name_prefix=""):
    """
    Plot all cells whose spots fall within (x_range, y_range). The output is
    always rendered horizontally (long edge = width); if the region is
    portrait (taller than wide), the figure is rotated 90 deg CCW.

    If `name_prefix` is provided, it is prepended to every output filename
    (with an underscore separator), so multiple regions can be written to the
    same out_dir without collision.

    If `plot_highlights` is False, only the all-chromosomes scatter is
    written; the per-chromosome (CHR_HIGHLIGHTS) and per-arm (ARM_HIGHLIGHTS)
    panels are skipped. Useful for large overview regions where the
    highlight plots aren't needed.

    If `target_cell_label` is provided, only spots from that cell's
    matched_label are scattered, and only that cell's hull is drawn. If
    `target_cell_hull=(hull_xs, hull_ys)` is also provided, that registered
    polygon is used directly; otherwise it is computed from the GeoJSON.

    Outputs (filenames shown without the optional prefix):
      - region_chromosomes_scatter.png / _overlay.svg  (all chromosomes)
      - region_chr_{chr}_scatter.png / _overlay.svg, one per chromosome
        in CHR_HIGHLIGHTS (whole chromosome highlighted, all else gray)
      - region_arms_{chr}_scatter.png / _overlay.svg, one per chromosome
        in ARM_HIGHLIGHTS (p arm and q arm colored, all else gray)
    """
    df_region = df[
        (df["x"] >= x_range[0]) & (df["x"] <= x_range[1]) &
        (df["y"] >= y_range[0]) & (df["y"] <= y_range[1])
    ].copy()
    df_region = df_region[df_region["chromosome_true"].notna()]

    if target_cell_label is not None:
        df_region = df_region[df_region["matched_label"] == target_cell_label]
        if len(df_region) == 0:
            raise ValueError(
                f"No spots remain after filtering region to "
                f"matched_label={target_cell_label}"
            )

    rotate = should_rotate(x_range, y_range)
    fx, fy = ("y", "x") if rotate else ("x", "y")

    unique_chrs = sort_chromosomes(df_region["chromosome_true"].unique())
    palette = PALETTE_HEX[: len(unique_chrs)]
    chr_to_color = dict(zip(unique_chrs, palette))

    # Precompute hulls per cell (so we don't reload the GeoJSON repeatedly).
    # If a target cell + hull is provided, use it directly and skip the
    # per-cell registration loop.
    hulls = {}
    if plot_hull:
        if target_cell_label is not None and target_cell_hull is not None:
            df_cell = df_region[df_region["matched_label"] == target_cell_label]
            ct = (df_cell["predicted_celltype"].iloc[0]
                  if len(df_cell) else None)
            color = celltype_to_color.get(ct, NONTUMOR_COLOR)
            hull_xs, hull_ys = target_cell_hull
            hulls[target_cell_label] = (hull_xs, hull_ys, color)
        else:
            geojson = get_segmentation_geojson(patient_id)
            cell_labels = df_region["matched_label"].unique()
            hull_desc = f"hulls [{name_prefix or 'region'}]"
            for cell_label in tqdm(cell_labels, desc=hull_desc, leave=False, unit="cell"):
                df_cell = df_region[df_region["matched_label"] == cell_label]
                dna_cell_id = df_cell["cell"].iloc[0]
                try:
                    hull_xs, hull_ys = load_nucleus_hull_from_geojson(
                        geojson, dna_cell_id,
                        points_x=df_cell["x"].values,
                        points_y=df_cell["y"].values,
                    )
                    ct = df_cell["predicted_celltype"].iloc[0]
                    color = celltype_to_color.get(ct, NONTUMOR_COLOR)
                    hulls[cell_label] = (hull_xs, hull_ys, color)
                except (KeyError, FileNotFoundError) as e:
                    print(f"Skipping hull for cell {cell_label} ({dna_cell_id}): {e}")

    def _draw_hulls(p):
        for hull_xs, hull_ys, hull_color in hulls.values():
            if rotate:
                _add_hull(p, hull_ys, hull_xs, line_color=hull_color)
            else:
                _add_hull(p, hull_xs, hull_ys, line_color=hull_color)

    def _make_overlay(name):
        """Vector overlay: hulls + scale bar only, at native (1x) size."""
        p = _make_base_figure(x_range, y_range, rotate=rotate)
        if plot_hull:
            _draw_hulls(p)
        _add_scale_bar(p, x_range, y_range, rotate=rotate)
        prefixed = f"{name_prefix}_{name}" if name_prefix else name
        _export_svg(p, f"{out_dir}/{prefixed}_overlay.svg")

    def _export_scatter(p, name):
        prefixed = f"{name_prefix}_{name}" if name_prefix else name
        scatter_svg = f"{out_dir}/{prefixed}_scatter.svg"
        scatter_png = f"{out_dir}/{prefixed}_scatter.png"
        _export_svg(p, scatter_svg)
        _rasterize_svg(scatter_svg, scatter_png)
        os.remove(scatter_svg)

    # Scatter point size in *pixels* at the rendered raster resolution. This
    # matches a point_size_pt diameter at the final TARGET_PT figure size.
    scatter_size_px = point_size_pt * SCATTER_LONG_EDGE_PX / TARGET_PT

    # --- Chromosome scatter (all chromosomes) ---
    p = _make_base_figure(x_range, y_range,
                          long_edge_px=SCATTER_LONG_EDGE_PX, rotate=rotate)
    for chr_val, color in zip(unique_chrs, palette):
        src = ColumnDataSource(df_region[df_region["chromosome_true"] == chr_val])
        p.scatter(fx, fy, source=src,
                  size=scatter_size_px,
                  fill_alpha=0.8, color=color)
    _export_scatter(p, "region_chromosomes")
    _make_overlay("region_chromosomes")

    if not plot_highlights:
        return

    # --- Whole-chromosome highlights (no arm split) ---
    for highlight_chr, highlight_color in CHR_HIGHLIGHTS:
        if highlight_chr not in unique_chrs:
            print(f"Skipping {highlight_chr} chromosome plot: not present in region")
            continue

        p_chr = _make_base_figure(x_range, y_range,
                                  long_edge_px=SCATTER_LONG_EDGE_PX, rotate=rotate)
        bg = BG_COLOR if plot_other_chr else "#FFFFFF"

        # Background: every chromosome except the highlighted one, drawn in gray
        for chr_val in unique_chrs:
            if chr_val == highlight_chr:
                continue
            src = ColumnDataSource(df_region[df_region["chromosome_true"] == chr_val])
            p_chr.scatter(fx, fy, source=src,
                          size=scatter_size_px,
                          fill_alpha=0.8, color=bg)

        # Highlighted chromosome on top
        sub = df_region[df_region["chromosome_true"] == highlight_chr]
        if len(sub) > 0:
            p_chr.scatter(fx, fy, source=ColumnDataSource(sub),
                          size=scatter_size_px,
                          fill_alpha=0.8, color=highlight_color)

        name = f"region_chr_{highlight_chr}"
        _export_scatter(p_chr, name)
        _make_overlay(name)

    # --- Arm scatter: one panel per highlighted chromosome (chr1, chr19) ---
    for highlight_chr, arm_colors in ARM_HIGHLIGHTS:
        if highlight_chr not in unique_chrs:
            print(f"Skipping {highlight_chr} arm plot: not present in region")
            continue

        p2 = _make_base_figure(x_range, y_range,
                               long_edge_px=SCATTER_LONG_EDGE_PX, rotate=rotate)
        bg = BG_COLOR if plot_other_chr else "#FFFFFF"

        # Background: every chromosome except the highlighted one, drawn in gray
        for chr_val in unique_chrs:
            if chr_val == highlight_chr:
                continue
            src = ColumnDataSource(df_region[df_region["chromosome_true"] == chr_val])
            p2.scatter(fx, fy, source=src,
                       size=scatter_size_px,
                       fill_alpha=0.8, color=bg)

        # Also gray out points on the highlighted chromosome that lack an arm assignment
        missing_arm = df_region[
            (df_region["chromosome_true"] == highlight_chr)
            & df_region["chrom_arm"].isna()
        ]
        if len(missing_arm) > 0:
            p2.scatter(fx, fy, source=ColumnDataSource(missing_arm),
                       size=scatter_size_px,
                       fill_alpha=0.8, color=bg)

        # Highlighted arms drawn on top
        for arm, arm_color in arm_colors:
            sub = df_region[
                (df_region["chromosome_true"] == highlight_chr)
                & (df_region["chrom_arm"] == arm)
            ]
            if len(sub) == 0:
                continue
            p2.scatter(fx, fy, source=ColumnDataSource(sub),
                       size=scatter_size_px,
                       fill_alpha=0.8, color=arm_color)

        name = f"region_arms_{highlight_chr}"
        _export_scatter(p2, name)
        _make_overlay(name)


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


def export_chr_legends(out_dir):
    """One small legend SVG per highlighted whole chromosome."""
    for highlight_chr, color in CHR_HIGHLIGHTS:
        p = _make_legend_figure(200, 80)
        p.scatter([-999], [-999], size=8, fill_alpha=0.8,
                  color=color, legend_label=str(highlight_chr))
        p.legend.title = "Chromosome"
        p.legend.title_text_color = "black"
        p.legend.label_text_color = "black"
        p.legend.location = "top_left"
        p.legend.border_line_color = None
        p.legend.background_fill_alpha = 0.0
        _export_svg(p, f"{out_dir}/chr_{highlight_chr}_legend.svg")


def export_arm_legends(out_dir):
    """One small legend SVG per highlighted chromosome, showing p and q arms."""
    for highlight_chr, arm_colors in ARM_HIGHLIGHTS:
        p = _make_legend_figure(200, 110)
        for arm, color in arm_colors:
            p.scatter([-999], [-999], size=8, fill_alpha=0.6,
                      color=color, legend_label=f"{highlight_chr}{arm}")
        p.legend.title = "Arm"
        p.legend.title_text_color = "black"
        p.legend.label_text_color = "black"
        p.legend.location = "top_left"
        p.legend.border_line_color = None
        p.legend.background_fill_alpha = 0.0
        _export_svg(p, f"{out_dir}/arms_{highlight_chr}_legend.svg")


def export_celltype_legend(out_dir):
    """Legend with just two entries: Tumor (red) and Non-tumor (gray)."""
    p = _make_legend_figure(220, 110)
    for label, color in [("Tumor", TUMOR_COLOR), ("Non-tumor", NONTUMOR_COLOR)]:
        p.line([-999, -998], [-999, -999], line_width=2, line_dash="dashed",
               line_color=color, legend_label=label)
    p.legend.title = "Cell type"
    p.legend.title_text_color = "black"
    p.legend.label_text_color = "black"
    p.legend.label_text_font_size = "9pt"
    p.legend.title_text_font_size = "10pt"
    p.legend.location = "top_left"
    p.legend.border_line_color = None
    p.legend.background_fill_alpha = 0.0
    p.legend.spacing = 4
    p.legend.padding = 6
    p.legend.margin = 4
    p.legend.glyph_height = 12
    p.legend.glyph_width = 18
    _export_svg(p, f"{out_dir}/celltype_legend.svg")


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


def build_celltype_color_map(df, x_range, y_range):
    """Map each cell type present in the region to a color. Tumor cell types
    (in TUMOR_CELLTYPES) are colored TUMOR_COLOR; all others are NONTUMOR_COLOR."""
    df_region = df[
        (df["x"] >= x_range[0]) & (df["x"] <= x_range[1]) &
        (df["y"] >= y_range[0]) & (df["y"] <= y_range[1])
    ]
    celltypes = sorted(df_region["predicted_celltype"].dropna().unique())
    color_map = {}
    for ct in celltypes:
        if ct in TUMOR_CELLTYPES:
            color_map[ct] = TUMOR_COLOR
        else:
            color_map[ct] = NONTUMOR_COLOR
    return color_map


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    chr_dict = build_chr_dict()
    chrom_arm_dict = load_chrom_arm_dict()

    df = load_patient(PATIENT_ID, chr_dict, chrom_arm_dict)

    # Plot each large region, prefixing filenames with its label.
    # Highlight panels (chr1 arms, chr19 arms, chr7) are skipped here; they
    # are produced only for the per-cell zoom-ins below.
    if PLOT_LARGE_REGIONS:
        for label, region_x, region_y in tqdm(REGIONS, desc="large regions", unit="region"):
            celltype_to_color = build_celltype_color_map(df, region_x, region_y)
            plot_region(
                df,
                x_range=region_x, y_range=region_y,
                out_dir=OUT_DIR, patient_id=PATIENT_ID,
                celltype_to_color=celltype_to_color,
                plot_hull=PLOT_HULL, plot_other_chr=PLOT_OTHER_CHR,
                plot_highlights=False,
                name_prefix=label,
            )

    # Per-cell zoom-ins. For each region, pick the cell whose nucleus
    # contains the most spots in the region, then plot only that cell's
    # spots (filtered by matched_label) with that cell's hull drawn.
    geojson = get_segmentation_geojson(PATIENT_ID)
    for label, cell_x, cell_y in tqdm(CELL_REGIONS, desc="cell zooms", unit="cell"):
        df_region = df[
            (df["x"] >= cell_x[0]) & (df["x"] <= cell_x[1]) &
            (df["y"] >= cell_y[0]) & (df["y"] <= cell_y[1])
        ]
        df_region = df_region[df_region["chromosome_true"].notna()]

        target_label, hull_xs, hull_ys = select_target_cell(df_region, geojson)
        print(f"[{label}] target matched_label={target_label}")

        cell_celltype_to_color = build_celltype_color_map(df, cell_x, cell_y)
        plot_region(
            df,
            x_range=cell_x, y_range=cell_y,
            out_dir=OUT_DIR, patient_id=PATIENT_ID,
            celltype_to_color=cell_celltype_to_color,
            plot_hull=PLOT_HULL, plot_other_chr=PLOT_OTHER_CHR,
            point_size_pt=POINT_SIZE_PT * CELL_POINT_SIZE_MULT,
            target_cell_label=target_label,
            target_cell_hull=(hull_xs, hull_ys),
            name_prefix=label,
        )

    # Build the chromosome legend from the union of chromosomes present
    # across all large regions, so palette indices line up with the scatters.
    all_chrs = set()
    for _, region_x, region_y in REGIONS:
        df_region = df[
            (df["x"] >= region_x[0]) & (df["x"] <= region_x[1]) &
            (df["y"] >= region_y[0]) & (df["y"] <= region_y[1])
        ]
        all_chrs.update(df_region["chromosome_true"].dropna().unique())
    unique_chrs = sort_chromosomes(all_chrs)
    palette = PALETTE_HEX[: len(unique_chrs)]
    export_chromosome_legend(unique_chrs, palette, OUT_DIR)
    export_chr_legends(OUT_DIR)
    export_arm_legends(OUT_DIR)
    export_celltype_legend(OUT_DIR)


if __name__ == "__main__":
    main()