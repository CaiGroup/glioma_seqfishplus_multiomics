"""
spatial_score_analysis.py
--------------------------
Spatial analysis of malignancy scores in single-cell spatial data.

Main steps
----------
1. build_spatial_df(adata, score_col)
       Convert adata.obs to a standard spatial DataFrame.

2. plot_spatial_scores(adata, score_col)
       One scatter panel per sample (ID), cells coloured by score.
       Cells without a score (NaN) are shown in grey.

3. compute_neighbor_score_correlation(adata, score_col, radius, cell_types)
       For each cell that has a score:
         - count how many cells of each type are within `radius`
         - normalise by the total number of cells of that type in the same sample
         - Spearman-correlate the normalised neighbour count to the cell score
       Results are reported per cancer group and per cell type.

4. assign_hex_region_labels(adata, score_col, ...)   ← NEW
       Tile each sample with a hexagonal grid of a set radius.
       Compute the mean score of scored cells in each hex.
       Apply global quantile thresholds to label hexes as 'high', 'low', or 'mid'.
       Label EVERY cell in each hex with that label and write back to adata.obs.
       Empty hexes (no scored cells) are labelled NaN.

5. compute_score_region_enrichment(adata, ...)
       Neighbour enrichment test (high vs low regions) using composition-corrected
       counts and BH-corrected one-sample t-tests per cancer group.

6. plot_score_region_enrichment(results, ...)
       Bar + dot plot of enrichment differences.

7. plot_score_regions_spatial(adata, ...)
       Spatial scatter showing raw score (left) and region labels (right).

Column conventions (adata.obs)
------------------------------
    ID                  sample identifier (spatial)
    sample_group        cancer subtype
    center_x / center_y spatial coordinates
    predicted.celltype  cell type label
    <score_col>         malignancy score (NaN for cells without a score)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from scipy.spatial import cKDTree
from typing import Optional
from collections import defaultdict


# ── Column name mapping ───────────────────────────────────────────────────────
OBS_TO_SPATIAL = {
    "ID":                 "sample_id",
    "center_x":           "x",
    "center_y":           "y",
    "predicted.celltype": "cell_type",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. Build spatial DataFrame from adata
# ══════════════════════════════════════════════════════════════════════════════

def build_spatial_df(
    adata,
    score_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convert adata.obs to a standard spatial DataFrame.

    Parameters
    ----------
    adata      : AnnData with obs columns ID, center_x, center_y,
                 predicted.celltype, and optionally sample_group + score_col.
    score_col  : if provided, include this score column in the output.

    Returns
    -------
    pd.DataFrame with columns:
        sample_id, cell_id, x, y, cell_type
        + sample_group and score_col if present
    """
    obs = adata.obs.copy()

    required = list(OBS_TO_SPATIAL.keys())
    missing  = [c for c in required if c not in obs.columns]
    if missing:
        raise ValueError(
            f"adata.obs is missing required columns: {missing}\n"
            f"Expected: {required}"
        )

    df = obs.rename(columns=OBS_TO_SPATIAL).copy()
    df.insert(0, "cell_id", adata.obs_names)

    if "sample_group" in obs.columns:
        df["sample_group"] = obs["sample_group"].values

    if score_col is not None:
        if score_col not in obs.columns:
            raise ValueError(
                f"score_col '{score_col}' not found in adata.obs. "
                f"Available columns: {list(obs.columns)}"
            )
        df[score_col] = obs[score_col].values

    print(f"build_spatial_df:")
    print(f"  Cells        : {len(df):,}")
    print(f"  Samples (ID) : {df['sample_id'].nunique()}  "
          f"→ {sorted(df['sample_id'].unique())}")
    print(f"  Cell types   : {sorted(df['cell_type'].unique())}")
    if score_col:
        n_scored = df[score_col].notna().sum()
        print(f"  Scored cells : {n_scored:,}  "
              f"({100*n_scored/len(df):.1f}% of total)")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. Plot spatial scores per sample
# ══════════════════════════════════════════════════════════════════════════════

def plot_spatial_scores(
    adata,
    score_col: str,
    sample_group_col: str = "sample_group",
    figsize_per_panel: tuple = (5, 5),
    point_size: float = 4.0,
    cmap: str = "RdYlBu_r",
    nan_color: str = "#d3d3d3",
    nan_alpha: float = 0.5,
    nan_size_factor: float = 0.5,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    dpi: float = 600,
    out_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the spatial distribution of the malignancy score for each sample.

    One panel per sample (ID). Cells with a score are coloured by score value.
    Cells without a score are shown in grey at reduced opacity.

    Parameters
    ----------
    adata            : AnnData with obs columns ID, center_x, center_y,
                       predicted.celltype, sample_group, and score_col.
    score_col        : column in adata.obs containing the score.
    sample_group_col : column identifying the cancer subgroup (for panel title).
    figsize_per_panel: (width, height) per panel.
    point_size       : dot size for scored cells.
    cmap             : colormap for scored cells.
    nan_color        : colour for unscored (NaN) cells.
    nan_alpha        : opacity for unscored cells.
    nan_size_factor  : size multiplier for unscored cells.
    vmin / vmax      : colour scale limits (default: 2nd–98th percentile).
    out_path         : if given, save the figure here.
    """
    obs = adata.obs.copy()

    for col in ["ID", "center_x", "center_y"]:
        if col not in obs.columns:
            raise ValueError(f"adata.obs missing '{col}'")
    if score_col not in obs.columns:
        raise ValueError(f"score_col '{score_col}' not in adata.obs")

    samples = sorted(obs["ID"].unique())
    n       = len(samples)
    ncols   = min(n, 4)
    nrows   = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols,
                 figsize_per_panel[1] * nrows),
        constrained_layout=True,
    )
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    all_scores = obs[score_col].dropna()
    if vmin is None:
        vmin = float(np.nanpercentile(all_scores, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(all_scores, 98))

    norm     = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)

    for ax_i, sample_id in enumerate(samples):
        ax          = axes_flat[ax_i]
        s_obs       = obs[obs["ID"] == sample_id]
        has_score   = s_obs[score_col].notna()
        scored_df   = s_obs[has_score]
        unscored_df = s_obs[~has_score]

        if len(unscored_df) > 0:
            ax.scatter(
                unscored_df["center_x"], unscored_df["center_y"],
                c=nan_color, s=point_size * nan_size_factor,
                alpha=nan_alpha, linewidths=0, rasterized=True,
            )

        if len(scored_df) > 0:
            sc = ax.scatter(
                scored_df["center_x"], scored_df["center_y"],
                c=scored_df[score_col].values,
                cmap=cmap_obj, norm=norm,
                s=point_size, linewidths=0,
                alpha=0.85, rasterized=True,
            )

        group = ""
        if sample_group_col in s_obs.columns:
            groups = s_obs[sample_group_col].dropna().unique()
            group  = f"  [{', '.join(groups)}]"
        ax.set_title(
            f"{sample_id}{group}\n"
            f"scored: {has_score.sum():,}  ·  unscored: {(~has_score).sum():,}",
            fontsize=9,
        )
        ax.set_xlabel("x", fontsize=8)
        ax.set_ylabel("y", fontsize=8)
        ax.set_aspect("equal")

    for ax_i in range(n, len(axes_flat)):
        axes_flat[ax_i].set_visible(False)

    sm  = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes_flat[:n], shrink=0.6, pad=0.02)
    cbar.set_label(score_col.replace("_", " "), fontsize=9)

    score_name = score_col.replace("score_", "").replace("_", " ")
    fig.suptitle(
        f"Spatial distribution of malignancy score: {score_name}\n"
        f"Grey = unscored cells  ·  Colour scale: {vmin:.3f} – {vmax:.3f}",
        fontsize=11, y=1.01,
    )

    if out_path:
        fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
        print(f"Saved: {out_path}")

    return fig

# ══════════════════════════════════════════════════════════════════════════════
# 2b. Four-panel cell-type + numeric-score overview for a single sample
# ══════════════════════════════════════════════════════════════════════════════
 
CELLTYPE_COLORS = {
    "AC":               "#E64B35",
    "AC.like":          "#F39B7F",
    "OPC":              "#4DBBD5",
    "OPC.like Prolif.": "#00A087",
    "Olig. normal":     "#3C5488",
    "Microglia":        "#7E6148",
    "Neuron GABA.":     "#B09C85",
    "Neuron Glut.":     "#8491B4",
    "Endothelial":      "#91D1C2",
    "Pericytes":        "#DC0000",
    "T":                "#F7DC6F",
    "VLMC":             "#AAAAAA",
}
 
 
def _add_scalebar(ax, x_range: float, scalebar_um: float = 1000.0) -> None:
    """
    Draw a simple horizontal scale bar of `scalebar_um` micrometres in the
    bottom-left corner of *ax*.  Coordinates are assumed to be in µm.
 
    Parameters
    ----------
    ax          : matplotlib Axes on which to draw the scale bar.
    x_range     : total width of the plotted region in µm (used to size the bar
                  relative to the panel).
    scalebar_um : length of the scale bar in µm (default 1 000 µm = 1 mm).
    """
    # Position: 5 % inset from the left/bottom of the current axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_start = xlim[0] + 0.05 * (xlim[1] - xlim[0])
    y_pos   = ylim[0] + 0.04 * (ylim[1] - ylim[0])
 
    ax.plot(
        [x_start, x_start + scalebar_um], [y_pos, y_pos],
        color="black", linewidth=2.5, solid_capstyle="butt",
        transform=ax.transData, clip_on=False,
    )
    ax.text(
        x_start + scalebar_um / 2, y_pos + 0.015 * (ylim[1] - ylim[0]),
        f"{int(scalebar_um):,} µm",
        ha="center", va="bottom", fontsize=7, color="black",
    )

def plot_celltype_overview(
    adata,
    sample_id: str,
    cell_type: str,
    score_cols: tuple,
    *,
    celltype_col: str = "predicted.celltype",
    sample_col: str = "ID",
    highlight_color: str = "#E64B35",
    background_color: str = "#d3d3d3",
    background_alpha: float = 0.35,
    background_size: float = 2.0,
    highlight_size: float = 14.0,
    all_point_size: float = 3.5,
    cmap: str = "RdYlBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    scalebar_um: float = 1000.0,
    figsize_per_panel: tuple = (4.5, 4.5),
    sample_group_col: str = "sample_group",
    dpi: float = 600,
    out_path: Optional[str] = None,
    celltype_colors: Optional[dict] = None,
) -> plt.Figure:
    """
    Four-panel spatial overview for a single sample.
 
    Panel layout (left → right)
    ---------------------------
    1. **All cell types** — every cell coloured by cell-type using
       `celltype_colors` (defaults to the module-level ``CELLTYPE_COLORS``).
    2. **Highlighted subset** — all cells shown in grey; cells of
       `cell_type` shown on top in `highlight_color` with larger dots.
    3. **Score A** — only cells of `cell_type`, coloured by ``score_cols[0]``.
    4. **Score B** — only cells of `cell_type`, coloured by ``score_cols[1]``.
 
    A 1 000 µm scale bar (or `scalebar_um` µm) is added to every panel.
 
    Parameters
    ----------
    adata              : AnnData whose .obs contains spatial coordinates and
                         the columns listed below.
    sample_id          : value of `sample_col` that selects the sample to plot.
    cell_type          : cell-type label to highlight / colour in panels 2–4.
    score_cols         : 2-tuple of adata.obs column names with numeric scores
                         used for panels 3 and 4.
    celltype_col       : adata.obs column holding cell-type labels
                         (default ``"predicted.celltype"``).
    sample_col         : adata.obs column holding sample IDs (default ``"ID"``).
    highlight_color    : fill colour for the highlighted cell type in panel 2.
    background_color   : colour for non-highlighted cells in panels 2–4.
    background_alpha   : opacity of background cells.
    background_size    : dot size of background cells.
    highlight_size     : dot size of the highlighted cell type.
    all_point_size     : dot size in the all-cell-types panel (panel 1).
    cmap               : colormap for panels 3 and 4.
    vmin / vmax        : shared colour-scale limits for panels 3 & 4.
                         Default: 2nd–98th percentile of `score_cols[0]` values
                         for the selected sample.
    scalebar_um        : length of the scale bar in µm (default 1 000).
    figsize_per_panel  : (width, height) of each individual panel.
    sample_group_col   : adata.obs column with cancer-group label (for title).
    dpi                : output resolution when saving.
    out_path           : if provided, save the figure to this path.
    celltype_colors    : mapping of cell-type → hex colour.  Defaults to the
                         module-level ``CELLTYPE_COLORS`` dict.
 
    Returns
    -------
    matplotlib.figure.Figure
    """
    if len(score_cols) != 2:
        raise ValueError("`score_cols` must be a 2-tuple of column names.")
 
    ct_colors = celltype_colors if celltype_colors is not None else CELLTYPE_COLORS
 
    obs = adata.obs.copy()
 
    # ── Validate required columns ─────────────────────────────────────────────
    required = ["center_x", "center_y", celltype_col, sample_col]
    missing  = [c for c in required if c not in obs.columns]
    if missing:
        raise ValueError(f"adata.obs is missing required columns: {missing}")
    for sc in score_cols:
        if sc not in obs.columns:
            raise ValueError(f"score_col '{sc}' not found in adata.obs.")
 
    # ── Filter to the requested sample ───────────────────────────────────────
    s = obs[obs[sample_col] == sample_id].copy()
    if s.empty:
        raise ValueError(
            f"No cells found for sample_id='{sample_id}' "
            f"in obs['{sample_col}']."
        )
 
    # ── Separate target cell type from the rest ───────────────────────────────
    is_target   = s[celltype_col].isin(cell_type)
    target_df   = s[is_target]
    other_df    = s[~is_target]
 
    if target_df.empty:
        raise ValueError(
            f"No cells of type '{cell_type}' found in sample '{sample_id}'."
        )
 
    # ── Shared colour scale for panels 3 & 4 ─────────────────────────────────
    combined_scores = pd.concat(
        [target_df[sc].dropna() for sc in score_cols], ignore_index=True
    )
    if vmin is None:
        vmin = float(np.nanpercentile(combined_scores, 2))  if len(combined_scores) else 0.0
    if vmax is None:
        vmax = float(np.nanpercentile(combined_scores, 98)) if len(combined_scores) else 1.0
    norm     = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)
 
    # ── Axis extent & scale bar ───────────────────────────────────────────────
    x_all    = s["center_x"].values.astype(float)
    y_all    = s["center_y"].values.astype(float)
    x_range  = float(x_all.max() - x_all.min())
    pad      = x_range * 0.03
 
    def _set_lims(ax):
        ax.set_xlim(x_all.min() - pad, x_all.max() + pad)
        ax.set_ylim(y_all.min() - pad, y_all.max() + pad)
        ax.set_aspect("equal")
        ax.axis("off")
 
    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        1, 4,
        figsize=(figsize_per_panel[0] * 4, figsize_per_panel[1]),
        constrained_layout=True,
    )
    ax_all, ax_hi, ax_s1, ax_s2 = axes
 
    group_label = ""
    if sample_group_col in s.columns:
        grp = s[sample_group_col].dropna().unique()
        group_label = f"  [{', '.join(grp)}]"
 
    # ── Panel 1: all cell types ───────────────────────────────────────────────
    # Plot in a fixed z-order so smaller populations are visible on top
    all_types = s[celltype_col].unique()
    for ct in all_types:
        ct_df  = s[s[celltype_col] == ct]
        color  = ct_colors.get(ct, "#999999")
        ax_all.scatter(
            ct_df["center_x"], ct_df["center_y"],
            c=color, s=all_point_size,
            linewidths=0, alpha=0.85, rasterized=True,
            label=ct,
        )
 
    _set_lims(ax_all)
    _add_scalebar(ax_all, x_range, scalebar_um)
 
    # Legend: small markers, outside or inside depending on space
    handles, labels = ax_all.get_legend_handles_labels()
    ax_all.legend(
        handles, labels,
        fontsize=6, markerscale=1.8, frameon=False,
        loc="upper right", ncol=1,
        handletextpad=0.4, borderpad=0.3,
    )
    n_shown = len(all_types)
    ax_all.set_title(
        f"{sample_id}{group_label}\nAll cell types  (n={len(s):,})",
        fontsize=9,
    )
 
    # ── Panel 2: highlight cell type ─────────────────────────────────────────
    if len(other_df):
        ax_hi.scatter(
            other_df["center_x"], other_df["center_y"],
            c=background_color, s=background_size,
            alpha=background_alpha, linewidths=0, rasterized=True,
        )
    ax_hi.scatter(
        target_df["center_x"], target_df["center_y"],
        c=highlight_color, s=highlight_size,
        alpha=0.9, linewidths=0, rasterized=True,
        label=cell_type,
    )
    _set_lims(ax_hi)
    _add_scalebar(ax_hi, x_range, scalebar_um)
    ax_hi.set_title(
        f"{cell_type}  (n={len(target_df):,})\n"
        f"highlighted on all {len(s):,} cells",
        fontsize=9,
    )
 
    # ── Panels 3 & 4: numeric scores ─────────────────────────────────────────
    for ax_score, sc in zip((ax_s1, ax_s2), score_cols):
        has_val   = target_df[sc].notna()
        scored    = target_df[has_val]
        unscored  = target_df[~has_val]
 
        # Background: all other cells
        if len(other_df):
            ax_score.scatter(
                other_df["center_x"], other_df["center_y"],
                c='white', s=0,
                alpha=background_alpha, linewidths=0, rasterized=True,
            )
        # Unscored target cells (NaN) in light grey
        if len(unscored):
            ax_score.scatter(
                unscored["center_x"], unscored["center_y"],
                c="#bbbbbb", s=highlight_size * 0.6,
                alpha=0.5, linewidths=0, rasterized=True,
            )
        # Scored target cells coloured by value
        if len(scored):
            sc_plot = ax_score.scatter(
                scored["center_x"], scored["center_y"],
                c=scored[sc].values, cmap=cmap_obj, norm=norm,
                s=highlight_size, linewidths=0,
                alpha=0.9, rasterized=True,
            )
 
        _set_lims(ax_score)
        _add_scalebar(ax_score, x_range, scalebar_um)
 
        # Colour bar attached to this panel
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax_score, shrink=0.6, pad=0.02)
        cbar.set_label(sc.replace("_", " "), fontsize=8)
        cbar.ax.tick_params(labelsize=7)
 
        n_scored = has_val.sum()
        ax_score.set_title(
            f"{cell_type}  ·  {sc.replace('_', ' ')}\n"
            f"scored: {n_scored:,}  ·  unscored: {(~has_val).sum():,}",
            fontsize=9,
        )
 
    if out_path:
        fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
        print(f"Saved: {out_path}")
 
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# 3. Neighbour-count → score correlation
# ══════════════════════════════════════════════════════════════════════════════

def compute_neighbor_score_correlation(
    adata,
    score_col: str,
    radius: float,
    cancer_groups: Optional[list] = None,
    cell_types_to_count: Optional[list] = None,
    min_cells_per_sample: int = 10,
    alpha: float = 0.05,
) -> dict:
    """
    For each scored cell, count how many cells of each type fall within
    `radius`, normalise by the total number of cells of that type in the
    same sample, then Spearman-correlate the normalised count to the cell score.

    Parameters
    ----------
    adata                 : AnnData with obs columns ID, center_x, center_y,
                            predicted.celltype, sample_group, and score_col.
    score_col             : score column in adata.obs (NaN for unscored cells).
    radius                : neighbourhood radius in coordinate units.
    cancer_groups         : list of sample_group values to analyse. If None,
                            uses all non-NaN groups.
    cell_types_to_count   : which neighbour cell types to count. If None,
                            counts all cell types present.
    min_cells_per_sample  : minimum scored cells per sample to include it.
    alpha                 : significance threshold (BH correction applied per group).

    Returns
    -------
    dict[group_label → pd.DataFrame]
    """
    obs = adata.obs.copy()
    obs["_cell_id"] = adata.obs_names

    for col in ["ID", "center_x", "center_y", "predicted.celltype", "sample_group"]:
        if col not in obs.columns:
            raise ValueError(f"adata.obs missing required column '{col}'")
    if score_col not in obs.columns:
        raise ValueError(f"score_col '{score_col}' not in adata.obs")

    all_cell_types = sorted(obs["predicted.celltype"].unique())
    ct_to_count    = cell_types_to_count or all_cell_types

    if cancer_groups is None:
        cancer_groups = sorted(obs["sample_group"].dropna().unique())

    print(f"\n── compute_neighbor_score_correlation ───────────────────────────")
    print(f"  Score col        : {score_col}")
    print(f"  Radius           : {radius}")
    print(f"  Cancer groups    : {cancer_groups}")

    sample_trees   = {}
    sample_types   = {}
    sample_ids_arr = {}

    for sample_id, s_df in obs.groupby("ID"):
        coords = s_df[["center_x", "center_y"]].values.astype(float)
        sample_trees[sample_id]   = cKDTree(coords)
        sample_types[sample_id]   = s_df["predicted.celltype"].values
        sample_ids_arr[sample_id] = s_df["_cell_id"].values

    norm_count_rows = []

    for sample_id in sorted(obs["ID"].unique()):
        s_obs    = obs[obs["ID"] == sample_id]
        scored_s = s_obs[s_obs[score_col].notna()]
        if scored_s.empty:
            continue

        tree     = sample_trees[sample_id]
        all_ct   = sample_types[sample_id]
        all_cids = sample_ids_arr[sample_id]

        type_totals     = {ct: int((all_ct == ct).sum()) for ct in ct_to_count}
        scored_coords   = scored_s[["center_x", "center_y"]].values.astype(float)
        neighbors_list  = tree.query_ball_point(scored_coords, r=radius)

        for row_i, (idx, row) in enumerate(scored_s.iterrows()):
            nb_indices = neighbors_list[row_i]
            self_cid   = row["_cell_id"]
            self_pos   = np.where(all_cids == self_cid)[0]
            nb_indices = [n for n in nb_indices
                          if len(self_pos) == 0 or n != self_pos[0]]

            nb_types = all_ct[nb_indices] if nb_indices else np.array([])

            row_data = {
                "_cell_id":     self_cid,
                "sample_id":    sample_id,
                score_col:      row[score_col],
                "sample_group": row.get("sample_group", np.nan),
            }
            for ct in ct_to_count:
                raw      = int((nb_types == ct).sum())
                total_ct = type_totals.get(ct, 0)
                row_data[f"nb_{ct}"] = raw / total_ct if total_ct > 0 else np.nan

            norm_count_rows.append(row_data)

    norm_df = pd.DataFrame(norm_count_rows).set_index("_cell_id")
    print(f"  Scored cells with neighbour counts: {len(norm_df):,}")

    def _correlate_group(label: str, group_df: pd.DataFrame):
        samples = group_df["sample_id"].unique()
        print(f"\n  ── {label} ──")
        print(f"    Cells: {len(group_df):,}  |  Samples: {len(samples)}")

        sample_corr_dict = {}
        for sid in samples:
            s_df     = group_df[group_df["sample_id"] == sid]
            s_scores = s_df[score_col].dropna()
            valid_ids = s_scores.index

            if len(valid_ids) < min_cells_per_sample:
                continue

            ct_corrs = {}
            for ct in ct_to_count:
                col = f"nb_{ct}"
                if col not in s_df.columns:
                    continue
                nb_vals  = s_df.loc[valid_ids, col].dropna()
                both_ids = s_scores.index.intersection(nb_vals.index)
                if len(both_ids) < min_cells_per_sample:
                    ct_corrs[ct] = np.nan
                    continue
                r, _ = stats.spearmanr(
                    s_scores.loc[both_ids].values,
                    nb_vals.loc[both_ids].values,
                )
                ct_corrs[ct] = float(r)

            sample_corr_dict[sid] = pd.Series(ct_corrs, name=sid)

        if len(sample_corr_dict) < 2:
            print(f"    < 2 usable samples — skipping")
            return None

        corr_matrix   = pd.DataFrame(sample_corr_dict).T
        mean_corr     = corr_matrix.mean(axis=0, skipna=True)
        std_corr      = corr_matrix.std(axis=0,  skipna=True, ddof=1)
        n_valid       = corr_matrix.notna().sum(axis=0)
        n_pos_samples = (corr_matrix > 0).sum(axis=0)
        n_neg_samples = (corr_matrix < 0).sum(axis=0)
        all_same_dir  = (n_pos_samples == n_valid) | (n_neg_samples == n_valid)

        t_stats = pd.Series(np.nan, index=ct_to_count)
        p_vals  = pd.Series(np.nan, index=ct_to_count)
        for ct in ct_to_count:
            s = corr_matrix[ct].dropna().values if ct in corr_matrix.columns else np.array([])
            if len(s) < 2:
                continue
            t, p = stats.ttest_1samp(s, popmean=0.0)
            t_stats[ct] = t
            p_vals[ct]  = p

        p_adj  = pd.Series(np.nan, index=ct_to_count)
        tested = p_vals.notna()
        if tested.sum() > 0:
            try:
                p_adj[tested] = stats.false_discovery_control(
                    p_vals[tested].values, method="bh", axis=0)
            except AttributeError:
                p_adj[tested] = _bh_correction(p_vals[tested].values)

        significant = p_adj < alpha
        direction   = mean_corr.apply(lambda x: "enriched" if x > 0 else "depleted")
        direction[~significant] = np.nan

        result = pd.DataFrame({
            "neighbour_type":     ct_to_count,
            "mean_corr":          mean_corr.round(4).values,
            "std_corr":           std_corr.round(4).values,
            "n_samples":          n_valid.values,
            "n_samples_positive": n_pos_samples.values,
            "n_samples_negative": n_neg_samples.values,
            "all_same_direction": all_same_dir.values,
            "t_stat":             t_stats.round(4).values,
            "p_value":            p_vals.round(6).values,
            "p_adj":              p_adj.round(6).values,
            "significant":        significant.values,
            "direction":          direction.values,
        }).sort_values("p_adj")

        return result

    results = {}
    for group in cancer_groups:
        grp_df = norm_df[norm_df["sample_group"] == group]
        res    = _correlate_group(group, grp_df)
        if res is not None:
            results[group] = res

    normal_df = norm_df[~norm_df["sample_group"].isin(cancer_groups)]
    if not normal_df.empty:
        res_n = _correlate_group("normal", normal_df)
        if res_n is not None:
            results["normal"] = res_n

    print(f"\n── Done. Groups: {list(results.keys())}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 4. Hex-based region labelling   ← NEW
# ══════════════════════════════════════════════════════════════════════════════

def _assign_hex_ids(
    x: np.ndarray,
    y: np.ndarray,
    hex_radius: float,
) -> np.ndarray:
    """
    Assign a hex-grid bin ID string to each point using axial (cube) coordinates.

    hex_radius is the distance from the hex centre to any corner (= the 'size'
    parameter in standard hex-grid literature).  The flat-top orientation is
    used so that hexagons tile without gaps and the coordinate system is
    straightforward.

    Returns
    -------
    hex_ids : np.ndarray of str, shape (n_points,)
    """
    sqrt3 = np.sqrt(3)

    # Flat-top hex: axial coords
    q = (2 / 3 * x) / hex_radius
    r = (-1 / 3 * x + sqrt3 / 3 * y) / hex_radius

    # Cube coordinates for proper rounding
    x_c =  q
    z_c =  r
    y_c = -x_c - z_c

    rx = np.round(x_c)
    ry = np.round(y_c)
    rz = np.round(z_c)

    # Fix the coordinate with the largest rounding error
    x_diff = np.abs(rx - x_c)
    y_diff = np.abs(ry - y_c)
    z_diff = np.abs(rz - z_c)

    mask_x = (x_diff > y_diff) & (x_diff > z_diff)
    mask_z = ~mask_x & (z_diff > y_diff)

    rx[mask_x] = -ry[mask_x] - rz[mask_x]
    rz[mask_z] = -rx[mask_z] - ry[mask_z]

    hex_q = rx.astype(int)
    hex_r = rz.astype(int)

    return np.array([f"{q}_{r}" for q, r in zip(hex_q, hex_r)])


def assign_hex_region_labels(
    adata,
    score_col: str,
    hex_radius: float,
    high_quantile: float = 0.75,
    low_quantile: float  = 0.25,
    min_scored_per_hex: int = 3,
    region_col: str = "hex_score_region",
    hex_mean_col: Optional[str] = None,
    cancer_groups: Optional[list] = None,
    sample_group_col: str = "sample_group",
) -> "AnnData":
    """
    Label every cell with the score region of its hexagonal bin.

    The tissue is divided into a hexagonal grid of radius `hex_radius`.
    For each hex, the mean score of all scored cells inside it is computed.
    Global quantile thresholds (pooled across all relevant samples) are used
    to label each hex as 'high', 'mid', or 'low'.  Every cell in a labelled
    hex — scored or not — inherits that label.  Cells in hexes with fewer
    than `min_scored_per_hex` scored cells receive NaN.

    This approach avoids mixed labelling: all cells in the same spatial unit
    share a single label, making downstream neighbour comparisons clean.

    Parameters
    ----------
    adata               : AnnData.  obs must contain:
                            ID, center_x, center_y, score_col.
                          If `cancer_groups` is given, obs must also contain
                          `sample_group_col`.
    score_col           : column in adata.obs with continuous scores
                          (NaN for cells that were not scored).
    hex_radius          : radius of each hexagon in the same coordinate units
                          as center_x / center_y (distance from centre to corner).
    high_quantile       : quantile above which a hex is labelled 'high'
                          (default 0.75).
    low_quantile        : quantile below which a hex is labelled 'low'
                          (default 0.25).
    min_scored_per_hex  : minimum number of scored cells in a hex to assign
                          a region label (default 3).  Hexes below this threshold
                          receive NaN.
    region_col          : name of the new column written to adata.obs
                          (default 'hex_score_region').
    hex_mean_col        : if provided, also write the per-hex mean score into
                          adata.obs under this column name.
    cancer_groups       : if given, only cells from samples whose
                          `sample_group_col` value is in this list contribute to
                          the global threshold calculation.  All samples are still
                          labelled.
    sample_group_col    : column in adata.obs that holds the group label.

    Returns
    -------
    adata : the input AnnData with adata.obs[region_col] added (and
            adata.obs[hex_mean_col] if requested).

    Notes
    -----
    - Global thresholds are computed from hexes that have ≥ min_scored_per_hex
      scored cells and (if cancer_groups is set) belong to those groups.
    - The function works per sample: each sample gets its own hex grid tiled
      independently, so coordinate systems do not need to be aligned.
    - Hexes that span sample boundaries cannot occur because the grid is
      built per sample.
    """
    obs = adata.obs.copy()

    for col in ["ID", "center_x", "center_y", score_col]:
        if col not in obs.columns:
            raise ValueError(f"adata.obs missing required column: '{col}'")

    if cancer_groups is not None and sample_group_col not in obs.columns:
        raise ValueError(
            f"sample_group_col '{sample_group_col}' not in adata.obs "
            f"but cancer_groups was specified."
        )

    samples = sorted(obs["ID"].unique())
    print(f"\n── assign_hex_region_labels ─────────────────────────────────────")
    print(f"  Score column       : {score_col}")
    print(f"  Hex radius         : {hex_radius}")
    print(f"  High / low quantile: {high_quantile} / {low_quantile}")
    print(f"  Min scored per hex : {min_scored_per_hex}")
    print(f"  Samples            : {samples}")
    if cancer_groups:
        print(f"  Threshold groups   : {cancer_groups}")

    # ── Pass 1: assign hex IDs and compute per-hex mean score per sample ──────
    # Store for later label assignment.
    # hex_mean_by_cell[obs_index] = mean score of the cell's hex
    hex_mean_by_cell  = {}   # obs_index  → hex mean score (float or NaN)
    hex_id_by_cell    = {}   # obs_index  → hex_id string

    # For global threshold: collect hex means from qualifying samples/hexes
    global_hex_means  = []

    for sample_id in samples:
        s = obs[obs["ID"] == sample_id].copy()

        x = s["center_x"].values.astype(float)
        y = s["center_y"].values.astype(float)
        hex_ids = _assign_hex_ids(x, y, hex_radius)

        # Store hex ID per cell
        for idx, hid in zip(s.index, hex_ids):
            hex_id_by_cell[idx] = hid

        # Scored cells in this sample
        scored_mask = s[score_col].notna()
        scored_s    = s[scored_mask]
        hex_ids_scored = hex_ids[scored_mask.values]  # align with scored rows

        # Per-hex mean score
        hex_score_map = {}  # hex_id → mean score (only for hexes with enough cells)
        hex_groups    = defaultdict(list)
        for hid, score_val in zip(hex_ids_scored, scored_s[score_col].values):
            hex_groups[hid].append(score_val)

        for hid, scores in hex_groups.items():
            if len(scores) >= min_scored_per_hex:
                hex_score_map[hid] = float(np.mean(scores))

        # Assign hex mean to every cell in the sample
        for idx, hid in zip(s.index, hex_ids):
            hex_mean_by_cell[idx] = hex_score_map.get(hid, np.nan)

        # Collect hex means for global threshold (only from qualifying groups)
        in_group = True
        if cancer_groups is not None and sample_group_col in s.columns:
            grp = s[sample_group_col].iloc[0] if len(s) > 0 else None
            in_group = grp in cancer_groups

        if in_group:
            global_hex_means.extend(hex_score_map.values())

        n_hexes_total  = len(set(hex_ids))
        n_hexes_scored = len(hex_score_map)
        n_scored_cells = scored_mask.sum()
        print(f"  {sample_id}: {len(s):,} cells  |  "
              f"{n_scored_cells:,} scored  |  "
              f"{n_hexes_total} hexes  |  "
              f"{n_hexes_scored} hexes with ≥{min_scored_per_hex} scored cells")

    if not global_hex_means:
        raise ValueError(
            "No hexes had enough scored cells to compute global thresholds. "
            "Try reducing min_scored_per_hex or hex_radius."
        )

    # ── Pass 2: global quantile thresholds ───────────────────────────────────
    all_means  = np.array(global_hex_means)
    hi_thresh  = float(np.quantile(all_means, high_quantile))
    lo_thresh  = float(np.quantile(all_means, low_quantile))

    print(f"\n  Global hex means: n={len(all_means):,}  "
          f"min={all_means.min():.4f}  max={all_means.max():.4f}")
    print(f"  Thresholds: low ≤ {lo_thresh:.4f}  |  mid  |  high ≥ {hi_thresh:.4f}")

    # ── Pass 3: label every cell ───────────────────────────────────────────────
    region_labels = pd.Series(pd.NA, index=obs.index, dtype=object)
    hex_means_out = pd.Series(np.nan, index=obs.index, dtype=float)

    for idx in obs.index:
        hm = hex_mean_by_cell.get(idx, np.nan)
        hex_means_out[idx] = hm
        if np.isnan(hm):
            region_labels[idx] = pd.NA
        elif hm >= hi_thresh:
            region_labels[idx] = "high"
        elif hm <= lo_thresh:
            region_labels[idx] = "low"
        else:
            region_labels[idx] = "mid"

    adata.obs[region_col] = region_labels.values
    if hex_mean_col is not None:
        adata.obs[hex_mean_col] = hex_means_out.values

    # ── Summary per sample ────────────────────────────────────────────────────
    print(f"\n  Per-sample label counts:")
    for sample_id in samples:
        s_idx = obs[obs["ID"] == sample_id].index
        s_labels = region_labels[s_idx]
        n_high = (s_labels == "high").sum()
        n_mid  = (s_labels == "mid").sum()
        n_low  = (s_labels == "low").sum()
        n_nan  = s_labels.isna().sum()
        print(f"    {sample_id}: high={n_high:,}  mid={n_mid:,}  "
              f"low={n_low:,}  unlabelled={n_nan:,}")

    total_labelled = (~region_labels.isna()).sum()
    print(f"\n  Total labelled cells : {total_labelled:,} / {len(obs):,} "
          f"({100*total_labelled/len(obs):.1f}%)")
    print(f"  Column written to adata.obs: '{region_col}'")
    if hex_mean_col:
        print(f"  Hex mean column      : '{hex_mean_col}'")

    return adata


# ══════════════════════════════════════════════════════════════════════════════
# 5. Score-region neighbour enrichment
# ══════════════════════════════════════════════════════════════════════════════

def compute_score_region_enrichment(
    adata,
    cancer_groups: list,
    score_col: str = "score_between_combined_contacts",
    region_col: str = "hex_score_region",
    neighbor_radius: float = 200.0,
    min_neighbors: int = 5,
    alpha: float = 0.05,
    sample_group_col: str = "sample_group",
) -> tuple:
    """
    Test whether specific neighbour cell types are enriched or depleted in
    high vs low score regions, corrected for per-sample cell type composition.

    This function is designed to work with region labels produced by
    assign_hex_region_labels().  The labels ('high', 'low', 'mid') apply to
    every cell in a hex — not just scored cells — so neighbour counts reflect
    the full microenvironment.

    Steps
    -----
    1. For each high/low labelled cell, count all cells within neighbor_radius
       by cell type (all cell types, scored or not).
    2. Correct for composition: enrichment(X) = observed(X) - expected(X)
       where expected(X) = n_neighbors × (cells of type X in sample / total cells).
    3. Aggregate per-sample mean enrichment differences (high − low) across
       samples with a one-sample t-test, then apply BH FDR correction.

    Parameters
    ----------
    adata           : AnnData.  obs must contain:
                        ID, center_x, center_y, predicted.celltype,
                        sample_group_col, region_col.
    cancer_groups   : list of sample_group values to analyse.
    score_col       : used only for validation / reporting — not for re-computing
                      regions; labels are taken from region_col.
    region_col      : column in adata.obs with 'high'/'mid'/'low'/NaN labels
                      (written by assign_hex_region_labels).
    neighbor_radius : radius for neighbour counting.
    min_neighbors   : minimum neighbours for a cell to be included in the test.
    alpha           : FDR significance threshold.
    sample_group_col: column holding the cancer group label.

    Returns
    -------
    (results, adata) where:
      results : dict[cancer_group → {"summary": pd.DataFrame,
                                      "per_sample": pd.DataFrame}]
      adata   : input AnnData (unchanged — labels were already added).
    """
    obs = adata.obs.copy()

    for col in ["ID", "center_x", "center_y", "predicted.celltype",
                sample_group_col, region_col]:
        if col not in obs.columns:
            raise ValueError(f"adata.obs missing required column: '{col}'")

    obs_grp = obs[obs[sample_group_col].isin(cancer_groups)].copy()
    if obs_grp.empty:
        raise ValueError(
            f"No samples found for cancer_groups={cancer_groups}. "
            f"Available: {sorted(obs[sample_group_col].unique())}"
        )

    all_cell_types = sorted(obs_grp["predicted.celltype"].unique())
    samples        = sorted(obs_grp["ID"].unique())

    print(f"\n── compute_score_region_enrichment ──────────────────────────────")
    print(f"  Cancer groups   : {cancer_groups}")
    print(f"  Region col      : {region_col}")
    print(f"  Neighbor radius : {neighbor_radius}")
    print(f"  Samples         : {samples}")

    per_sample_stats = defaultdict(lambda: defaultdict(list))

    for sample_id in samples:
        s     = obs_grp[obs_grp["ID"] == sample_id].copy()
        group = s[sample_group_col].iloc[0]

        type_counts = s["predicted.celltype"].value_counts()
        n_total     = len(s)
        type_frac   = {ct: type_counts.get(ct, 0) / n_total for ct in all_cell_types}

        all_coords = s[["center_x", "center_y"]].values.astype(float)
        all_types  = s["predicted.celltype"].values
        tree_all   = cKDTree(all_coords)

        # Cells with a region label (includes scored + unscored cells in labelled hexes)
        labelled_mask = s[region_col].isin(["high", "low"])
        labelled_df   = s[labelled_mask].copy()

        n_high = (labelled_df[region_col] == "high").sum()
        n_low  = (labelled_df[region_col] == "low").sum()

        if n_high < 5 or n_low < 5:
            print(f"  {sample_id}: too few high/low labelled cells "
                  f"(high={n_high}, low={n_low}) — skipping")
            continue

        print(f"  {sample_id} [{group}]: high={n_high:,}  low={n_low:,}")

        region_enrichments = {"high": defaultdict(list),
                              "low":  defaultdict(list)}

        for region in ("high", "low"):
            region_coords = labelled_df.loc[
                labelled_df[region_col] == region,
                ["center_x", "center_y"],
            ].values.astype(float)

            nbr_idxs_list = tree_all.query_ball_point(region_coords, r=neighbor_radius)

            for nbr_idxs in nbr_idxs_list:
                n_nbrs = len(nbr_idxs)
                if n_nbrs < min_neighbors:
                    continue
                nbr_types = all_types[nbr_idxs]
                for ct in all_cell_types:
                    observed_count = (nbr_types == ct).sum()
                    expected_count = n_nbrs * type_frac[ct]
                    region_enrichments[region][ct].append(observed_count - expected_count)

        for ct in all_cell_types:
            hi_vals = region_enrichments["high"][ct]
            lo_vals = region_enrichments["low"][ct]
            if len(hi_vals) < 3 or len(lo_vals) < 3:
                continue
            per_sample_stats[group][ct].append((
                sample_id,
                float(np.mean(hi_vals)),
                float(np.mean(lo_vals)),
            ))

    # ── Aggregate and test ────────────────────────────────────────────────────
    results = {}

    for group in cancer_groups:
        ct_stats = per_sample_stats.get(group, {})
        if not ct_stats:
            print(f"  No data for group {group}")
            continue

        summary_records    = []
        per_sample_records = []
        ct_diffs = {}
        ct_hi    = {}
        ct_lo    = {}

        for ct_name, sample_vals in ct_stats.items():
            for (sid, hi_enr, lo_enr) in sample_vals:
                per_sample_records.append({
                    "sample_id":             sid,
                    "neighbour_type":        ct_name,
                    "mean_enrichment_high":  round(hi_enr, 4),
                    "mean_enrichment_low":   round(lo_enr, 4),
                    "mean_diff":             round(hi_enr - lo_enr, 4),
                })
            hi_arr = np.array([v[1] for v in sample_vals])
            lo_arr = np.array([v[2] for v in sample_vals])
            ct_diffs[ct_name] = hi_arr - lo_arr
            ct_hi[ct_name]    = hi_arr
            ct_lo[ct_name]    = lo_arr

        p_vals      = {}
        t_stats_out = {}
        for ct_name, diffs in ct_diffs.items():
            if len(diffs) < 2:
                p_vals[ct_name]      = np.nan
                t_stats_out[ct_name] = np.nan
            else:
                t, p = stats.ttest_1samp(diffs, popmean=0.0)
                t_stats_out[ct_name] = float(t)
                p_vals[ct_name]      = float(p)

        ct_list = list(p_vals.keys())
        pv_arr  = np.array([p_vals[ct_name] for ct_name in ct_list])
        valid   = ~np.isnan(pv_arr)
        p_adj   = np.full(len(pv_arr), np.nan)
        if valid.sum() > 1:
            try:
                p_adj[valid] = stats.false_discovery_control(
                    pv_arr[valid], method="bh", axis=0)
            except AttributeError:
                p_adj[valid] = _bh_correction(pv_arr[valid])

        for i, ct_name in enumerate(ct_list):
            mean_diff = float(np.mean(ct_diffs[ct_name]))
            sig       = not np.isnan(p_adj[i]) and p_adj[i] < alpha
            summary_records.append({
                "neighbour_type":       ct_name,
                "mean_enrichment_high": round(float(np.mean(ct_hi[ct_name])), 4),
                "mean_enrichment_low":  round(float(np.mean(ct_lo[ct_name])), 4),
                "mean_diff":            round(mean_diff, 4),
                "n_samples":            len(ct_diffs[ct_name]),
                "t_stat":               round(t_stats_out[ct_name], 4)
                                        if not np.isnan(t_stats_out[ct_name]) else np.nan,
                "p_value":              round(p_vals[ct_name], 6)
                                        if not np.isnan(p_vals[ct_name]) else np.nan,
                "p_adj":                round(p_adj[i], 6)
                                        if not np.isnan(p_adj[i]) else np.nan,
                "significant":          sig,
                "direction":            ("enriched_in_high" if mean_diff > 0
                                         else "enriched_in_low") if sig else np.nan,
            })

        summary_df    = pd.DataFrame(summary_records).sort_values("p_adj")
        per_sample_df = pd.DataFrame(per_sample_records).sort_values(
            ["neighbour_type", "sample_id"])

        n_sig = summary_df["significant"].sum()
        print(f"\n  {group}: {n_sig} significant neighbour types  "
              f"(enriched in high: {(summary_df['direction']=='enriched_in_high').sum()}  "
              f"·  enriched in low: {(summary_df['direction']=='enriched_in_low').sum()})")

        results[group] = {"summary": summary_df, "per_sample": per_sample_df}

    return results, adata


# ══════════════════════════════════════════════════════════════════════════════
# BH correction fallback
# ══════════════════════════════════════════════════════════════════════════════

def _bh_correction(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction fallback for older scipy."""
    n     = len(p_values)
    order = np.argsort(p_values)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    p_adj = np.minimum(1.0, p_values * n / ranks)
    for i in range(n - 2, -1, -1):
        p_adj[order[i]] = min(p_adj[order[i]], p_adj[order[i + 1]])
    return p_adj


# ══════════════════════════════════════════════════════════════════════════════
# 6. Plot score-region neighbour enrichment
# ══════════════════════════════════════════════════════════════════════════════

def plot_score_region_enrichment(
    results: dict,
    cancer_groups: Optional[list] = None,
    n_top: int = 20,
    alpha: float = 0.05,
    out_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar plot of neighbour cell type enrichment in high vs low score regions,
    with individual sample values overlaid as dots.

    Parameters
    ----------
    results       : output of compute_score_region_enrichment() —
                    dict[group → {"summary": df, "per_sample": df}].
                    Also accepts plain dict[group → summary_df].
    cancer_groups : order of groups to plot (default: all in results).
    n_top         : max cell types to show per group.
    alpha         : FDR significance threshold.
    out_path      : optional save path.
    """
    GROUP_PALETTE = ["#E8604C", "#F5A623", "#7ED321", "#9B59B6", "#2ECC71"]

    if cancer_groups is None:
        cancer_groups = list(results.keys())
    groups_to_plot = [g for g in cancer_groups if g in results]
    if not groups_to_plot:
        print("No results to plot.")
        return None

    def _get_summary(group):
        v = results[group]
        return v["summary"] if isinstance(v, dict) else v

    def _get_per_sample(group):
        v = results[group]
        return v.get("per_sample") if isinstance(v, dict) else None

    group_colors = {g: GROUP_PALETTE[i % len(GROUP_PALETTE)]
                    for i, g in enumerate(groups_to_plot)}

    n_rows = len(groups_to_plot)
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(10, max(4, n_top * 0.38 + 2) * n_rows),
        constrained_layout=True,
        squeeze=False,
    )

    for row_i, group in enumerate(groups_to_plot):
        ax      = axes[row_i, 0]
        summary = _get_summary(group).copy()
        ps      = _get_per_sample(group)

        summary["abs_diff"] = summary["mean_diff"].abs()
        top       = summary.nlargest(n_top, "abs_diff").sort_values("mean_diff")
        top_types = top["neighbour_type"].tolist()
        y_pos     = np.arange(len(top))
        color     = group_colors[group]

        for bar_i, (_, row) in enumerate(top.iterrows()):
            is_sig    = row["significant"]
            bar_color = color if row["mean_diff"] >= 0 else "#4C9BE8"
            ax.barh(
                bar_i, row["mean_diff"],
                height=0.55,
                color=bar_color,
                alpha=0.80 if is_sig else 0.30,
                hatch=None if is_sig else "//",
                zorder=2,
            )
            if is_sig:
                p     = row["p_adj"]
                stars = "***" if p < 0.001 else "**" if p < 0.01 else "*"
                x_ann = row["mean_diff"] + 0.15 * np.sign(row["mean_diff"])
                ax.text(x_ann, bar_i, stars, va="center", fontsize=8,
                        ha="left" if row["mean_diff"] > 0 else "right",
                        color=bar_color, zorder=4)

        if ps is not None and not ps.empty:
            for bar_i, ct_name in enumerate(top_types):
                ct_ps = ps[ps["neighbour_type"] == ct_name]
                if ct_ps.empty:
                    continue
                diffs  = ct_ps["mean_diff"].values
                jitter = np.linspace(-0.22, 0.22, len(diffs)) if len(diffs) > 1 else [0]
                bar_color = color if np.mean(diffs) >= 0 else "#4C9BE8"
                ax.scatter(
                    diffs,
                    [bar_i + j for j in jitter],
                    color=bar_color, s=22, zorder=5,
                    linewidths=0.4, edgecolors="white", alpha=0.9,
                )

        ax.axvline(0, color="#888780", lw=0.8, ls="--", alpha=0.6, zorder=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_types, fontsize=9)
        ax.set_xlabel(
            "Mean enrichment difference  (high-score region − low-score region)\n"
            "Positive = enriched near high-score cells  ·  "
            "Negative = enriched near low-score cells",
            fontsize=8,
        )
        ax.set_title(group, fontsize=11)

        n_samp = int(top["n_samples"].max()) if not top.empty else 0
        ax.text(0.99, 0.99, f"n samples = {n_samp}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="#888780")

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor="#E8604C", alpha=0.80, label="Enriched in high-score regions (FDR sig)"),
        Patch(facecolor="#4C9BE8", alpha=0.80, label="Enriched in low-score regions (FDR sig)"),
        Patch(facecolor="#888888", alpha=0.30, hatch="//", label="Not FDR significant"),
        Line2D([0],[0], marker="o", color="#888888", markersize=5,
               linestyle="none", label="Individual samples"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(
        "Neighbour cell type enrichment: high-score vs low-score regions\n"
        "Composition-corrected  ·  bars = mean  ·  dots = individual samples  ·  * FDR<0.05",
        fontsize=10, y=1.01,
    )

    if out_path:
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {out_path}")

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 7. Plot spatial score regions
# ══════════════════════════════════════════════════════════════════════════════

def plot_score_regions_spatial(
    adata,
    score_col: str,
    region_col: str = "hex_score_region",
    cancer_groups: Optional[list] = None,
    sample_group_col: str = "sample_group",
    point_size: float = 6.0,
    out_path: Optional[str] = None,
) -> plt.Figure:
    """
    Spatial scatter showing raw score (left) and hex region labels (right).

    One sample per row.  Left panel: cells coloured by raw score (continuous).
    Right panel: cells coloured by region label (high=red, low=blue, mid=grey).
    Unlabelled cells (NaN hex) are shown as light grey background.

    Parameters
    ----------
    adata            : AnnData with obs: ID, center_x, center_y,
                       sample_group, score_col, region_col.
    score_col        : continuous score column.
    region_col       : region label column from assign_hex_region_labels.
    cancer_groups    : which sample_group values to include (default: all with labels).
    sample_group_col : column holding the cancer group label.
    point_size       : dot size.
    out_path         : optional save path.
    """
    obs = adata.obs.copy()

    for col in ["ID", "center_x", "center_y", region_col, score_col]:
        if col not in obs.columns:
            raise ValueError(f"adata.obs missing column: '{col}'")

    if cancer_groups is not None:
        obs = obs[obs[sample_group_col].isin(cancer_groups)]

    has_region   = obs[region_col].isin(["high", "low", "mid"])
    valid_samples = obs.loc[has_region, "ID"].unique()
    obs = obs[obs["ID"].isin(valid_samples)]

    samples = sorted(obs["ID"].unique())
    if not samples:
        print("No samples with region labels found.")
        return None

    REGION_COLORS = {"high": "#D85A30", "low": "#378ADD", "mid": "#cccccc"}
    CONTEXT_COLOR = "#e8e8e8"
    CONTEXT_ALPHA = 0.4
    CONTEXT_SIZE  = point_size * 0.4

    all_scores = obs[score_col].dropna()
    vmin = float(np.nanpercentile(all_scores, 2))
    vmax = float(np.nanpercentile(all_scores, 98))
    cmap = plt.get_cmap("RdYlBu_r")
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    n_samples = len(samples)
    fig, axes = plt.subplots(
        n_samples, 2,
        figsize=(12, 5 * n_samples),
        constrained_layout=True,
        squeeze=False,
    )

    for row_i, sample_id in enumerate(samples):
        s       = obs[obs["ID"] == sample_id]
        group   = s[sample_group_col].iloc[0] if sample_group_col in s.columns else ""
        has_reg = s[region_col].isin(["high", "low", "mid"])
        lab_df  = s[has_reg]
        ctx_df  = s[~has_reg]

        ax_score  = axes[row_i, 0]
        ax_region = axes[row_i, 1]

        for ax in (ax_score, ax_region):
            ax.set_aspect("equal")
            ax.axis("off")
            if len(ctx_df):
                ax.scatter(ctx_df["center_x"], ctx_df["center_y"],
                           c=CONTEXT_COLOR, s=CONTEXT_SIZE, alpha=CONTEXT_ALPHA,
                           linewidths=0, rasterized=True)

        # Left: continuous score
        scored   = lab_df[lab_df[score_col].notna()]
        unscored = lab_df[lab_df[score_col].isna()]
        if len(unscored):
            ax_score.scatter(unscored["center_x"], unscored["center_y"],
                             c="#cccccc", s=point_size, alpha=0.5, linewidths=0)
        if len(scored):
            sc = ax_score.scatter(
                scored["center_x"], scored["center_y"],
                c=scored[score_col], cmap=cmap, norm=norm,
                s=point_size, linewidths=0, rasterized=True,
            )
            plt.colorbar(sc, ax=ax_score, shrink=0.6, pad=0.02,
                         label=score_col.replace("_", " "))

        ax_score.set_title(f"{sample_id}  [{group}]\nRaw score", fontsize=9)

        # Right: region labels
        for region, color in REGION_COLORS.items():
            region_df = lab_df[lab_df[region_col] == region]
            if len(region_df):
                ax_region.scatter(
                    region_df["center_x"], region_df["center_y"],
                    c=color, s=point_size,
                    alpha=0.85 if region in ("high", "low") else 0.35,
                    linewidths=0, rasterized=True,
                    label=f"{region} (n={len(region_df)})",
                )

        ax_region.legend(loc="upper right", fontsize=7, framealpha=0.7,
                         markerscale=1.5, handletextpad=0.3)
        ax_region.set_title(f"{sample_id}  [{group}]\nHex score region", fontsize=9)

    fig.suptitle(
        f"Spatial hex score regions  ·  score: {score_col}\n"
        f"Left: continuous score  ·  Right: high (red) / mid (grey) / low (blue)\n"
        f"Unlabelled cells (empty hexes) shown as light grey background",
        fontsize=10, y=1.01,
    )

    if out_path:
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {out_path}")

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 8. Plot hex grid with region colours
# ══════════════════════════════════════════════════════════════════════════════

def _hex_corners(cx: float, cy: float, radius: float) -> np.ndarray:
    """
    Return the 6 corner coordinates of a flat-top hexagon centred at (cx, cy).

    Returns np.ndarray of shape (6, 2).
    """
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]   # 0°, 60°, 120°, …, 300°
    return np.column_stack([
        cx + radius * np.cos(angles),
        cy + radius * np.sin(angles),
    ])


def _hex_center_from_axial(q: int, r: int, radius: float):
    """
    Convert flat-top axial hex coordinates (q, r) to Cartesian centre (x, y).
    Must be the inverse of the coordinate transform used in _assign_hex_ids().
    """
    sqrt3 = np.sqrt(3)
    x = radius * (3 / 2 * q)
    y = radius * (sqrt3 / 2 * q + sqrt3 * r)
    return x, y


def plot_hex_grid(
    adata,
    score_col: str,
    hex_radius: float,
    region_col: str = "hex_score_region",
    cancer_groups: Optional[list] = None,
    sample_group_col: str = "sample_group",
    samples: Optional[list] = None,
    show_cells: bool = True,
    cell_size: float = 2.0,
    cell_alpha: float = 0.6,
    hex_alpha: float = 0.45,
    hex_edge_color: str = "white",
    hex_linewidth: float = 0.4,
    figsize_per_panel: tuple = (6, 6),
    out_path: Optional[str] = None,
) -> plt.Figure:
    """
    Draw the hexagonal grid with each hex filled by its region colour.

    Two panels per sample:
      Left  — hexes filled by mean score (continuous colourmap), cells overlaid
               as grey dots for spatial context.
      Right — hexes filled by region label (high=red, mid=grey, low=blue),
               cells overlaid as small dots.

    Use this as the primary diagnostic plot to verify that hex_radius and the
    quantile thresholds produce sensible spatial regions before running the
    enrichment analysis.

    Parameters
    ----------
    adata            : AnnData. obs must contain ID, center_x, center_y,
                       score_col, and region_col (from assign_hex_region_labels).
    score_col        : continuous score column in adata.obs.
    hex_radius       : must match the radius used in assign_hex_region_labels().
    region_col       : region label column written by assign_hex_region_labels.
    cancer_groups    : restrict to samples in these groups. Default: all samples.
    sample_group_col : column holding the cancer group label.
    samples          : explicit list of sample IDs to plot. Overrides
                       cancer_groups if provided.
    show_cells       : overlay cell positions as small dots (default True).
    cell_size        : dot size for cell overlay.
    cell_alpha       : opacity of cell dots.
    hex_alpha        : opacity of hex fill.
    hex_edge_color   : colour of hex border lines.
    hex_linewidth    : width of hex border lines.
    figsize_per_panel: (width, height) of each individual panel.
    out_path         : if given, save the figure here.

    Returns
    -------
    matplotlib Figure
    """
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection
    import matplotlib.cm as cm

    obs = adata.obs.copy()

    for col in ["ID", "center_x", "center_y", score_col, region_col]:
        if col not in obs.columns:
            raise ValueError(f"adata.obs missing column: '{col}'")

    # ── Select samples to plot ────────────────────────────────────────────────
    if samples is not None:
        plot_samples = [s for s in samples if s in obs["ID"].unique()]
    elif cancer_groups is not None:
        if sample_group_col not in obs.columns:
            raise ValueError(f"sample_group_col '{sample_group_col}' not in adata.obs")
        plot_samples = sorted(
            obs.loc[obs[sample_group_col].isin(cancer_groups), "ID"].unique()
        )
    else:
        plot_samples = sorted(obs["ID"].unique())

    # Keep only samples that have at least some labelled hexes
    plot_samples = [
        s for s in plot_samples
        if obs.loc[obs["ID"] == s, region_col].isin(["high", "low", "mid"]).any()
    ]

    if not plot_samples:
        print("No samples with hex region labels found.")
        return None

    REGION_COLORS    = {"high": "#D85A30", "low": "#378ADD", "mid": "#cccccc"}
    UNLABELLED_COLOR = "#f0f0f0"

    # Global score colour scale across all samples
    all_scores = obs[score_col].dropna()
    vmin = float(np.nanpercentile(all_scores, 2))
    vmax = float(np.nanpercentile(all_scores, 98))
    score_norm = plt.Normalize(vmin=vmin, vmax=vmax)
    score_cmap = cm.get_cmap("RdYlBu_r")

    n_samples = len(plot_samples)
    fig, axes = plt.subplots(
        n_samples, 2,
        figsize=(figsize_per_panel[0] * 2, figsize_per_panel[1] * n_samples),
        constrained_layout=True,
        squeeze=False,
    )

    for row_i, sample_id in enumerate(plot_samples):
        s     = obs[obs["ID"] == sample_id].copy()
        group = s[sample_group_col].iloc[0] if sample_group_col in s.columns else ""

        x = s["center_x"].values.astype(float)
        y = s["center_y"].values.astype(float)

        # Re-assign hex IDs using the same radius
        hex_ids  = _assign_hex_ids(x, y, hex_radius)
        s["_hex_id"] = hex_ids

        # Per-hex: mean score + region label (take the first non-null label;
        # all cells in a hex share the same label by construction)
        hex_info = (
            s.groupby("_hex_id")
            .agg(
                mean_score=(score_col,
                            lambda v: v.dropna().mean() if v.notna().any() else np.nan),
                region=(region_col,
                        lambda v: v.dropna().iloc[0] if v.notna().any() else None),
            )
            .reset_index()
        )

        # Parse axial q, r from "q_r" string
        hex_info[["_q", "_r"]] = (
            hex_info["_hex_id"].str.split("_", expand=True).astype(int)
        )

        ax_score  = axes[row_i, 0]
        ax_region = axes[row_i, 1]

        for ax in (ax_score, ax_region):
            ax.set_aspect("equal")
            ax.axis("off")

        # ── Build hex patch collections ───────────────────────────────────────
        score_patches  = []
        score_fcolors  = []
        region_patches = []
        region_fcolors = []

        for _, hrow in hex_info.iterrows():
            cx, cy  = _hex_center_from_axial(int(hrow["_q"]), int(hrow["_r"]), hex_radius)
            corners = _hex_corners(cx, cy, hex_radius)

            # Score panel
            ms = hrow["mean_score"]
            fc = score_cmap(score_norm(ms)) if not np.isnan(ms) else (0.94, 0.94, 0.94, 1.0)
            score_patches.append(MplPolygon(corners, closed=True))
            score_fcolors.append(fc)

            # Region panel
            reg = hrow["region"]
            rc  = REGION_COLORS.get(reg, UNLABELLED_COLOR)
            region_patches.append(MplPolygon(corners, closed=True))
            region_fcolors.append(rc)

        if score_patches:
            ax_score.add_collection(PatchCollection(
                score_patches, facecolor=score_fcolors,
                edgecolor=hex_edge_color, linewidth=hex_linewidth,
                alpha=hex_alpha, zorder=1,
            ))

        if region_patches:
            ax_region.add_collection(PatchCollection(
                region_patches, facecolor=region_fcolors,
                edgecolor=hex_edge_color, linewidth=hex_linewidth,
                alpha=hex_alpha, zorder=1,
            ))

        # ── Overlay cell positions ────────────────────────────────────────────
        if show_cells:
            for ax in (ax_score, ax_region):
                ax.scatter(
                    x, y,
                    c="#444444", s=cell_size, alpha=cell_alpha,
                    linewidths=0, zorder=2, rasterized=True,
                )

        # ── Axis limits (pad by one hex radius) ──────────────────────────────
        pad = hex_radius * 1.2
        for ax in (ax_score, ax_region):
            ax.set_xlim(x.min() - pad, x.max() + pad)
            ax.set_ylim(y.min() - pad, y.max() + pad)

        # ── Colourbar for score panel ─────────────────────────────────────────
        sm = plt.cm.ScalarMappable(cmap=score_cmap, norm=score_norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax_score, shrink=0.55, pad=0.02,
                     label=score_col.replace("_", " "))

        # ── Titles ────────────────────────────────────────────────────────────
        n_hex_total    = len(hex_info)
        n_hex_labelled = hex_info["region"].notna().sum()
        ax_score.set_title(
            f"{sample_id}  [{group}]\nMean score per hex  (radius={hex_radius})",
            fontsize=9,
        )
        ax_region.set_title(
            f"{sample_id}  [{group}]\nRegion label per hex  "
            f"({n_hex_labelled}/{n_hex_total} hexes labelled)",
            fontsize=9,
        )

    # ── Shared legend for region panel ────────────────────────────────────────
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#D85A30", alpha=hex_alpha, label="high"),
        Patch(facecolor="#cccccc", alpha=hex_alpha, label="mid"),
        Patch(facecolor="#378ADD", alpha=hex_alpha, label="low"),
        Patch(facecolor=UNLABELLED_COLOR, alpha=hex_alpha,
              edgecolor="#aaaaaa", label="unlabelled (< min scored cells)"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center", ncol=4,
        fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        f"Hexagonal grid  ·  radius={hex_radius}  ·  score: {score_col}\n"
        f"Left: mean score per hex  ·  Right: region label per hex",
        fontsize=10, y=1.01,
    )

    if out_path:
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {out_path}")

    return fig
