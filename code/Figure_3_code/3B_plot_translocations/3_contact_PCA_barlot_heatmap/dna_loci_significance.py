"""
dna_loci_significance.py
------------------------
Adapted versions of the five core analysis functions for DNA loci data
(block.bin columns).

Key differences from the RNA/chromatin versions in feature_significance.py
--------------------------------------------------------------------------
* No feature groups — every non-meta column is a locus (block.bin format).
* Meta columns use DNA-data naming:
    cellID            → cell identifier
    patient           → sample identifier  (was sample_id)
    Malignant_status  → 'Normal' | 'Malig.'  (was class == 'normal')
    sample_group      → cancer group label
* correct_by_normals_loci() is fully self-contained (no external import).
* A single corrected DataFrame is returned (not two — no loci/contacts split).

Functions
---------
1.  correct_by_normals_loci()
2.  test_feature_significance_loci()
3.  summarise_significant_features_loci()
4.  plot_selected_features_loci()
5.  plot_updown_bars_loci()

Expected DataFrame layout
--------------------------
Rows  : cells
Index : any (cellID also stored as a column)
Cols  : [meta columns]  +  [block.bin locus columns]

Default meta columns (override via META_COLS_DEFAULT or the meta_cols argument):
    cellID, Malignant_status, predicted.celltype,
    patient, type_1, type_2, resection, sample_group
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

from typing import Optional

# ── Illustrator-compatible export settings ────────────────────────────────────
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["ps.fonttype"]  = 42

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "axes.linewidth":    0.5,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "figure.dpi":        300,
    "pdf.fonttype":      42,
    "path.simplify":     False,
    "figure.facecolor":  "none",
    "axes.facecolor":    "none",
})

# ── Palettes ─────────────────────────────────────────────────────────────────
C_NORMAL_DEFAULT    = "#AAAAAA"
GROUP_PALETTE_DEFAULT = ["#F5A623", "#E8604C", "#9B59B6", "#2ECC71", "#C0392B",
                         "#4878CF", "#2ECC71"]

# Standard meta columns for the DNA loci DataFrame.
# Pass a custom list to any function via the `meta_cols` argument.
META_COLS_DEFAULT = [
    "cellID", "Malignant_status", "predicted.celltype",
    "patient", "type_1", "type_2", "resection", "sample_group",
]


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _loci_cols(df: pd.DataFrame, meta_cols: list[str]) -> list[str]:
    """Return all columns that are not meta columns."""
    return [
        c for c in df.columns
        if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

def _bh_correction(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction (fallback for older scipy)."""
    n      = len(p_values)
    order  = np.argsort(p_values)
    ranks  = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    p_adj  = np.minimum(1.0, p_values * n / ranks)
    for i in range(n - 2, -1, -1):
        p_adj[order[i]] = min(p_adj[order[i]], p_adj[order[i + 1]])
    return p_adj


def _apply_bh(df: pd.DataFrame, group_col: str, pval_col: str = "p_value") -> np.ndarray:
    """Apply BH correction within each level of group_col; return p_adj array."""
    p_adj_all = np.full(len(df), np.nan)
    for grp, idx in df.groupby(group_col).groups.items():
        pv    = df.loc[idx, pval_col].values
        valid = ~np.isnan(pv)
        padj  = np.full(len(pv), np.nan)
        if valid.sum() > 1:
            padj[valid] = (
                stats.false_discovery_control(pv[valid], method="bh", axis=0)
                if hasattr(stats, "false_discovery_control")
                else _bh_correction(pv[valid])
            )
        elif valid.sum() == 1:
            padj[valid] = pv[valid]          # single test: p_adj = p
        p_adj_all[df.index.get_indexer(idx)] = padj
    return p_adj_all


# ══════════════════════════════════════════════════════════════════════════════
# 1. correct_by_normals_loci
# ══════════════════════════════════════════════════════════════════════════════

def correct_by_normals_loci(
    df: pd.DataFrame,
    id_col = 'ROI',
    method: str = "zscore",
    normal_cell_types: Optional[list[str]] = None,
    meta_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Correct DNA loci values by the per-sample normal-cell baseline.

    For each patient/sample:
      1. Identify baseline normal cells:
           Malignant_status == 'Normal'
           (optionally restricted to predicted.celltype in normal_cell_types)
      2. Apply correction per locus:
           zscore   : (x - µ_normal) / σ_normal   → null = 0
           quantile : fraction of normal cells ≤ x  → null ≈ 0.5

    Samples with fewer than 2 normal cells get NaN-corrected values.

    Parameters
    ----------
    df                : cell × (meta + loci) DataFrame.
    method            : 'zscore' or 'quantile'.
    normal_cell_types : optional list of predicted.celltype values to use
                        as the normal baseline. If None, all cells with
                        Malignant_status == 'Normal' are used.
    meta_cols         : meta column names to exclude from correction.
                        Defaults to META_COLS_DEFAULT.

    Returns
    -------
    pd.DataFrame — same structure as input; loci columns replaced by
    corrected values. Meta columns are carried through unchanged.
    """
    if method not in ("zscore", "quantile"):
        raise ValueError(f"method must be 'zscore' or 'quantile', got '{method}'")

    if meta_cols is None:
        meta_cols = META_COLS_DEFAULT
    loci = _loci_cols(df, meta_cols)
    if not loci:
        raise ValueError("No loci columns found. Check meta_cols.")

    corrected = df.copy()
    corrected[loci] = np.nan   # fill with NaN; overwrite per sample below

    samples = df[id_col].unique()
    n_no_normals = 0

    for sample in samples:
        sample_mask = df[id_col] == sample

        # Normal baseline cells for this patient
        norm_mask = sample_mask & (df["Malignant_status"] == "Normal")
        if normal_cell_types is not None and "cell_type" in df.columns:
            norm_mask = norm_mask & df["cell_type"].isin(normal_cell_types)

        norm_idx = df.index[norm_mask]
        all_idx  = df.index[sample_mask]

        if len(norm_idx) < 2:
            n_no_normals += 1
            # Leave as NaN
            continue

        X_norm = df.loc[norm_idx, loci].values.astype(float)   # (n_normals, n_loci)
        X_all  = df.loc[all_idx,  loci].values.astype(float)   # (n_cells,   n_loci)

        if method == "zscore":
            mu  = X_norm.mean(axis=0)                           # (n_loci,)
            sd  = X_norm.std(axis=0, ddof=1)
            sd[sd == 0] = np.nan                                # avoid /0
            corrected.loc[all_idx, loci] = (X_all - mu) / sd

        else:  # quantile
            # For each cell × locus: fraction of normal values ≤ cell value
            # = empirical CDF evaluated at x → range [0, 1], median ≈ 0.5
            result = np.empty_like(X_all)
            n_norm = len(norm_idx)
            for j in range(X_all.shape[1]):
                col_norm = X_norm[:, j]
                for i, x in enumerate(X_all[:, j]):
                    result[i, j] = np.sum(col_norm <= x) / n_norm
            corrected.loc[all_idx, loci] = result

    if n_no_normals:
        print(f"  Warning: {n_no_normals} sample(s) had <2 normal cells "
              f"— their corrected values are NaN.")

    null_value = 0.0 if method == "zscore" else 0.5
    print(f"  Correction done [{method}]. "
          f"Null = {null_value}. "
          f"{len(samples) - n_no_normals}/{len(samples)} samples corrected.")
    return corrected
