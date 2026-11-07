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
    method: str = "zscore",
    id_col      = "ROI", 
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
        if normal_cell_types is not None and "predicted.celltype" in df.columns:
            norm_mask = norm_mask & df["predicted.celltype"].isin(normal_cell_types)

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


# ══════════════════════════════════════════════════════════════════════════════
# 2. test_feature_significance_loci
# ══════════════════════════════════════════════════════════════════════════════

def test_feature_significance_loci(
    df_original: pd.DataFrame,
    loci_corrected: pd.DataFrame,
    cancer_groups: list[str],
    method: str,
    alpha: float = 0.05,
    min_samples: int = 2,
    test_vs_null: bool = True,
    test_between_groups: bool = False,
    meta_cols: Optional[list[str]] = None,
) -> dict[str, pd.DataFrame]:
    """
    Significance tests on per-patient means of corrected DNA loci values.

    Test 1 — vs null  (test_vs_null=True)
    ---------------------------------------
    For each cancer group × locus: one-sample t-test of the per-patient means
    against the correction null (0 for zscore, 0.5 for quantile).
    BH correction is applied per cancer group.

    Test 2 — between groups  (test_between_groups=True)
    -----------------------------------------------------
    For every pair of cancer groups × locus that is significant in Test 1:
    two-sample Welch t-test comparing per-patient means between the two groups.
    BH correction is applied per pair.

    Parameters
    ----------
    df_original      : original (uncorrected) DataFrame — used for metadata.
    loci_corrected   : corrected DataFrame from correct_by_normals_loci().
    cancer_groups    : list of sample_group values to test.
    method           : 'zscore' or 'quantile' — determines the null value.
    alpha            : FDR threshold for significance (default 0.05).
    min_samples      : minimum number of patients required per group.
    test_vs_null     : run Test 1 (default True).
    test_between_groups : run Test 2 (default False).
    meta_cols        : meta columns to exclude from feature list.

    Returns
    -------
    dict with keys 'vs_null' and/or 'between_groups' (pd.DataFrames).

    vs_null columns:
        locus, cancer_group, n_samples, mean_of_sample_means,
        std_of_sample_means, t_stat, p_value, p_adj, significant, direction

    between_groups columns:
        locus, group_a, group_b, n_samples_a, n_samples_b,
        mean_a, mean_b, t_stat, p_value, p_adj, significant, higher_group
    """
    assert test_vs_null or test_between_groups, \
        "At least one of test_vs_null or test_between_groups must be True."

    if meta_cols is None:
        meta_cols = META_COLS_DEFAULT
    null_value = 0.0 if method == "zscore" else 0.5

    loci = _loci_cols(loci_corrected, meta_cols)

    # ── Build a corrected frame with only malignant cells and needed meta ─────
    # Merge meta from df_original onto the corrected values via cellID
    meta = df_original[["cell_id", "ROI", "Malignant_status", "sample_group"]].copy()
    df_corr = meta.merge(
        loci_corrected[["cell_id"] + loci],
        on="cell_id", how="inner",
    )
    mal_df = df_corr[df_corr["Malignant_status"] != "Normal"].copy()

    # ── Pre-compute per-patient means for each cancer group ───────────────────
    group_patient_means: dict[str, pd.DataFrame] = {}
    valid_groups: list[str] = []

    for group in cancer_groups:
        gdf      = mal_df[mal_df["sample_group"] == group]
        patients = gdf["ROI"].unique()
        if len(patients) < min_samples:
            print(f"  Skipping '{group}': only {len(patients)} patient(s) "
                  f"(need ≥ {min_samples})")
            continue
        group_patient_means[group] = gdf.groupby("ROI")[loci].mean()
        valid_groups.append(group)

    output: dict[str, pd.DataFrame] = {}

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 1 — each group vs null
    # ══════════════════════════════════════════════════════════════════════════
    if test_vs_null:
        print(f"\nTest 1: each group vs null ({null_value}) "
              f"[one-sample t-test, BH per group]")

        records = []
        for group in cancer_groups:
            n_patients = len(
                mal_df[mal_df["sample_group"] == group]["patient"].unique()
            )
            if group not in valid_groups:
                for locus in loci:
                    records.append({
                        "locus":                locus,
                        "cancer_group":         group,
                        "n_samples":            n_patients,
                        "mean_of_sample_means": np.nan,
                        "std_of_sample_means":  np.nan,
                        "t_stat":               np.nan,
                        "p_value":              np.nan,
                        "p_adj":                np.nan,
                        "significant":          False,
                        "direction":            np.nan,
                    })
                continue

            pat_means = group_patient_means[group]   # (n_patients, n_loci)

            for locus in loci:
                col = pat_means[locus].dropna().values
                if len(col) < min_samples:
                    t_stat, p_val = np.nan, np.nan
                else:
                    t_stat, p_val = stats.ttest_1samp(col, popmean=null_value)

                records.append({
                    "locus":                locus,
                    "cancer_group":         group,
                    "n_samples":            len(pat_means),
                    "mean_of_sample_means": float(np.nanmean(col)) if len(col) else np.nan,
                    "std_of_sample_means":  float(np.nanstd(col))  if len(col) else np.nan,
                    "t_stat":               float(t_stat) if not np.isnan(t_stat) else np.nan,
                    "p_value":              float(p_val)  if not np.isnan(p_val)  else np.nan,
                    "p_adj":                np.nan,
                    "significant":          False,
                    "direction":            np.nan,
                })

        res_null = pd.DataFrame(records)

        # BH correction per cancer group
        res_null = res_null.reset_index(drop=True)
        res_null["p_adj"] = _apply_bh(res_null, group_col="cancer_group")

        res_null["significant"] = res_null["p_adj"] < alpha
        res_null["direction"] = np.where(
            res_null["mean_of_sample_means"] > null_value, "elevated", "depleted"
        )
        res_null.loc[~res_null["significant"], "direction"] = np.nan

        n_sig = res_null["significant"].sum()
        print(f"  {n_sig} significant locus × group combinations (FDR < {alpha})")
        output["vs_null"] = res_null

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 2 — pairwise between groups
    # ══════════════════════════════════════════════════════════════════════════
    if test_between_groups:
        from itertools import combinations

        if "vs_null" in output:
            sig_loci = set(
                output["vs_null"].loc[output["vs_null"]["significant"], "locus"]
            )
            test_loci = [l for l in loci if l in sig_loci]
            print(f"\nTest 2: restricted to {len(test_loci)} loci significant "
                  f"in Test 1 (out of {len(loci)} total)")
        else:
            test_loci = loci
            print(f"\nTest 2: using all {len(test_loci)} loci "
                  f"(Test 1 was not run)")

        # Direction lookup from Test 1
        feat_direction: dict[str, str] = {}
        if "vs_null" in output:
            sig_null = output["vs_null"][output["vs_null"]["significant"]]
            feat_direction = (
                sig_null.dropna(subset=["direction"])
                .groupby("locus")["direction"]
                .first()
                .to_dict()
            )

        pairs = list(combinations(valid_groups, 2))
        print(f"  {len(pairs)} pair(s), Welch t-test, BH per pair")

        records_pairs = []
        for group_a, group_b in pairs:
            means_a = group_patient_means[group_a]
            means_b = group_patient_means[group_b]

            for locus in test_loci:
                va = means_a[locus].dropna().values if locus in means_a.columns else np.array([])
                vb = means_b[locus].dropna().values if locus in means_b.columns else np.array([])

                if len(va) < min_samples or len(vb) < min_samples:
                    t_stat, p_val = np.nan, np.nan
                else:
                    t_stat, p_val = stats.ttest_ind(va, vb, equal_var=False)

                records_pairs.append({
                    "locus":         locus,
                    "group_a":       group_a,
                    "group_b":       group_b,
                    "n_samples_a":   len(va),
                    "n_samples_b":   len(vb),
                    "mean_a":        float(np.nanmean(va)) if len(va) else np.nan,
                    "mean_b":        float(np.nanmean(vb)) if len(vb) else np.nan,
                    "t_stat":        float(t_stat) if not np.isnan(t_stat) else np.nan,
                    "p_value":       float(p_val)  if not np.isnan(p_val)  else np.nan,
                    "p_adj":         np.nan,
                    "significant":   False,
                    "higher_group":  np.nan,
                    "direction":     feat_direction.get(locus, np.nan),
                })

        res_pairs = pd.DataFrame(records_pairs).reset_index(drop=True)

        # BH correction per pair using a combined pair-key column
        res_pairs["_pair_key"] = res_pairs["group_a"] + "||" + res_pairs["group_b"]
        res_pairs["p_adj"] = _apply_bh(res_pairs, group_col="_pair_key")
        res_pairs.drop(columns=["_pair_key"], inplace=True)

        res_pairs["significant"] = res_pairs["p_adj"] < alpha
        res_pairs["higher_group"] = np.where(
            res_pairs["mean_a"] >= res_pairs["mean_b"],
            res_pairs["group_a"],
            res_pairs["group_b"],
        )
        res_pairs.loc[~res_pairs["significant"], "higher_group"] = np.nan

        n_sig = res_pairs["significant"].sum()
        print(f"  {n_sig} significant locus × pair combinations (FDR < {alpha})")
        output["between_groups"] = res_pairs

    return output


# ══════════════════════════════════════════════════════════════════════════════
# 3. summarise_significant_features_loci
# ══════════════════════════════════════════════════════════════════════════════

def summarise_significant_features_loci(
    results: dict[str, pd.DataFrame] | pd.DataFrame,
    cancer_groups: list[str],
) -> dict[str, pd.DataFrame]:
    """
    Print and return human-readable summaries of significant loci.

    Test 1 (vs_null): loci significantly different from the correction null
    in each cancer group, broken down by direction (elevated / depleted).

    Test 2 (between_groups): subset of Test-1 loci that are also significantly
    different between cancer groups.

    Parameters
    ----------
    results       : dict returned by test_feature_significance_loci(), or a
                    bare vs_null DataFrame (for backwards compatibility).
    cancer_groups : list of cancer group labels (for display order).

    Returns
    -------
    dict with up to three keys:
        'vs_null'        : per-group direction counts (pd.DataFrame)
        'between_groups' : per-pair higher-group counts (pd.DataFrame)
        'joint'          : one row per significant locus with direction +
                           between-group pair information (pd.DataFrame)
    """
    if isinstance(results, pd.DataFrame):
        results = {"vs_null": results}

    summaries: dict[str, pd.DataFrame] = {}

    # ── Test 1 summary ────────────────────────────────────────────────────────
    if "vs_null" in results:
        res = results["vs_null"]
        sig = res[res["significant"]].copy()

        # Counts by cancer_group × direction
        summary = (
            sig.groupby(["cancer_group", "direction"])
            .size()
            .reset_index(name="n_loci")
        )
        totals = (
            sig.groupby("cancer_group")
            .size()
            .reset_index(name="n_loci")
            .assign(direction="TOTAL")
        )
        summary = (
            pd.concat([summary, totals], ignore_index=True)
            .sort_values(["cancer_group", "direction"])
        )

        print("\n── Test 1: significant loci vs null ──────────────────────────────")
        print(f"{'Cancer group':<28} {'Direction':<12} {'N loci':>8}")
        print("─" * 52)
        for _, row in summary.iterrows():
            print(f"{row['cancer_group']:<28} {row['direction']:<12} {row['n_loci']:>8}")

        summaries["vs_null"] = summary

    # ── Test 2 summary ────────────────────────────────────────────────────────
    if "between_groups" in results:
        res_btw = results["between_groups"]
        sig_btw = res_btw[res_btw["significant"]].copy()

        summary = (
            sig_btw.groupby(["group_a", "group_b", "higher_group"])
            .size()
            .reset_index(name="n_loci")
        )
        totals = (
            sig_btw.groupby(["group_a", "group_b"])
            .size()
            .reset_index(name="n_loci")
            .assign(higher_group="TOTAL")
        )
        summary = (
            pd.concat([summary, totals], ignore_index=True)
            .sort_values(["group_a", "group_b", "higher_group"])
        )

        print("\n── Test 2: loci sig between groups (subset of Test 1) ────────────")
        print(f"{'Group A':<22} {'Group B':<22} {'Higher':<22} {'N loci':>8}")
        print("─" * 76)
        for _, row in summary.iterrows():
            print(f"{row['group_a']:<22} {row['group_b']:<22} "
                  f"{str(row['higher_group']):<22} {row['n_loci']:>8}")

        summaries["between_groups"] = summary

    # ── Joint summary — one row per significant locus ─────────────────────────
    if "vs_null" in results and "between_groups" in results:
        res_null = results["vs_null"]
        res_btw  = results["between_groups"]
        sig_null = res_null[res_null["significant"]].copy()

        records = []
        for locus, grp in sig_null.groupby("locus"):
            sig_groups  = grp["cancer_group"].tolist()
            directions  = grp["direction"].dropna().unique().tolist()
            direction   = directions[0] if len(directions) == 1 else "mixed"

            # Between-group pairs that are also significant for this locus
            btw_pairs = []
            if locus in res_btw["locus"].values:
                locus_btw = res_btw[(res_btw["locus"] == locus) & res_btw["significant"]]
                for _, row in locus_btw.iterrows():
                    btw_pairs.append(
                        f"{row['group_a']} vs {row['group_b']} "
                        f"(higher: {row['higher_group']})"
                    )

            records.append({
                "locus":             locus,
                "direction":         direction,
                "sig_in_groups":     ", ".join(sig_groups),
                "n_sig_groups":      len(sig_groups),
                "sig_between_pairs": "; ".join(btw_pairs) if btw_pairs else "—",
                "n_sig_pairs":       len(btw_pairs),
            })

        joint = pd.DataFrame(records).sort_values(
            ["n_sig_pairs", "n_sig_groups"], ascending=False
        )

        n_both = (joint["n_sig_pairs"] > 0).sum()
        print(f"\n── Joint: {len(joint)} loci sig vs null, "
              f"{n_both} also sig between groups ──")
        print(f"{'Locus':<30} {'Dir':<12} {'Groups':>8} {'Pairs':>7}")
        print("─" * 60)
        for _, row in joint.head(20).iterrows():
            print(f"{row['locus']:<30} {row['direction']:<12} "
                  f"{row['n_sig_groups']:>8} {row['n_sig_pairs']:>7}")
        if len(joint) > 20:
            print(f"  ... {len(joint) - 20} more (see summaries['joint'])")

        summaries["joint"] = joint

    return summaries


# ══════════════════════════════════════════════════════════════════════════════
# 4. plot_selected_features_loci
# ══════════════════════════════════════════════════════════════════════════════

def plot_selected_features_loci(
    loci: list[str],
    df_original: pd.DataFrame,
    loci_corrected: pd.DataFrame,
    cancer_groups: list[str],
    method: str,
    results: Optional[pd.DataFrame] = None,
    out_path: Optional[str] = None,
    group_order: Optional[list[str]] = None,
    meta_cols: Optional[list[str]] = None,
) -> plt.Figure:
    """
    Plot per-patient corrected means for a user-specified list of loci.

    Each locus gets one panel. Within each panel:
      - A boxplot (IQR + median) provides the background distribution summary.
      - Each dot is one patient's mean corrected value for its malignant cells.
      - The leftmost column always shows the same statistic for normal cells.
      - A dashed line marks the correction null (0 for zscore, 0.5 for quantile).
      - If `results` is provided, significant loci are annotated with stars.
      - All subplots share a common y-axis range.

    Parameters
    ----------
    loci             : list of block.bin locus names to plot.
    df_original      : original (pre-correction) DataFrame (for metadata).
    loci_corrected   : corrected DataFrame from correct_by_normals_loci().
    cancer_groups    : cancer group labels present in sample_group.
    method           : 'zscore' or 'quantile'.
    results          : (optional) vs_null DataFrame from
                       test_feature_significance_loci(). When supplied,
                       significant loci are annotated (* / ** / ***).
    out_path         : (optional) file path to save the figure.
    group_order      : (optional) explicit left-to-right order for cancer
                       groups. Normal cells are always plotted first.
    meta_cols        : meta columns to exclude from loci list.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not loci:
        raise ValueError("`loci` must be a non-empty list.")

    if meta_cols is None:
        meta_cols = META_COLS_DEFAULT

    null_value     = 0.0 if method == "zscore" else 0.5
    ordered_groups = group_order if group_order is not None else list(cancer_groups)
    group_colors   = {g: GROUP_PALETTE_DEFAULT[i % len(GROUP_PALETTE_DEFAULT)]
                      for i, g in enumerate(ordered_groups)}

    # ── Merge meta + corrected values ─────────────────────────────────────────
    all_loci_in_corr = _loci_cols(loci_corrected, meta_cols)
    meta    = df_original[["cellID", "patient", "Malignant_status", "sample_group"]]
    df_corr = meta.merge(
        loci_corrected[["cellID"] + all_loci_in_corr],
        on="cellID", how="inner",
    )
    mal_df  = df_corr[df_corr["Malignant_status"] != "Normal"]
    norm_df = df_corr[df_corr["Malignant_status"] == "Normal"]

    # Validate requested loci
    available = set(df_corr.columns) - {"cellID", "patient", "Malignant_status", "sample_group"}
    missing   = [l for l in loci if l not in available]
    if missing:
        print(f"  Warning: the following loci were not found in the corrected "
              f"data and will be skipped: {missing}")
    plot_loci = [l for l in loci if l in available]

    if not plot_loci:
        print("  No valid loci to plot.")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No valid loci found",
                ha="center", va="center", transform=ax.transAxes)
        if out_path:
            fig.savefig(out_path, bbox_inches="tight", transparent=True, dpi=300)
        return fig

    # ── Significance lookup ───────────────────────────────────────────────────
    sig_lookup: pd.DataFrame | None = None
    if results is not None and "significant" in results.columns:
        sig = results[results["significant"]].copy()
        if not sig.empty:
            sig_lookup = sig

    # ── Global y-limits (shared across all panels) ───────────────────────────
    all_vals = []
    for l in plot_loci:
        if l in norm_df.columns:
            all_vals.extend(norm_df.groupby("patient")[l].mean().tolist())
        for g in ordered_groups:
            gdf = mal_df[mal_df["sample_group"] == g]
            if l in gdf.columns:
                all_vals.extend(gdf.groupby("patient")[l].mean().tolist())

    all_vals = [v for v in all_vals if not np.isnan(v)]
    if all_vals:
        v_min, v_max = np.nanmin(all_vals), np.nanmax(all_vals)
        pad  = (v_max - v_min) * 0.15 or 0.1
        y_lo = v_min - pad
        y_hi = v_max + pad
    else:
        y_lo, y_hi = -1, 1

    # ── Layout ────────────────────────────────────────────────────────────────
    n_panels = len(plot_loci)
    ncols    = min(n_panels, 4)
    nrows    = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2.4 * ncols, 5 * nrows),
        sharey=True,
    )
    fig.patch.set_visible(False)
    axes_flat = np.array(axes).flatten() if n_panels > 1 else [axes]

    normal_x_pos = -1   # normal is always the leftmost column

    for ax_i, locus in enumerate(plot_loci):
        ax = axes_flat[ax_i]

        # ── Normal reference column ───────────────────────────────────────────
        if locus in norm_df.columns:
            norm_means = norm_df.groupby("patient")[locus].mean().dropna()
            if len(norm_means) > 0:
                ax.boxplot(
                    norm_means.values,
                    positions=[normal_x_pos],
                    widths=0.35,
                    patch_artist=True,
                    boxprops=dict(facecolor=C_NORMAL_DEFAULT, alpha=0.25,
                                  linewidth=0.8, edgecolor=C_NORMAL_DEFAULT),
                    medianprops=dict(color=C_NORMAL_DEFAULT, linewidth=1.5),
                    whiskerprops=dict(color=C_NORMAL_DEFAULT, linewidth=0.8),
                    capprops=dict(color=C_NORMAL_DEFAULT, linewidth=0.8),
                    flierprops=dict(marker="", markersize=0),
                    zorder=2, manage_ticks=False,
                )
                rng    = np.random.default_rng(0)
                jitter = rng.uniform(-0.12, 0.12, len(norm_means))
                ax.scatter(
                    np.full(len(norm_means), normal_x_pos) + jitter,
                    norm_means.values,
                    color=C_NORMAL_DEFAULT, s=20, alpha=0.75, zorder=3,
                    label="Normal" if ax_i == 0 else "_nolegend_",
                )

        # ── Cancer groups ─────────────────────────────────────────────────────
        for g_i, group in enumerate(ordered_groups):
            gdf          = mal_df[mal_df["sample_group"] == group]
            pat_means    = (
                gdf.groupby("patient")[locus].mean().dropna()
                if locus in gdf.columns
                else pd.Series(dtype=float)
            )
            color = group_colors[group]
            x_pos = g_i   # normal sits at -1

            if len(pat_means) > 0:
                ax.boxplot(
                    pat_means.values,
                    positions=[x_pos],
                    widths=0.35,
                    patch_artist=True,
                    boxprops=dict(facecolor=color, alpha=0.20,
                                  linewidth=0.8, edgecolor=color),
                    medianprops=dict(color=color, linewidth=1.5),
                    whiskerprops=dict(color=color, linewidth=0.8),
                    capprops=dict(color=color, linewidth=0.8),
                    flierprops=dict(marker="", markersize=0),
                    zorder=2, manage_ticks=False,
                )
                rng    = np.random.default_rng(42 + g_i)
                jitter = rng.uniform(-0.12, 0.12, len(pat_means))
                ax.scatter(
                    np.full(len(pat_means), x_pos) + jitter,
                    pat_means.values,
                    color=color, s=20, alpha=0.85, zorder=3,
                    label=group if ax_i == 0 else "_nolegend_",
                )

            # Significance annotation
            if sig_lookup is not None and "locus" in sig_lookup.columns:
                row = sig_lookup[
                    (sig_lookup["locus"] == locus) &
                    (sig_lookup.get("cancer_group", pd.Series()) == group)
                ]
                if not row.empty and row["significant"].values[0]:
                    p     = row["p_adj"].values[0]
                    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*"
                    ax.text(x_pos, y_hi * 0.97, stars,
                            ha="center", va="top",
                            color=color, fontsize=10, fontweight="bold")

        # Null reference line
        ax.axhline(null_value, color="grey", lw=1.0, ls="--", alpha=0.6,
                   label=f"null ({null_value})" if ax_i == 0 else "_nolegend_")

        ax.set_ylim(y_lo, y_hi)

        # Panel title: shorten locus name for display
        ax.set_title(locus, fontsize=8)

        x_tick_pos    = [normal_x_pos] + list(range(len(ordered_groups)))
        x_tick_labels = ["Normal"] + [g.split("|")[0].strip() for g in ordered_groups]
        ax.set_xticks(x_tick_pos)
        ax.set_xticklabels(x_tick_labels, fontsize=8, rotation=15, ha="right")
        ax.set_xlim(normal_x_pos - 0.6, len(ordered_groups) - 0.4)

        if ax_i % ncols == 0:
            ax.set_ylabel(f"{method} corrected value", fontsize=8)

    # Remove unused panels
    for ax_i in range(len(plot_loci), len(axes_flat)):
        fig.delaxes(axes_flat[ax_i])

    # Shared legend
    handles = (
        [plt.Line2D([0], [0], marker="o", color="w",
                    markerfacecolor=C_NORMAL_DEFAULT, markersize=8,
                    label="Normal")]
        + [plt.Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=group_colors[g], markersize=8, label=g)
           for g in ordered_groups]
        + [plt.Line2D([0], [0], ls="--", color="grey",
                      label=f"null ({null_value})")]
    )
    fig.legend(handles=handles, loc="lower center",
               ncol=len(ordered_groups) + 2,
               fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    annot_note = " · * p_adj<0.05" if sig_lookup is not None else ""
    fig.suptitle(
        f"Selected loci  [{method.upper()} correction]\n"
        f"Dots = per-patient means · Box = IQR / median{annot_note}",
        fontsize=10, y=1.01,
    )

    if out_path:
        fig.savefig(out_path, bbox_inches="tight", transparent=True, dpi=300)
        print(f"Saved: {out_path}")

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 5. plot_updown_bars_loci
# ══════════════════════════════════════════════════════════════════════════════

def plot_updown_bars_loci(
    df: pd.DataFrame,
    threshold: float = 0.15,
    figsize: Optional[tuple] = None,
    group_order: Optional[list] = None,
    out_path: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    100% stacked bar chart: for each patient, the fraction of loci that are
    elevated (red), depleted (blue), or unchanged (light gray) relative to
    the correction threshold.

    Patients are grouped by sample_group, with a gap between groups.
    Within each group, patients are sorted by fraction unchanged (descending),
    so the most dysregulated patients appear on the right.

    Parameters
    ----------
    df          : pd.DataFrame with columns:
                    'ID'      — patient/sample identifier
                    'sample_group' — group label
                    all other columns are treated as loci (corrected values).
                  One row per patient (i.e. per-patient means should be
                  pre-computed before passing to this function).
    threshold   : absolute value cutoff.
                    > +threshold  → elevated
                    < -threshold  → depleted
                    otherwise     → unchanged
    figsize     : (width, height) in inches. Auto-sized if None.
    group_order : explicit left-to-right group order. Defaults to order of
                  appearance in df.
    out_path    : if provided, save as PDF to this path.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    df = df.reset_index(drop=True).copy()

    meta_cols = ["ID", "sample_group"]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    #feat_cols = [c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]
    data      = df[feat_cols].values.astype(float)
    n_feat    = data.shape[1]

    frac_up      = (data >  threshold).sum(axis=1) / n_feat
    frac_down    = (data < -threshold).sum(axis=1) / n_feat
    frac_neutral = 1.0 - frac_up - frac_down

    UP_COLOR      = "#c0392b"
    DOWN_COLOR    = "#2980b9"
    NEUTRAL_COLOR = "#d8d8d8"
    BAR_WIDTH     = 0.7
    GROUP_GAP     = 1.2

    if group_order is None:
        groups = list(dict.fromkeys(df["sample_group"].tolist()))
    else:
        groups = group_order

    group_palette = {
        g: c for g, c in zip(
            groups,
            ["#1d6fa4", "#c0582a", "#2e8b57",
             "#7b3fa0", "#b5860d", "#a0522d", "#4682b4"],
        )
    }

    # Build plot order: within each group sort by frac_neutral descending
    plot_order    = []
    x_values      = []
    group_label_x = {}
    group_spans   = {}
    x_cursor      = 0

    for g in groups:
        row_positions = df.index[df["sample_group"] == g].tolist()
        row_positions = sorted(
            row_positions, key=lambda p: frac_neutral[p], reverse=True
        )
        g_start = x_cursor
        for pos in row_positions:
            plot_order.append(pos)
            x_values.append(x_cursor)
            x_cursor += 1
        group_label_x[g] = (g_start + x_cursor - 1) / 2
        group_spans[g]   = (g_start - 0.5, x_cursor - 0.5)
        x_cursor += GROUP_GAP

    plot_order = np.array(plot_order)
    x_values   = np.array(x_values, dtype=float)

    n_samples = len(df)
    if figsize is None:
        figsize = (max(8, n_samples * 0.45 + 2), 5)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#f8f7f4")
    ax.set_facecolor("#f8f7f4")

    for i, pos in enumerate(plot_order):
        x  = x_values[i]
        fd = frac_down[pos]
        fn = frac_neutral[pos]
        fu = frac_up[pos]

        ax.bar(x, fd, width=BAR_WIDTH, bottom=0,
               color=DOWN_COLOR, alpha=0.88, linewidth=0)
        ax.bar(x, fu, width=BAR_WIDTH, bottom=fd,
               color=UP_COLOR, alpha=0.88, linewidth=0)
        ax.bar(x, fn, width=BAR_WIDTH, bottom=fd + fu,
               color=NEUTRAL_COLOR, alpha=0.9, linewidth=0)

    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"],
                       fontsize=8, color="#666666")

    for i, (g, (x0, x1)) in enumerate(group_spans.items()):
        col = group_palette.get(g, "#888888")
        if i > 0:
            ax.axvline(x0, color="#cccccc", linewidth=0.6,
                       linestyle="--", zorder=0)
        ax.text(group_label_x[g], 1.15, g,
                ha="center", va="top",
                fontsize=9, fontweight="bold", color=col)

    ax.set_xticks(x_values)
    ax.set_xticklabels(
        df["ID"].iloc[plot_order].tolist(),
        rotation=45, ha="right", fontsize=7, color="#444444",
    )
    ax.set_xlim(x_values.min() - 0.8, x_values.max() + 0.8)

    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.tick_params(axis="both", length=0)

    legend_handles = [
        mpatches.Patch(color=UP_COLOR,      alpha=0.88,
                       label=f"Elevated (> +{threshold})"),
        mpatches.Patch(color=NEUTRAL_COLOR, alpha=0.90,
                       label="Unchanged"),
        mpatches.Patch(color=DOWN_COLOR,    alpha=0.88,
                       label=f"Depleted (< -{threshold})"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8,
              frameon=True, framealpha=0.85, edgecolor="#cccccc")

    ax.set_title(
        f"Locus direction per patient  |  threshold = ±{threshold}",
        fontsize=10, color="#333333", pad=10,
    )

    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
        print(f"Saved: {out_path}")

    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
# 6. plot_annotated_heatmap_loci
# ══════════════════════════════════════════════════════════════════════════════

def plot_annotated_heatmap_loci(
    df: pd.DataFrame,
    loci_corrected: pd.DataFrame,
    sig_loci: list[str],
    loci_map: pd.DataFrame,
    out_path: str = "loci_heatmap_annotated.pdf",
    main_types: Optional[list[str]] = None,
    extra_type_keywords: Optional[list[str]] = None,
    figsize_width_per_locus: float = 0.025,
    figsize_base_height: float = 6.0,
    cmap: str = "RdBu_r",
    vmax_pct: float = 95,
    meta_cols: Optional[list[str]] = None,
) -> plt.Figure:
    """
    Clustered annotated heatmap of significant DNA loci.

    Rows   : per-patient × class means (clustered by hierarchical clustering).
    Columns: significant loci in the genomic order defined by `loci_map_path`
             (NO column clustering). Columns are annotated with chromosome and
             chromosomal arm (p / q) using coloured strips above the heatmap.

    Parameters
    ----------
    df                     : cell × (meta + loci) DataFrame with columns:
                               cellID, patient, Malignant_status, sample_group
                             (and the raw loci values, not used for plotting
                              — only for building the row metadata).
    loci_corrected         : corrected DataFrame from correct_by_normals_loci().
                             Must contain 'cellID' plus all loci columns.
    sig_loci               : list of locus column names to display
                             (e.g. output of test_feature_significance_loci).
    loci_map               : DataFrame with locus metadata.
                             Expected columns: blockbin, arm, chromosome_true.
                             Used to order columns and build the chromosome /
                             arm annotation strips
                             (e.g. pd.read_csv("DNA_loci_ordered.csv")).
    out_path               : PDF output path.
    main_types             : sample_group values for the main panel.
                             If None, inferred as all groups NOT matching
                             extra_type_keywords.
    extra_type_keywords    : keywords identifying additional cancer groups to
                             show in a separate panel below (e.g. ['GBM']).
                             If None, all groups go in the main panel.
    figsize_width_per_locus: figure width per locus column (inches). Default
                             0.025 — adjust for very wide or narrow heatmaps.
    figsize_base_height    : base figure height in inches.
    cmap                   : diverging colormap (default 'RdBu_r').
    vmax_pct               : percentile of |z-scores| used to clip colormap
                             (default 95).
    meta_cols              : meta column names to exclude from loci detection.
                             Defaults to META_COLS_DEFAULT.

    Returns
    -------
    matplotlib Figure
    """
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import pdist
    from matplotlib.colors import TwoSlopeNorm
    import matplotlib.gridspec as gridspec

    if meta_cols is None:
        meta_cols = META_COLS_DEFAULT

    sig_loci = list(sig_loci)

    # ── Load loci map and build genomic column order ──────────────────────────
    map_indexed = loci_map.set_index("blockbin")
    ordered_sig = [l for l in loci_map["blockbin"] if l in sig_loci]
    # Any sig_loci not in the map are appended at the end (unmapped)
    unmapped = [l for l in sig_loci if l not in map_indexed.index]
    if unmapped:
        print(f"  Warning: {len(unmapped)} loci not found in loci_map — "
              f"appended at end: {unmapped[:5]}{'...' if len(unmapped)>5 else ''}")
    col_order = ordered_sig + unmapped

    if not col_order:
        raise ValueError("No significant loci found in the loci map. "
                         "Check that sig_loci names match the 'blockbin' column.")

    print(f"  Column order: {len(col_order)} loci  "
          f"({len(ordered_sig)} mapped + {len(unmapped)} unmapped)")

    # Build per-column annotation arrays (chromosome, arm)
    chrom_labels = []
    arm_labels   = []
    for locus in col_order:
        if locus in map_indexed.index:
            chrom_labels.append(map_indexed.loc[locus, "chromosome_true"])
            arm_labels.append(map_indexed.loc[locus, "arm"])
        else:
            chrom_labels.append("unknown")
            arm_labels.append("unknown")

    # ── Chromosome colour palette (alternating for visual separation) ─────────
    unique_chroms = list(dict.fromkeys(chrom_labels))   # preserve genomic order
    # Alternating grey / coloured palette
    chrom_palette_base = [
        "#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66",
        "#77BEDB", "#F5A623", "#E8604C", "#9B59B6", "#2ECC71",
        "#C0392B", "#16A085",
    ]
    chrom_colors = {}
    for i, ch in enumerate(unique_chroms):
        chrom_colors[ch] = chrom_palette_base[i % len(chrom_palette_base)]
    chrom_colors["unknown"] = "#CCCCCC"

    arm_colors = {"p": "#AAAAAA", "q": "#555555", "unknown": "#EEEEEE"}

    # ── Classify sample groups into main vs extra ─────────────────────────────
    all_groups = sorted(df["sample_group"].unique())

    if extra_type_keywords:
        extra_types = [t for t in all_groups
                       if any(kw in t for kw in extra_type_keywords)]
    else:
        extra_types = []

    if main_types is None:
        main_types = [t for t in all_groups if t not in extra_types]

    print(f"  Main panel groups:  {main_types}")
    print(f"  Extra panel groups: {extra_types}")

    # ── Build per-patient mean matrix ─────────────────────────────────────────
    # Merge meta onto corrected values
    meta   = df[["cellID", "patient", "Malignant_status", "sample_group"]].copy()
    df_c   = meta.merge(
        loci_corrected[["cellID"] + col_order],
        on="cellID", how="inner",
    )

    def _build_matrix(group_list: list[str]) -> pd.DataFrame:
        """Return a DataFrame with one row per (patient, Malignant_status)."""
        rows = []
        for sg in group_list:
            sub = df_c[df_c["sample_group"] == sg]
            for pid in sub["patient"].unique():
                sub_p = sub[sub["patient"] == pid]
                for cls in sub_p["Malignant_status"].unique():
                    cells = sub_p[sub_p["Malignant_status"] == cls]
                    means = cells[col_order].mean()
                    rows.append({
                        "patient":        pid,
                        "Malignant_status": cls,
                        "sample_group":   sg,
                        **means.to_dict(),
                    })
        return pd.DataFrame(rows)

    main_mat  = _build_matrix(main_types)
    extra_mat = _build_matrix(extra_types) if extra_types else None
    has_extra = extra_mat is not None and len(extra_mat) > 0

    print(f"  Main matrix:  {main_mat.shape[0]} rows × {len(col_order)} loci")
    if has_extra:
        print(f"  Extra matrix: {extra_mat.shape[0]} rows × {len(col_order)} loci")

    # ── Z-score and cluster rows only ─────────────────────────────────────────
    def _zscore(data: np.ndarray) -> np.ndarray:
        mu = np.nanmean(data, axis=0, keepdims=True)
        sd = np.nanstd(data,  axis=0, keepdims=True)
        sd[sd == 0] = 1
        return (data - mu) / sd

    def _cluster_rows(mat: pd.DataFrame):
        data   = mat[col_order].values.astype(float)
        data_z = _zscore(data)
        clean  = np.nan_to_num(data_z)
        r_link = hierarchy.linkage(pdist(clean, "euclidean"), "average")
        r_ord  = hierarchy.leaves_list(r_link)
        return data_z, r_link, r_ord

    data_z, row_link, row_ord = _cluster_rows(main_mat)

    # Apply row ordering (columns stay as col_order — no column clustering)
    data_ordered = data_z[row_ord, :]
    mat_ordered  = main_mat.iloc[row_ord].reset_index(drop=True)

    row_labels = [
        f"{r['patient']}  {r['Malignant_status']}"
        for _, r in mat_ordered.iterrows()
    ]
    n_rows = data_ordered.shape[0]
    n_cols = data_ordered.shape[1]

    vmax = float(np.nanpercentile(np.abs(data_ordered), vmax_pct))
    vmax = max(vmax, 0.01)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    # ── Figure layout ─────────────────────────────────────────────────────────
    # Rows from top:
    #   [chrom annotation strip]
    #   [arm annotation strip]
    #   [main heatmap]
    # Row dendrogram sits to the left of the heatmap.
    # Extra panel (if any) below.

    fig_w = max(14, n_cols * figsize_width_per_locus)
    fig_h = figsize_base_height + (2.5 if has_extra else 0)
    fig   = plt.figure(figsize=(fig_w, fig_h))

    # Annotation strip height (in relative units)
    ann_strip_h = 0.018   # fraction of heatmap height per strip

    if has_extra:
        n_extra_rows = len(extra_mat)
        outer = gridspec.GridSpec(
            2, 1,
            height_ratios=[n_rows, max(n_extra_rows, 2)],
            hspace=0.45, left=0.01, right=0.82, top=0.94, bottom=0.08,
        )
    else:
        outer = gridspec.GridSpec(
            1, 1, left=0.01, right=0.82, top=0.94, bottom=0.08,
        )

    def _make_inner(subplot_spec, n_data_rows):
        """
        Three-row inner grid:
          0 — row dendrogram (left portion only — we fake it with a full-width ax)
          1 — chrom annotation strip
          2 — arm annotation strip
          3 — heatmap data
        We use height ratios to keep annotation strips thin.
        """
        dend_h = max(0.06, 0.5 / n_data_rows)
        return gridspec.GridSpecFromSubplotSpec(
            4, 2,
            subplot_spec=subplot_spec,
            height_ratios=[dend_h, ann_strip_h, ann_strip_h, 1],
            width_ratios=[0.04, 1],      # narrow left column for row dendrogram
            hspace=0.01, wspace=0.01,
        )

    inner_main = _make_inner(outer[0], n_rows)
    ax_row_dend  = fig.add_subplot(inner_main[3, 0])   # row dendrogram (left col, heatmap row)
    ax_heat      = fig.add_subplot(inner_main[3, 1])   # main heatmap
    ax_chrom     = fig.add_subplot(inner_main[1, 1])   # chromosome strip
    ax_arm       = fig.add_subplot(inner_main[2, 1])   # arm strip

    # Spacer axes (top-left corners — unused, turn off)
    for r in [0, 1, 2]:
        ax_sp = fig.add_subplot(inner_main[r, 0])
        ax_sp.axis("off")
    ax_top_right = fig.add_subplot(inner_main[0, 1])
    ax_top_right.axis("off")

    # Colorbar
    ax_cbar = fig.add_axes([0.845, 0.3, 0.012, 0.4])

    # ── Row dendrogram ────────────────────────────────────────────────────────
    hierarchy.dendrogram(
        row_link,
        ax=ax_row_dend,
        orientation="left",
        link_color_func=lambda k: "#555555",
        no_labels=True,
    )
    ax_row_dend.invert_yaxis()
    ax_row_dend.axis("off")

    # ── Chromosome annotation strip ───────────────────────────────────────────
    chrom_data = np.arange(n_cols).reshape(1, -1)   # dummy data, coloured by patches
    ax_chrom.imshow(
        np.zeros((1, n_cols)), aspect="auto", cmap="Greys",
        vmin=0, vmax=1, alpha=0,
    )
    for j, ch in enumerate(chrom_labels):
        ax_chrom.add_patch(mpatches.Rectangle(
            (j - 0.5, -0.5), 1, 1,
            color=chrom_colors.get(ch, "#CCCCCC"),
            lw=0, transform=ax_chrom.transData,
        ))

    # Chromosome name labels at midpoint of each chromosome block
    prev_ch = None
    ch_start = 0
    chrom_ticks = []
    for j, ch in enumerate(chrom_labels):
        if ch != prev_ch:
            if prev_ch is not None:
                mid = (ch_start + j - 1) / 2
                chrom_ticks.append((mid, prev_ch.replace("chr", "")))
            ch_start = j
            prev_ch = ch
    # Last chromosome
    mid = (ch_start + n_cols - 1) / 2
    chrom_ticks.append((mid, prev_ch.replace("chr", "")))

    ax_chrom.set_xticks([t[0] for t in chrom_ticks])
    ax_chrom.set_xticklabels([t[1] for t in chrom_ticks],
                              fontsize=4.5, rotation=0)
    ax_chrom.tick_params(length=0, pad=1)
    ax_chrom.set_xlim(-0.5, n_cols - 0.5)
    ax_chrom.set_yticks([])
    ax_chrom.set_ylabel("Chr", fontsize=4.5, rotation=0, labelpad=18, va="center")
    for sp in ax_chrom.spines.values():
        sp.set_visible(False)

    # ── Arm annotation strip ──────────────────────────────────────────────────
    ax_arm.imshow(
        np.zeros((1, n_cols)), aspect="auto", cmap="Greys",
        vmin=0, vmax=1, alpha=0,
    )
    for j, arm in enumerate(arm_labels):
        ax_arm.add_patch(mpatches.Rectangle(
            (j - 0.5, -0.5), 1, 1,
            color=arm_colors.get(arm, "#EEEEEE"),
            lw=0, transform=ax_arm.transData,
        ))
    ax_arm.set_xticks([])
    ax_arm.set_yticks([])
    ax_arm.set_xlim(-0.5, n_cols - 0.5)
    ax_arm.set_ylabel("Arm", fontsize=4.5, rotation=0, labelpad=18, va="center")
    for sp in ax_arm.spines.values():
        sp.set_visible(False)

    # ── Main heatmap ──────────────────────────────────────────────────────────
    im = ax_heat.imshow(
        data_ordered, aspect="auto", cmap=cmap,
        norm=norm, interpolation="nearest",
    )
    ax_heat.set_xticks([])
    ax_heat.set_yticks(range(n_rows))
    ax_heat.set_yticklabels(row_labels, fontsize=6.5)
    ax_heat.tick_params(length=0)
    for sp in ax_heat.spines.values():
        sp.set_visible(False)
    ax_heat.set_title(
        " & ".join(main_types), fontsize=7, fontweight="bold",
        loc="left", pad=3,
    )

    # ── Colorbar ──────────────────────────────────────────────────────────────
    cb = fig.colorbar(im, cax=ax_cbar, orientation="vertical")
    cb.ax.tick_params(labelsize=5)
    cb.set_label("Z-score", fontsize=5.5)

    # ── Row annotation strips (class + group) drawn as patches ────────────────
    # Colour maps for class and group
    CLASS_PAL = {
        "Normal":  "#74AFE9",
        "Malig.":  "#E57061",
    }
    TYPE_PAL = {g: GROUP_PALETTE_DEFAULT[i % len(GROUP_PALETTE_DEFAULT)]
                for i, g in enumerate(all_groups)}
    FALLBACK  = "#AAAAAA"

    ann_step = n_cols * 0.015
    ann_x0   = n_cols + 0.3

    def _draw_row_annotations(heat_ax, mat_ord):
        for i, (_, row) in enumerate(mat_ord.iterrows()):
            # Malignant_status strip
            heat_ax.add_patch(mpatches.Rectangle(
                (ann_x0,              i - 0.5), ann_step, 1,
                color=CLASS_PAL.get(row["Malignant_status"], FALLBACK),
                lw=0, transform=heat_ax.transData, clip_on=False,
            ))
            # sample_group strip
            heat_ax.add_patch(mpatches.Rectangle(
                (ann_x0 + ann_step * 1.4, i - 0.5), ann_step, 1,
                color=TYPE_PAL.get(row["sample_group"], FALLBACK),
                lw=0, transform=heat_ax.transData, clip_on=False,
            ))
        # Column headers above strips
        heat_ax.text(
            ann_x0 + ann_step * 0.5, -0.7, "status",
            ha="center", va="bottom", fontsize=4.5, rotation=45,
            color="#555", transform=heat_ax.transData, clip_on=False,
        )
        heat_ax.text(
            ann_x0 + ann_step * 1.9, -0.7, "group",
            ha="center", va="bottom", fontsize=4.5, rotation=45,
            color="#555", transform=heat_ax.transData, clip_on=False,
        )

    _draw_row_annotations(ax_heat, mat_ordered)

    # ── Extra panel ───────────────────────────────────────────────────────────
    if has_extra:
        inner_extra  = _make_inner(outer[1], len(extra_mat))
        ax_ex_row_d  = fig.add_subplot(inner_extra[3, 0])
        ax_ex_heat   = fig.add_subplot(inner_extra[3, 1])
        ax_ex_chrom  = fig.add_subplot(inner_extra[1, 1])
        ax_ex_arm    = fig.add_subplot(inner_extra[2, 1])

        for r in [0, 1, 2]:
            fig.add_subplot(inner_extra[r, 0]).axis("off")
        fig.add_subplot(inner_extra[0, 1]).axis("off")

        # Use same column order as main panel; cluster extra rows independently
        ex_data   = extra_mat[col_order].values.astype(float)
        ex_z      = _zscore(ex_data)
        ex_clean  = np.nan_to_num(ex_z)
        ex_r_link = hierarchy.linkage(pdist(ex_clean, "euclidean"), "average")
        ex_r_ord  = hierarchy.leaves_list(ex_r_link)

        ex_ordered = ex_z[ex_r_ord, :]
        ex_mat_ord = extra_mat.iloc[ex_r_ord].reset_index(drop=True)
        ex_labels  = [
            f"{r['patient']}  {r['Malignant_status']}"
            for _, r in ex_mat_ord.iterrows()
        ]
        n_ex = len(ex_labels)

        # Row dendrogram
        hierarchy.dendrogram(
            ex_r_link,
            ax=ax_ex_row_d,
            orientation="left",
            link_color_func=lambda k: "#555555",
            no_labels=True,
        )
        ax_ex_row_d.invert_yaxis()
        ax_ex_row_d.axis("off")

        # Repeat annotation strips (share same col_order)
        for ax_strip, labels, pal, ylabel in [
            (ax_ex_chrom, chrom_labels, chrom_colors, "Chr"),
            (ax_ex_arm,   arm_labels,   arm_colors,   "Arm"),
        ]:
            ax_strip.imshow(
                np.zeros((1, n_cols)), aspect="auto", cmap="Greys",
                vmin=0, vmax=1, alpha=0,
            )
            for j, val in enumerate(labels):
                ax_strip.add_patch(mpatches.Rectangle(
                    (j - 0.5, -0.5), 1, 1,
                    color=pal.get(val, "#CCCCCC"),
                    lw=0, transform=ax_strip.transData,
                ))
            ax_strip.set_xticks([])
            ax_strip.set_yticks([])
            ax_strip.set_xlim(-0.5, n_cols - 0.5)
            ax_strip.set_ylabel(ylabel, fontsize=4.5, rotation=0,
                                labelpad=18, va="center")
            for sp in ax_strip.spines.values():
                sp.set_visible(False)

        # Chromosome ticks on extra panel too
        ax_ex_chrom.set_xticks([t[0] for t in chrom_ticks])
        ax_ex_chrom.set_xticklabels([t[1] for t in chrom_ticks],
                                     fontsize=4.5, rotation=0)
        ax_ex_chrom.tick_params(length=0, pad=1)

        ax_ex_heat.imshow(
            ex_ordered, aspect="auto", cmap=cmap,
            norm=norm, interpolation="nearest",
        )
        ax_ex_heat.set_xticks([])
        ax_ex_heat.set_yticks(range(n_ex))
        ax_ex_heat.set_yticklabels(ex_labels, fontsize=6.5)
        ax_ex_heat.tick_params(length=0)
        for sp in ax_ex_heat.spines.values():
            sp.set_visible(False)
        ax_ex_heat.set_title(
            ", ".join(extra_types), fontsize=7, fontweight="bold",
            loc="left", pad=3,
        )
        _draw_row_annotations(ax_ex_heat, ex_mat_ord)

    # ── Legend ────────────────────────────────────────────────────────────────
    unique_classes = mat_ordered["Malignant_status"].unique().tolist()
    unique_groups  = mat_ordered["sample_group"].unique().tolist()
    if has_extra:
        unique_classes += extra_mat["Malignant_status"].unique().tolist()
        unique_groups  += extra_mat["sample_group"].unique().tolist()

    handles = (
        [mpatches.Patch(color=CLASS_PAL.get(c, FALLBACK), label=f"status: {c}")
         for c in sorted(set(unique_classes))]
        + [mpatches.Patch(color=TYPE_PAL.get(g, FALLBACK), label=f"group: {g}")
           for g in sorted(set(unique_groups))]
        + [mpatches.Patch(color=arm_colors["p"], label="arm: p (light)"),
           mpatches.Patch(color=arm_colors["q"], label="arm: q (dark)")]
    )
    fig.legend(
        handles=handles, loc="center right", fontsize=6,
        bbox_to_anchor=(1.0, 0.5), ncol=1,
        title="Annotation", title_fontsize=6, frameon=False,
    )

    fig.suptitle(
        f"Significant loci — {len(col_order)} loci  |  "
        f"z-scored per locus  |  rows clustered, columns in genomic order",
        fontsize=8, fontweight="bold",
    )

    if out_path:
        fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
        print(f"Saved: {out_path}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Convenience wrapper — run all steps end-to-end
# ══════════════════════════════════════════════════════════════════════════════

def run_dna_loci_analysis(
    df: pd.DataFrame,
    method: str = "zscore",
    alpha: float = 0.05,
    normal_cell_types: Optional[list[str]] = None,
    meta_cols: Optional[list[str]] = None,
    test_between_groups: bool = False,
    out_dir: Optional[str] = None,
) -> dict:
    """
    End-to-end analysis pipeline for DNA loci data.

    Steps
    -----
    1. Correct loci values by per-patient normal cells.
    2. Run significance tests on malignant cells.
    3. Summarise significant loci.
    4. (optional) Save CSV results if out_dir is provided.

    Parameters
    ----------
    df                  : cell × (meta + loci) DataFrame.
    method              : 'zscore' or 'quantile'.
    alpha               : FDR significance threshold.
    normal_cell_types   : predicted.celltype values to use as normal baseline.
    meta_cols           : meta column names. Defaults to META_COLS_DEFAULT.
    test_between_groups : also run pairwise between-group tests.
    out_dir             : directory to save result CSVs. None = no saving.

    Returns
    -------
    dict with keys:
        'corrected'  : corrected DataFrame
        'results'    : dict from test_feature_significance_loci
        'summaries'  : dict from summarise_significant_features_loci
    """
    import os

    if meta_cols is None:
        meta_cols = META_COLS_DEFAULT

    cancer_groups = (
        df.loc[df["Malignant_status"] != "Normal", "sample_group"]
        .unique().tolist()
    )
    print(f"Cancer groups: {cancer_groups}")

    print("\nStep 1: correcting by normals...")
    corrected = correct_by_normals_loci(
        df,
        method=method,
        normal_cell_types=normal_cell_types,
        meta_cols=meta_cols,
    )

    print("\nStep 2: testing feature significance...")
    results = test_feature_significance_loci(
        df_original         = df,
        loci_corrected      = corrected,
        cancer_groups       = cancer_groups,
        method              = method,
        alpha               = alpha,
        test_vs_null        = True,
        test_between_groups = test_between_groups,
        meta_cols           = meta_cols,
    )

    print("\nStep 3: summarising...")
    summaries = summarise_significant_features_loci(results, cancer_groups)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        if "vs_null" in results:
            path = os.path.join(out_dir, f"loci_significance_vs_null_{method}.csv")
            results["vs_null"].to_csv(path, index=False)
            print(f"  Saved: {path}")
        if "between_groups" in results:
            path = os.path.join(out_dir, f"loci_significance_between_groups_{method}.csv")
            results["between_groups"].to_csv(path, index=False)
            print(f"  Saved: {path}")
        if "vs_null" in summaries:
            path = os.path.join(out_dir, f"loci_summary_vs_null_{method}.csv")
            summaries["vs_null"].to_csv(path, index=False)
            print(f"  Saved: {path}")

    return {
        "corrected":  corrected,
        "results":    results,
        "summaries":  summaries,
    }