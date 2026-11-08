"""
feature_significance.py
------------------------
Two-step analysis of corrected single-cell features:

Step 1 — Feature significance testing
    For each cancer group and each feature, test whether the per-sample means
    are significantly different from the correction null (0 for MAD, 0.5 for
    quantile) using a one-sample t-test across samples. Summarise how many
    significant features belong to each group and whether they are elevated
    or depleted relative to the null.

Step 2 — Combined score
    For significant features, compute a per-cell combined score and plot its
    distribution in: normal cells, all malignant cells, and malignant cells
    split by cancer group.

Usage:
    python feature_significance.py
    # or import individual functions into a notebook
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm

from scipy import stats
from scipy.stats import gaussian_kde
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

import sys
from typing import Optional, Union

sys.path.insert(0, '/mnt/user-data/outputs')
from normalize_by_normals import correct_by_normals, identify_feature_groups

# ── Illustrator-compatible export settings ────────────────────────────────────
# fonttype 42  → TrueType fonts embedded as whole glyphs (words editable in AI)
# svg.fonttype "none" → text stays as <text> elements, not converted to paths
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["ps.fonttype"]  = 42

# ── Palette ───────────────────────────────────────────────────────────────────
C_NORMAL      = "#4C9BE8"
GROUP_PALETTE = ["#E8604C", "#F5A623", "#7ED321", "#9B59B6", "#2ECC71"]

plt.rcParams.update({
    # Arial is the safest font for Illustrator round-trips
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
    # Disable path simplification — prevents invisible/merged path artefacts in AI
    "path.simplify":     False,
    # Transparent backgrounds avoid hidden white rectangles in AI
    "figure.facecolor":  "none",
    "axes.facecolor":    "none",
})

# ── default palettes — extend as needed ──────────────────────────────────────
CLASS_PAL = {
    'normal':       '#4878CF',
    'LGG_MT':       '#724C6D',    # changed
    'LGG_noMT':     '#E69E84',    # changed
    'Transformed':  '#9B59B6',    # unchanged
    'GBM':          '#C0392B',
}
TYPE_PAL = {
    'LGG_MT':       '#724C6D',    # changed
    'LGG_noMT':     '#E69E84',    # changed
    'Transformed':  '#9B59B6',    # unchanged
    'GBM':          '#C0392B',
}
FALLBACK_COLOR = '#AAAAAA'

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Feature significance testing
# ══════════════════════════════════════════════════════════════════════════════

def test_feature_significance(
    df_original: pd.DataFrame,
    loci_corrected: pd.DataFrame,
    contacts_corrected: pd.DataFrame,
    feat_groups: dict,
    method: str,
    cancer_groups: list[str],
    alpha: float = 0.05,
    min_samples: int = 2,
    test_vs_null: bool = True,
    test_between_groups: bool = False,
    group_col: str = 'sample_group',
    id_col: str = 'ROI'
) -> dict[str, pd.DataFrame]:
    """
    Two complementary significance tests on corrected per-sample means of
    malignant cells.

    IMPORTANT: df_original (and the corrected dataframes) must be pre-filtered
    before calling this function — pass only the malignant cells from the
    cancer_groups of interest plus the intended normal cell types from those
    same samples. The easiest way is to use the filtering block in main().
    This keeps the function simple and ensures correct_by_normals was also run
    on the right subset.

    Test 1 — vs null  (test_vs_null=True)
    ----------------------------------------
    For each cancer group × feature, one-sample t-test of the per-sample means
    against the correction null (0 for zscore, 0.5 for quantile).
    Answers: "Is this group's signal different from baseline?"

    Test 2 — between groups  (test_between_groups=True)
    --------------------------------------------------------
    For every pair of cancer groups × feature, two-sample Welch t-test comparing
    the per-sample means of the two groups against each other.
    Answers: "Is this feature different *between* these two groups?"

    Parameters
    ----------
    df_original         : pre-filtered dataframe (relevant cancer groups +
                          intended normal cell types only)
    loci_corrected      : corrected loci_frac dataframe (must contain 'cell_id')
    contacts_corrected  : corrected contacts_corr dataframe
    feat_groups         : {'loci_frac': [...], 'contacts_corr': [...]}
    method              : 'zscore' or 'quantile' — determines the null value
    cancer_groups       : list of sample_group values to test
    alpha               : FDR threshold after BH correction
    min_samples         : minimum samples a group must have to be tested
    test_vs_null        : run Test 1 (default True)
    test_between_groups : run Test 2 (default False)

    Returns
    -------
    dict with keys 'vs_null' and/or 'between_groups' (pd.DataFrames).
    """
    assert test_vs_null or test_between_groups, \
        "At least one of test_vs_null or test_between_groups must be True."

    null_value = 0.5 if method == "quantile" else 0.0

    all_features = feat_groups["loci_frac"] + feat_groups["contacts_corr"]
    feat_to_type = {f: "loci_frac" for f in feat_groups["loci_frac"]}
    feat_to_type.update({f: "contacts_corr" for f in feat_groups["contacts_corr"]})

    # ── Build corrected dataframe joined on cell_id ───────────────────────────
    meta_cols = [c for c in df_original.columns if c not in all_features]
    meta = df_original[meta_cols].copy()

    loci_feats     = [f for f in feat_groups["loci_frac"]     if f in loci_corrected.columns]
    contacts_feats = [f for f in feat_groups["contacts_corr"] if f in contacts_corrected.columns]

    corrected_feats = pd.concat([
        loci_corrected[["cell_id"] + loci_feats],
        contacts_corrected[contacts_feats],
    ], axis=1)

    df_corr = meta.merge(corrected_feats, on="cell_id", how="inner")

    # df_original is already pre-filtered (only the relevant cancer groups and
    # specified normal cell types). Simply separate malignant from normal here.
    malignant_df = df_corr[df_corr["class"] != "normal"].copy()

    # ── Pre-compute per-sample means for every group ──────────────────────────
    # Dict: group -> DataFrame of shape (n_samples, n_features), index = sample_id
    group_sample_means: dict[str, pd.DataFrame] = {}
    valid_groups: list[str] = []

    for group in cancer_groups:
        group_df = malignant_df[malignant_df[group_col] == group]
        samples  = group_df[id_col].unique()
        if len(samples) < min_samples:
            print(f"  Skipping '{group}': only {len(samples)} sample(s) "
                  f"(need ≥ {min_samples})")
            continue
        group_sample_means[group] = (
            group_df.groupby(id_col)[all_features].mean()
        )
        valid_groups.append(group)

    output: dict[str, pd.DataFrame] = {}

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 1 — each group vs the correction null
    # ══════════════════════════════════════════════════════════════════════════
    if test_vs_null:
        print(f"\nTest 1: each group vs null ({null_value}) "
              f"[one-sample t-test, BH correction per group]")

        records_null = []

        for group in cancer_groups:
            n_samples_group = len(
                malignant_df[malignant_df[group_col] == group][id_col].unique()
            )
            if group not in valid_groups:
                # Still record so all groups appear in output
                for feat in all_features:
                    records_null.append({
                        "feature":              feat,
                        "feat_type":            feat_to_type[feat],
                        "cancer_group":         group,
                        "n_samples":            n_samples_group,
                        "mean_of_sample_means": np.nan,
                        "std_of_sample_means":  np.nan,
                        "t_stat":               np.nan,
                        "p_value":              np.nan,
                        "p_adj":                np.nan,
                        "significant":          False,
                        "direction":            np.nan,
                    })
                continue

            sample_means = group_sample_means[group]

            for feat in all_features:
                col_means = sample_means[feat].dropna().values
                if len(col_means) < min_samples:
                    t_stat, p_val = np.nan, np.nan
                else:
                    t_stat, p_val = stats.ttest_1samp(col_means, popmean=null_value)

                records_null.append({
                    "feature":              feat,
                    "feat_type":            feat_to_type[feat],
                    "cancer_group":         group,
                    "n_samples":            len(sample_means),
                    "mean_of_sample_means": float(np.nanmean(col_means)),
                    "std_of_sample_means":  float(np.nanstd(col_means)),
                    "t_stat":               float(t_stat) if not np.isnan(t_stat) else np.nan,
                    "p_value":              float(p_val)  if not np.isnan(p_val)  else np.nan,
                    "p_adj":                np.nan,
                    "significant":          False,
                    "direction":            np.nan,
                })

        res_null = pd.DataFrame(records_null)

        # BH correction per cancer group
        for group in cancer_groups:
            mask   = res_null["cancer_group"] == group
            p_vals = res_null.loc[mask, "p_value"].values
            valid  = ~np.isnan(p_vals)
            p_adj  = np.full(len(p_vals), np.nan)
            if valid.sum() > 0:
                p_adj[valid] = (
                    stats.false_discovery_control(p_vals[valid], method="bh", axis=0)
                    if hasattr(stats, "false_discovery_control")
                    else _bh_correction(p_vals[valid])
                )
            res_null.loc[mask, "p_adj"] = p_adj

        res_null["significant"] = res_null["p_adj"] < alpha
        res_null["direction"] = np.where(
            res_null["mean_of_sample_means"] > null_value, "elevated", "depleted"
        )
        res_null.loc[~res_null["significant"], "direction"] = np.nan

        n_sig = res_null["significant"].sum()
        print(f"  {n_sig} significant feature × group combinations "
              f"(FDR < {alpha})")

        output["vs_null"] = res_null

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 2 — pairwise between groups, restricted to vs_null significant features
    # ══════════════════════════════════════════════════════════════════════════
    if test_between_groups:
        from itertools import combinations

        # Only test features that are significant in at least one group in Test 1.
        # This ensures:
        #   (a) we only compare features that already deviate from normal,
        #   (b) direction labels from Test 1 can be carried forward unambiguously.
        # If Test 1 was not run, fall back to all features.
        if "vs_null" in output:
            vs_null_sig_feats = set(
                output["vs_null"].loc[output["vs_null"]["significant"], "feature"]
            )
            between_feats = [f for f in all_features if f in vs_null_sig_feats]
            print(f"\nTest 2: restricted to {len(between_feats)} features significant "
                  f"in Test 1 (out of {len(all_features)} total)")
        else:
            between_feats = all_features
            print(f"\nTest 2: using all {len(between_feats)} features "
                  f"(Test 1 was not run)")

        pairs = list(combinations(valid_groups, 2))
        print(f"  {len(pairs)} pair(s), two-sample Welch t-test, BH correction per pair")

        # Build a lookup: feature -> direction from Test 1 (majority direction
        # across groups if both groups significant, else whichever is significant)
        if "vs_null" in output:
            sig_null = output["vs_null"][output["vs_null"]["significant"]]
            # For each feature take the first non-null direction
            feat_direction = (
                sig_null.dropna(subset=["direction"])
                .groupby("feature")["direction"]
                .first()
                .to_dict()
            )
        else:
            feat_direction = {}

        records_pairs = []

        for group_a, group_b in pairs:
            means_a = group_sample_means[group_a]
            means_b = group_sample_means[group_b]

            for feat in between_feats:
                vals_a = means_a[feat].dropna().values if feat in means_a.columns else np.array([])
                vals_b = means_b[feat].dropna().values if feat in means_b.columns else np.array([])

                if len(vals_a) < min_samples or len(vals_b) < min_samples:
                    t_stat, p_val = np.nan, np.nan
                else:
                    t_stat, p_val = stats.ttest_ind(vals_a, vals_b, equal_var=False)

                records_pairs.append({
                    "feature":      feat,
                    "feat_type":    feat_to_type[feat],
                    "group_a":      group_a,
                    "group_b":      group_b,
                    "n_samples_a":  len(vals_a),
                    "n_samples_b":  len(vals_b),
                    "mean_a":       float(np.nanmean(vals_a)) if len(vals_a) else np.nan,
                    "mean_b":       float(np.nanmean(vals_b)) if len(vals_b) else np.nan,
                    "t_stat":       float(t_stat) if not np.isnan(t_stat) else np.nan,
                    "p_value":      float(p_val)  if not np.isnan(p_val)  else np.nan,
                    "p_adj":        np.nan,
                    "significant":  False,
                    "higher_group": np.nan,
                    # Carry direction forward from Test 1 so scoring uses the
                    # same sign convention as the vs_null score
                    "direction":    feat_direction.get(feat, np.nan),
                })

        res_pairs = pd.DataFrame(records_pairs)

        # BH correction per pair
        for group_a, group_b in pairs:
            mask   = (res_pairs["group_a"] == group_a) & (res_pairs["group_b"] == group_b)
            p_vals = res_pairs.loc[mask, "p_value"].values
            valid  = ~np.isnan(p_vals)
            p_adj  = np.full(len(p_vals), np.nan)
            if valid.sum() > 0:
                p_adj[valid] = (
                    stats.false_discovery_control(p_vals[valid], method="bh", axis=0)
                    if hasattr(stats, "false_discovery_control")
                    else _bh_correction(p_vals[valid])
                )
            res_pairs.loc[mask, "p_adj"] = p_adj

        res_pairs["significant"] = res_pairs["p_adj"] < alpha

        res_pairs["higher_group"] = np.where(
            res_pairs["mean_a"] >= res_pairs["mean_b"],
            res_pairs["group_a"],
            res_pairs["group_b"],
        )
        res_pairs.loc[~res_pairs["significant"], "higher_group"] = np.nan

        n_sig = res_pairs["significant"].sum()
        print(f"  {n_sig} significant feature × pair combinations "
              f"(FDR < {alpha})")

        output["between_groups"] = res_pairs

    return output


def _bh_correction(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction (fallback for older scipy)."""
    n    = len(p_values)
    order = np.argsort(p_values)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    p_adj = np.minimum(1.0, p_values * n / ranks)
    # Enforce monotonicity (from largest to smallest)
    for i in range(n - 2, -1, -1):
        p_adj[order[i]] = min(p_adj[order[i]], p_adj[order[i + 1]])
    return p_adj


def summarise_significant_features(
    results: dict[str, pd.DataFrame] | pd.DataFrame,
    cancer_groups: list[str],
) -> dict[str, pd.DataFrame]:
    """
    Print and return human-readable summaries of significant features.

    Accepts either the full dict returned by test_feature_significance, or
    (for backwards compatibility) a bare vs_null DataFrame.

    Logic
    -----
    Test 1 (vs_null): features significantly different from the correction null
    in each cancer group. Direction (elevated / depleted) is assigned here.

    Test 2 (between_groups): subset of Test 1 features that are ALSO
    significantly different between cancer groups. Direction is inherited
    from Test 1. This section reports how many Test-1 features are
    further differentiated between groups.

    Returns
    -------
    dict with up to three keys:
        "vs_null"        -> per-group summary of Test 1 features
        "between_groups" -> per-pair summary of features sig in both tests
        "joint"          -> per-feature table showing Test 1 direction +
                            which pairs it is also significant between
    """
    if isinstance(results, pd.DataFrame):
        results = {"vs_null": results}

    summaries: dict[str, pd.DataFrame] = {}

    # ── Test 1 summary ────────────────────────────────────────────────────────
    if "vs_null" in results:
        res = results["vs_null"]
        sig = res[res["significant"]].copy()

        summary = (
            sig.groupby(["cancer_group", "feat_type", "direction"])
            .size()
            .reset_index(name="n_features")
        )
        totals = (
            sig.groupby("cancer_group")
            .size()
            .reset_index(name="n_features")
            .assign(feat_type="TOTAL", direction="—")
        )
        summary = pd.concat([summary, totals], ignore_index=True)
        summary = summary.sort_values(["cancer_group", "feat_type", "direction"])

        print("\n── Test 1: significant features vs null ──────────────────────────")
        print(f"{'Cancer group':<25} {'Feature type':<16} {'Direction':<12} {'N features':>10}")
        print("─" * 65)
        for _, row in summary.iterrows():
            print(f"{row['cancer_group']:<25} {row['feat_type']:<16} "
                  f"{row['direction']:<12} {row['n_features']:>10}")

        summaries["vs_null"] = summary

    # ── Test 2 summary ────────────────────────────────────────────────────────
    if "between_groups" in results:
        res_btw = results["between_groups"]
        sig_btw = res_btw[res_btw["significant"]].copy()

        pairs = sig_btw[["group_a","group_b"]].drop_duplicates().values.tolist()

        summary = (
            sig_btw.groupby(["group_a", "group_b", "feat_type", "higher_group"])
            .size()
            .reset_index(name="n_features")
        )
        totals = (
            sig_btw.groupby(["group_a", "group_b"])
            .size()
            .reset_index(name="n_features")
            .assign(feat_type="TOTAL", higher_group="—")
        )
        summary = pd.concat([summary, totals], ignore_index=True)
        summary = summary.sort_values(["group_a", "group_b", "feat_type"])

        print("\n── Test 2: features sig between groups (subset of Test 1) ────────")
        print(f"{'Group A':<20} {'Group B':<20} {'Feat type':<16} {'Higher':<20} {'N':>6}")
        print("─" * 85)
        for _, row in summary.iterrows():
            print(f"{row['group_a']:<20} {row['group_b']:<20} {row['feat_type']:<16} "
                  f"{str(row['higher_group']):<20} {row['n_features']:>6}")

        summaries["between_groups"] = summary

    # ── Joint summary — one row per feature ───────────────────────────────────
    # Combines Test 1 direction with Test 2 between-group significance.
    if "vs_null" in results and "between_groups" in results:
        res_null = results["vs_null"]
        res_btw  = results["between_groups"]

        # All features significant in at least one group in Test 1
        sig_null = res_null[res_null["significant"]].copy()

        # For each feature: collect which cancer groups it is significant in,
        # its direction (should be consistent), and whether it is also
        # significant between groups.
        feat_records = []
        for feat, grp in sig_null.groupby("feature"):
            sig_groups   = grp["cancer_group"].tolist()
            directions   = grp["direction"].dropna().unique().tolist()
            direction    = directions[0] if len(directions) == 1 else "mixed"
            feat_type    = grp["feat_type"].iloc[0]

            # Which pairs is this feature also significant between?
            btw_sig_pairs = []
            if feat in res_btw["feature"].values:
                feat_btw = res_btw[
                    (res_btw["feature"] == feat) & res_btw["significant"]
                ]
                for _, row in feat_btw.iterrows():
                    btw_sig_pairs.append(
                        f"{row['group_a']} > {row['higher_group']}"
                        if row["higher_group"] == row["group_a"]
                        else f"{row['group_b']} > {row['higher_group']}"
                    )
                    # Simpler: just record "A vs B (higher=X)"
                    btw_sig_pairs[-1] = (
                        f"{row['group_a']} vs {row['group_b']} "
                        f"(higher: {row['higher_group']})"
                    )

            feat_records.append({
                "feature":             feat,
                "feat_type":           feat_type,
                "direction":           direction,
                "sig_in_groups":       ", ".join(sig_groups),
                "n_sig_groups":        len(sig_groups),
                "sig_between_pairs":   "; ".join(btw_sig_pairs) if btw_sig_pairs else "—",
                "n_sig_pairs":         len(btw_sig_pairs),
            })

        joint = pd.DataFrame(feat_records).sort_values(
            ["n_sig_pairs", "n_sig_groups"], ascending=False
        )

        n_both = (joint["n_sig_pairs"] > 0).sum()
        print(f"\n── Joint: {len(joint)} features sig vs null, "
              f"{n_both} also sig between groups ──────────")
        print(f"{'Feature':<45} {'Type':<14} {'Dir':<10} {'Groups':>8} {'Pairs':>7}")
        print("─" * 90)
        for _, row in joint.head(20).iterrows():
            print(f"{row['feature']:<45} {row['feat_type']:<14} {row['direction']:<10} "
                  f"{row['n_sig_groups']:>8} {row['n_sig_pairs']:>7}")
        if len(joint) > 20:
            print(f"  ... {len(joint)-20} more features (see summaries['joint'])")

        summaries["joint"] = joint

    return summaries


def plot_top_features(
    results: pd.DataFrame,
    df_original: pd.DataFrame,
    loci_corrected: pd.DataFrame,
    contacts_corrected: pd.DataFrame,
    feat_groups: dict,
    cancer_groups: list[str],
    method: str,
    n_top: int = 5,
    out_path: str = None,
) -> plt.Figure:
    """
    For each cancer group, plot the per-sample means of the top N significant
    features, ranked by absolute t-statistic.

    Each feature gets one panel. Within each panel, each dot is one sample's
    mean corrected value in its malignant cells, coloured by cancer group.
    The null value (0 for mad/zscore, 0.5 for quantile) is shown as a
    dashed reference line.

    Parameters
    ----------
    results         : output of test_feature_significance()
    df_original     : original (pre-correction) dataframe, for metadata
    loci_corrected  : corrected loci_frac dataframe
    contacts_corrected : corrected contacts_corr dataframe
    feat_groups     : output of identify_feature_groups()
    cancer_groups   : list of sample_group values
    method          : 'mad', 'zscore', or 'quantile'
    n_top           : number of top features to plot per cancer group
    out_path        : path to save the PNG
    """
    null_value = 0.5 if method == "quantile" else 0.0
    group_colors = {g: GROUP_PALETTE[i % len(GROUP_PALETTE)]
                    for i, g in enumerate(cancer_groups)}

    loci_feats     = [f for f in feat_groups["loci_frac"]     if f in loci_corrected.columns]
    contacts_feats = [f for f in feat_groups["contacts_corr"] if f in contacts_corrected.columns]

    corrected_vals = pd.concat([
        loci_corrected[["cell_id"] + loci_feats],
        contacts_corrected[contacts_feats],
    ], axis=1)

    meta = df_original[["cell_id", "ROI", "state", "class"]]
    df_corr = meta.merge(corrected_vals, on="cell_id", how="inner")
    malignant_df = df_corr[df_corr["class"] != "normal"]

    # Pick top N features per group by |t_stat|, among significant ones.
    # Normalise between_groups schema (group_a/group_b) to vs_null schema
    # (cancer_group) so the rest of the function works uniformly.
    sig = results[results["significant"]].copy()
    sig["abs_t"] = sig["t_stat"].abs()

    if "cancer_group" not in sig.columns:
        # between_groups schema: create a row per group per feature
        rows = []
        for _, r in sig.iterrows():
            for grp in [r["group_a"], r["group_b"]]:
                if grp in cancer_groups:
                    rows.append({**r.to_dict(), "cancer_group": grp})
        sig = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=list(sig.columns) + ["cancer_group"])

    top_feats_per_group = {}
    for group in cancer_groups:
        grp_sig = sig[sig["cancer_group"] == group].nlargest(n_top, "abs_t")
        top_feats_per_group[group] = grp_sig["feature"].tolist()

    # Union of all top features (may overlap between groups)
    all_top = []
    seen = set()
    for group in cancer_groups:
        for f in top_feats_per_group[group]:
            if f not in seen:
                all_top.append(f)
                seen.add(f)

    if not all_top:
        print("  No significant features to plot.")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No significant features found",
                ha="center", va="center", transform=ax.transAxes)
        if out_path:
            fig.savefig(out_path, bbox_inches="tight", transparent=True, dpi=300)
        return fig

    n_panels = len(all_top)
    ncols    = min(n_panels, 3)
    nrows    = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 3.5 * nrows))
    fig.patch.set_visible(False)
    axes_flat = np.array(axes).flatten() if n_panels > 1 else [axes]

    # Pre-compute per-sample normal cell means for reference line per feature
    normal_corr_df = df_corr[df_corr["class"] == "normal"]
    # x-position for normal column: one slot to the left of cancer groups
    normal_x_pos = -1

    for ax_i, feat in enumerate(all_top):
        ax = axes_flat[ax_i]

        # ── Normal cells: per-sample means shown as a reference column ────────
        if feat in normal_corr_df.columns:
            normal_sample_means = normal_corr_df.groupby("ROI")[feat].mean()
            rng    = np.random.default_rng(0)
            jitter = rng.uniform(-0.15, 0.15, len(normal_sample_means))
            ax.scatter(
                np.full(len(normal_sample_means), normal_x_pos) + jitter,
                normal_sample_means.values,
                color=C_NORMAL, s=50, alpha=0.75, zorder=3,
                label="Normal" if ax_i == 0 else "_nolegend_",
            )
            if len(normal_sample_means) > 0:
                nm = normal_sample_means.mean()
                ns = normal_sample_means.std() if len(normal_sample_means) > 1 else 0
                ax.errorbar(normal_x_pos, nm, yerr=ns,
                            fmt="D", color=C_NORMAL, markersize=7,
                            capsize=4, lw=2, zorder=4)

        # ── Cancer groups ─────────────────────────────────────────────────────
        for g_i, group in enumerate(cancer_groups):
            group_mask = malignant_df["state"] == group
            group_data = malignant_df[group_mask]

            sample_means = (
                group_data.groupby("ROI")[feat].mean()
                if feat in group_data.columns else pd.Series(dtype=float)
            )

            color = group_colors[group]
            x_pos = g_i

            rng    = np.random.default_rng(42 + g_i)
            jitter = rng.uniform(-0.15, 0.15, len(sample_means))
            ax.scatter(
                np.full(len(sample_means), x_pos) + jitter,
                sample_means.values,
                color=color, s=60, alpha=0.85, zorder=3,
                label=group if ax_i == 0 else "_nolegend_",
            )

            if len(sample_means) > 0:
                m = sample_means.mean()
                s = sample_means.std() if len(sample_means) > 1 else 0
                ax.errorbar(x_pos, m, yerr=s,
                            fmt="D", color=color, markersize=7,
                            capsize=4, lw=2, zorder=4)

            # Annotate p_adj for this group x feature
            row = sig[
                (sig["feature"] == feat) &
                (sig["cancer_group"] == group)
            ]
            if not row.empty and row["significant"].values[0]:
                p = row["p_adj"].values[0]
                stars = "***" if p < 0.001 else "**" if p < 0.01 else "*"
                ax.text(x_pos, ax.get_ylim()[1],
                        stars, ha="center", va="bottom",
                        color=color, fontsize=10, fontweight="bold")

        # Null reference line
        ax.axhline(null_value, color="grey", lw=1.0, ls="--", alpha=0.6,
                   label=f"null ({null_value})" if ax_i == 0 else "_nolegend_")

        short = feat.replace("loci_frac_", "loci:").replace("contacts_corr_", "ctct:")
        ax.set_title(short, fontsize=9)
        x_tick_pos    = [normal_x_pos] + list(range(len(cancer_groups)))
        x_tick_labels = ["Normal"] + [g.split("|")[0].strip() for g in cancer_groups]
        ax.set_xticks(x_tick_pos)
        ax.set_xticklabels(x_tick_labels, fontsize=8, rotation=15, ha="right")
        ax.set_xlim(normal_x_pos - 0.6, len(cancer_groups) - 0.4)
        ax.set_ylabel(f"{method} corrected value", fontsize=8)

        # Shade groups that have this feature in their top list
        for g_i, group in enumerate(cancer_groups):
            if feat in top_feats_per_group.get(group, []):
                ax.axvspan(g_i - 0.45, g_i + 0.45,
                           color=group_colors[group], alpha=0.05, zorder=0)

    # Fully remove unused panels (set_visible leaves hidden objects in AI)
    for ax_i in range(len(all_top), len(axes_flat)):
        fig.delaxes(axes_flat[ax_i])

    # Single legend
    handles = (
        [plt.Line2D([0], [0], marker="o", color="w",
                    markerfacecolor=C_NORMAL, markersize=8, label="Normal")]
        + [plt.Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=group_colors[g], markersize=8, label=g)
           for g in cancer_groups]
        + [plt.Line2D([0], [0], ls="--", color="grey", label=f"null ({null_value})")]
    )

    fig.legend(handles=handles, loc="lower center",
               ncol=len(cancer_groups) + 1,
               fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"Top {n_top} significant features per cancer group  [{method.upper()} correction]\n"
        f"Dots = per-sample means · Diamond = group mean ± SD · * p_adj<0.05",
        fontsize=10, y=1.01,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 0.97])

    if out_path:
        fig.savefig(out_path, bbox_inches="tight", transparent=True, dpi=300)
        print(f"Saved: {out_path}")
    return fig


def plot_selected_features(
    features: list[str],
    df_original: pd.DataFrame,
    loci_corrected: pd.DataFrame,
    contacts_corrected: pd.DataFrame,
    feat_groups: dict,
    cancer_groups: list[str],
    method: str,
    results: pd.DataFrame | None = None,
    out_path: str | None = None,
    group_order: list[str] | None = None,
) -> plt.Figure:
    """
    Plot per-sample corrected means for a user-specified list of features.

    Each feature gets one panel. Within each panel:
      - Each dot is one sample's mean corrected value in its malignant cells.
      - A reference column (always leftmost) shows the same statistic for normal cells.
      - A dashed line marks the correction null (0 for zscore, 0.5 for quantile).
      - If `results` is provided, significant features are annotated with stars.
      - All subplots share a common y-axis range.

    Parameters
    ----------
    features           : list of feature names to plot (must exist in the
                         corrected dataframes).
    df_original        : original (pre-correction) dataframe, used for metadata
                         columns (cell_id, sample_id, sample_group, class).
    loci_corrected     : corrected loci_frac dataframe (must contain 'cell_id').
    contacts_corrected : corrected contacts_corr dataframe.
    feat_groups        : {'loci_frac': [...], 'contacts_corr': [...]} — used to
                         identify which corrected dataframe each feature lives in.
    cancer_groups      : list of sample_group values to display (used to look up
                         data; ordering is overridden by group_order if supplied).
    method             : 'zscore' or 'quantile' — sets the null reference value
                         and the y-axis label.
    results            : (optional) vs_null significance DataFrame produced by
                         test_feature_significance(). When supplied, significant
                         features are annotated with asterisks (* / ** / ***).
                         Pass None to skip annotations entirely.
    out_path           : (optional) file path to save the figure as PNG.
    group_order        : (optional) explicit left-to-right order for cancer groups
                         on the x-axis. Normal is always plotted first (leftmost).
                         Defaults to the order of cancer_groups.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not features:
        raise ValueError("`features` must be a non-empty list of feature names.")

    # Colors: normal = gray, cancer groups use the image palette (salmon, amber, purple …)
    C_NORMAL_NEW    = "#AAAAAA"
    GROUP_PALETTE_NEW = ["#F5A623", "#E8604C",  "#9B59B6", "#2ECC71", "#C0392B"]

    null_value = 0.5 if method == "quantile" else 0.0

    # Determine x-axis group order (normal always first)
    ordered_groups = group_order if group_order is not None else list(cancer_groups)
    group_colors   = {g: GROUP_PALETTE_NEW[i % len(GROUP_PALETTE_NEW)]
                      for i, g in enumerate(ordered_groups)}

    # ── Build combined corrected dataframe ────────────────────────────────────
    loci_feats     = [f for f in feat_groups["loci_frac"]
                      if f in loci_corrected.columns]
    contacts_feats = [f for f in feat_groups["contacts_corr"]
                      if f in contacts_corrected.columns]

    corrected_vals = pd.concat([
        loci_corrected[["cell_id"] + loci_feats],
        contacts_corrected[contacts_feats],
    ], axis=1)

    meta    = df_original[["cell_id", "ROI", "state", "class"]]
    df_corr = meta.merge(corrected_vals, on="cell_id", how="inner")
    mal_df  = df_corr[df_corr["class"] != "normal"]
    norm_df = df_corr[df_corr["class"] == "normal"]

    # ── Validate requested features ───────────────────────────────────────────
    available = set(df_corr.columns) - {"cell_id", "ROI", "state", "class"}
    missing   = [f for f in features if f not in available]
    if missing:
        print(f"  Warning: the following features were not found in the corrected "
              f"data and will be skipped: {missing}")
    plot_feats = [f for f in features if f in available]

    if not plot_feats:
        print("  No valid features to plot.")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No valid features found",
                ha="center", va="center", transform=ax.transAxes)
        if out_path:
            fig.savefig(out_path, bbox_inches="tight", transparent=True, dpi=300)
        return fig

    # ── Prepare significance lookup (optional) ────────────────────────────────
    sig_lookup: pd.DataFrame | None = None
    if results is not None:
        sig = results[results["significant"]].copy()
        sig["abs_t"] = sig["t_stat"].abs()

        if "cancer_group" not in sig.columns and not sig.empty:
            rows = []
            for _, r in sig.iterrows():
                for grp in [r.get("group_a"), r.get("group_b")]:
                    if grp in cancer_groups:
                        rows.append({**r.to_dict(), "cancer_group": grp})
            sig = pd.DataFrame(rows) if rows else pd.DataFrame(
                columns=list(sig.columns) + ["cancer_group"])

        sig_lookup = sig if not sig.empty else None

    # ── Pre-compute global y-limits across all features & groups ─────────────
    all_vals = []
    for feat in plot_feats:
        if feat in norm_df.columns:
            all_vals.extend(norm_df.groupby("ROI")[feat].mean().tolist())
        for group in ordered_groups:
            gdf = mal_df[mal_df["state"] == group]
            if feat in gdf.columns:
                all_vals.extend(gdf.groupby("ROI")[feat].mean().tolist())

    if all_vals:
        v_min, v_max = np.nanmin(all_vals), np.nanmax(all_vals)
        pad   = (v_max - v_min) * 0.15 or 0.1
        y_lo  = v_min - pad
        y_hi  = v_max + pad
    else:
        y_lo, y_hi = -1, 1

    # ── Layout ────────────────────────────────────────────────────────────────
    n_panels = len(plot_feats)
    ncols    = min(n_panels, 4)
    nrows    = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(1 * ncols, 1.5 * nrows),
        sharey=True,   # shared y-axis across all subplots
    )
    fig.patch.set_visible(False)
    axes_flat = np.array(axes).flatten() if n_panels > 1 else [axes]

    normal_x_pos = -1   # normal is always the leftmost column

    for ax_i, feat in enumerate(plot_feats):
        ax = axes_flat[ax_i]

        # ── Normal reference column (gray, always left) ───────────────────────
        if feat in norm_df.columns:
            norm_means = norm_df.groupby("ROI")[feat].mean()
            rng        = np.random.default_rng(0)
            jitter     = rng.uniform(-0.12, 0.12, len(norm_means))
            # Boxplot in the background
            if len(norm_means) > 0:
                ax.boxplot(
                    norm_means.values,
                    positions=[normal_x_pos],
                    widths=0.35,
                    patch_artist=True,
                    boxprops=dict(facecolor=C_NORMAL_NEW, alpha=0.25, linewidth=0.8,
                                  edgecolor=C_NORMAL_NEW),
                    medianprops=dict(color=C_NORMAL_NEW, linewidth=1.5),
                    whiskerprops=dict(color=C_NORMAL_NEW, linewidth=0.8),
                    capprops=dict(color=C_NORMAL_NEW, linewidth=0.8),
                    flierprops=dict(marker="", markersize=0),
                    zorder=2,
                    manage_ticks=False,
                )
            ax.scatter(
                np.full(len(norm_means), normal_x_pos) + jitter,
                norm_means.values,
                color=C_NORMAL_NEW, s=10, alpha=0.75, zorder=3,
                label="Normal" if ax_i == 0 else "_nolegend_",
            )

        # ── Cancer groups in user-defined order ───────────────────────────────
        for g_i, group in enumerate(ordered_groups):
            group_data   = mal_df[mal_df["state"] == group]
            sample_means = (
                group_data.groupby("ROI")[feat].mean()
                if feat in group_data.columns
                else pd.Series(dtype=float)
            )

            color = group_colors[group]
            x_pos = g_i   # 0-based; normal sits at -1

            # Boxplot in the background
            if len(sample_means) > 0:
                ax.boxplot(
                    sample_means.values,
                    positions=[x_pos],
                    widths=0.35,
                    patch_artist=True,
                    boxprops=dict(facecolor=color, alpha=0.20, linewidth=0.8,
                                  edgecolor=color),
                    medianprops=dict(color=color, linewidth=1.5),
                    whiskerprops=dict(color=color, linewidth=0.8),
                    capprops=dict(color=color, linewidth=0.8),
                    flierprops=dict(marker="", markersize=0),
                    zorder=2,
                    manage_ticks=False,
                )

            rng    = np.random.default_rng(42 + g_i)
            jitter = rng.uniform(-0.12, 0.12, len(sample_means))
            ax.scatter(
                np.full(len(sample_means), x_pos) + jitter,
                sample_means.values,
                color=color, s=10, alpha=0.85, zorder=3,
                label=group if ax_i == 0 else "_nolegend_",
            )

            # Significance annotation
            if sig_lookup is not None:
                row = sig_lookup[
                    (sig_lookup["feature"] == feat) &
                    (sig_lookup["cancer_group"] == group)
                ]
                if not row.empty and row["significant"].values[0]:
                    p     = row["p_adj"].values[0]
                    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*"
                    ax.text(x_pos, y_hi * 0.97,
                            stars, ha="center", va="top",
                            color=color, fontsize=10, fontweight="bold")

        # Null reference line
        ax.axhline(null_value, color="grey", lw=1.0, ls="--", alpha=0.6,
                   label=f"null ({null_value})" if ax_i == 0 else "_nolegend_")

        # Apply shared y-limits
        ax.set_ylim(y_lo, y_hi)

        short = feat.replace("loci_frac_", "loci:").replace("contacts_corr_", "ctct:")
        ax.set_title(short, fontsize=9)

        x_tick_pos    = [normal_x_pos] + list(range(len(ordered_groups)))
        x_tick_labels = ["Normal"] + [g.split("|")[0].strip() for g in ordered_groups]
        ax.set_xticks(x_tick_pos)
        ax.set_xticklabels(x_tick_labels, fontsize=8, rotation=15, ha="right")
        ax.set_xlim(normal_x_pos - 0.6, len(ordered_groups) - 0.4)

        # Only show y-label on the leftmost column of each row
        if ax_i % ncols == 0:
            ax.set_ylabel(f"{method} corrected value", fontsize=8)

    # Fully remove unused panels (set_visible leaves hidden objects in AI)
    for ax_i in range(len(plot_feats), len(axes_flat)):
        fig.delaxes(axes_flat[ax_i])

    # ── Single shared legend ──────────────────────────────────────────────────
    handles = (
        [plt.Line2D([0], [0], marker="o", color="w",
                    markerfacecolor=C_NORMAL_NEW, markersize=8, label="Normal")]
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
        f"Selected features  [{method.upper()} correction]\n"
        f"Dots = per-sample means · Box = IQR / median{annot_note}",
        fontsize=10, y=1.01,
    )

    if out_path:
        fig.savefig(out_path, bbox_inches="tight", transparent=True, dpi=300)
        print(f"Saved: {out_path}")

    return fig
 
# ________________________________
# Plot significant feature summary
# ________________________________
DEPL_COL   = '#4878CF'
ELEV_COL   = '#D65F5F'
NS_COL     = '#CCCCCC'
NORMAL_COL = '#888888'
CANCER_PAL = {'LGG_MT': '#E08B3A', 'LGG_noMT': '#6BAE75'}
SIG_THRESH = 0.05
 
def plot_feature_significance(
    df: pd.DataFrame,
    vs_null: pd.DataFrame,
    between: pd.DataFrame,
    out_path: str = None,
    feat_type: Union[str, list] = 'contacts_corr',
    cancer_types: Optional[list] = None,
    figsize: tuple = (17, 4.8),
) -> plt.Figure:
    """
    Five-panel publication figure.

    Parameters
    ----------
    df           : cell-level feature matrix with metadata columns
                   (sample_id, class, type_1, cell_type, ...).
                   Must already be normalised — no further normalisation
                   is applied inside this function.
    vs_null      : statistics from vs-null test. Required columns:
                     feature, feat_type, cancer_group,
                     mean_of_sample_means, p_adj, significant, direction
    between      : statistics from between-group test. Required columns:
                     feature, feat_type, significant, direction
    out_path     : PDF output path
    feat_type    : single string  e.g. 'contacts_corr'
                   OR list        e.g. ['contacts_corr', 'loci_frac']
                   Each entry must match a column prefix in df AND a value
                   in the feat_type column of vs_null / between.
                   When a list is passed, all types are plotted together.
    cancer_types : two cancer type labels matching df['type_1'],
                   e.g. ['LGG_MT', 'LGG_noMT']. Inferred if None.
    figsize      : figure size in inches

    Returns
    -------
    matplotlib Figure
    """

    # ── normalise feat_type to list ───────────────────────────────────────────
    feat_types = [feat_type] if isinstance(feat_type, str) else list(feat_type)

    # ── infer cancer types ────────────────────────────────────────────────────
    if cancer_types is None:
        cancer_types = sorted(df['histology'].unique())
        if len(cancer_types) != 2:
            raise ValueError(
                f"Expected exactly 2 cancer types, found: {cancer_types}. "
                "Pass cancer_types explicitly."
            )
    ct_a, ct_b = cancer_types

    # ── feature columns ───────────────────────────────────────────────────────
    feat_cols = [c for ft in feat_types for c in df.columns if c.startswith(ft)]
    if not feat_cols:
        raise ValueError(f"No columns found in df for feat_type={feat_types}.")
    n_feats = len(feat_cols)
    print(f"Total features: {n_feats}  |  " +
          "  ".join(f"{ft}: {sum(1 for c in feat_cols if c.startswith(ft))}"
                    for ft in feat_types))

    # ── between-group DE features ─────────────────────────────────────────────
    between_sig   = between[between['significant']].set_index('feature')
    between_feats = set(between_sig.index)

    def _between_color(feat):
        if feat not in between_feats:
            return NS_COL
        return DEPL_COL if between_sig.loc[feat, 'direction'] == 'depleted' else ELEV_COL

    colors = np.array([_between_color(f) for f in feat_cols])
    is_bt  = np.array([f in between_feats for f in feat_cols])

    # ── strip: mean of per-sample means (malignant only) ──────────────────────
    def _sample_means(df_sub, class_label):
        grp = df_sub[df_sub['class'] == class_label].groupby('ROI')[feat_cols].mean()
        return grp.mean().values

    strip = {}
    for ct in cancer_types:
        sub = df[df['state'] == ct]
        strip[f'mal_{ct}'] = _sample_means(sub, ct)

    # extra malignant groups from sample_group
    extra_strip = {}
    if 'state' in df.columns:
        covered = set()
        for ct in cancer_types:
            covered.update(df[df['state'] == ct]['state'].unique())
        extra_groups_in_data = [sg for sg in sorted(df['state'].unique())
                                if sg not in covered]
        for sg in extra_groups_in_data:
            sub     = df[df['state'] == sg]
            mal_cls = [c for c in sub['class'].unique() if c != 'normal']
            for mc in mal_cls:
                sub_mc = sub[sub['class'] == mc]
                if len(sub_mc) > 0:
                    grp = sub_mc.groupby('ROI')[feat_cols].mean()
                    key = f'{sg} | {mc}'
                    extra_strip[key] = (grp.mean().values, sg, mc)
    else:
        extra_groups_in_data = []

    # ── vs_null stats ─────────────────────────────────────────────────────────
    vn = vs_null[vs_null['feat_type'].isin(feat_types)].copy()
    n_feats_vn = vn[vn['cancer_group'] == f'{ct_a} | Init']['feature'].nunique()

    bar_data = {}
    for ct in cancer_types:
        label = f'{ct} | Init'
        rows  = vn[vn['cancer_group'] == label]
        bar_data[ct] = {
            'depleted': (rows['significant'] & (rows['direction'] == 'depleted')).sum() / n_feats_vn,
            'elevated': (rows['significant'] & (rows['direction'] == 'elevated')).sum() / n_feats_vn,
        }

    # ── figure layout: bar_null | bar_between | volcano_A | volcano_B | strip ─
    n_extra     = len(extra_strip)
    strip_ratio = 1.6 + max(0, n_extra) * 0.4
    fig = plt.figure(figsize=(figsize[0] + n_extra * 0.4, figsize[1]))
    gs  = gridspec.GridSpec(
        1, 5,
        width_ratios=[1.6, 1.6, 0.9, 0.9, strip_ratio],
        wspace=0.45, left=0.06, right=0.97,
        top=0.87, bottom=0.15,
    )
    ax_v1     = fig.add_subplot(gs[0])   # volcano ct_a
    ax_v2     = fig.add_subplot(gs[1])   # volcano ct_b
    ax_bar    = fig.add_subplot(gs[2])   # vs-null bar
    ax_bar_bt = fig.add_subplot(gs[3])   # between bar
    ax_strip  = fig.add_subplot(gs[4])   # strip (malignant only)

    # ── BAR 1: vs-null ────────────────────────────────────────────────────────
    bar_w = 0.3
    for i, ct in enumerate(cancer_types):
        dep = bar_data[ct]['depleted']
        elv = bar_data[ct]['elevated']
        ax_bar.bar(i, -dep, width=bar_w, color=DEPL_COL, alpha=0.85,
                   edgecolor='white', linewidth=0.4)
        ax_bar.bar(i,  elv, width=bar_w, color=ELEV_COL, alpha=0.85,
                   edgecolor='white', linewidth=0.4)
        if dep > 0:
            ax_bar.text(i, -dep - 0.01, f'{dep*100:.1f}%', ha='center', va='top',
                        fontsize=6, color=DEPL_COL, fontweight='bold')
        if elv > 0:
            ax_bar.text(i,  elv + 0.01, f'{elv*100:.1f}%', ha='center', va='bottom',
                        fontsize=6, color=ELEV_COL, fontweight='bold')

    ax_bar.axhline(0, lw=0.7, color='#888')
    ax_bar.set_xticks(range(len(cancer_types)))
    ax_bar.set_xticklabels(cancer_types, fontsize=7)
    ax_bar.set_xlim(-0.6, len(cancer_types) - 0.4)
    ydep_b = max(bar_data[ct]['depleted'] for ct in cancer_types) + 0.05
    yelv_b = max(bar_data[ct]['elevated'] for ct in cancer_types) + 0.05
    ax_bar.set_ylabel('Fraction of features', fontsize=7)
    ax_bar.set_title(f'Sig. vs null\n(n={n_feats_vn})', fontsize=7, fontweight='bold')
    ax_bar.legend(handles=[
        mpatches.Patch(color=DEPL_COL, label='Depleted'),
        mpatches.Patch(color=ELEV_COL, label='Elevated'),
    ], fontsize=5.5, loc='lower right')

    # ── BAR 2: between-group DE ───────────────────────────────────────────────
    bt_filt   = between[between['feat_type'].isin(feat_types)]
    n_feats_bt = bt_filt['feature'].nunique()

    n_dep_bt = (bt_filt['significant'] & (bt_filt['direction'] == 'depleted')).sum()
    n_elv_bt = (bt_filt['significant'] & (bt_filt['direction'] == 'elevated')).sum()
    dep_bt   = n_dep_bt / n_feats_bt if n_feats_bt > 0 else 0
    elv_bt   = n_elv_bt / n_feats_bt if n_feats_bt > 0 else 0

    ax_bar_bt.bar(0, -dep_bt, width=bar_w, color=DEPL_COL, alpha=0.85,
                  edgecolor='white', linewidth=0.4)
    ax_bar_bt.bar(0,  elv_bt, width=bar_w, color=ELEV_COL, alpha=0.85,
                  edgecolor='white', linewidth=0.4)
    if dep_bt > 0:
        ax_bar_bt.text(0, -dep_bt - 0.01, f'{dep_bt*100:.1f}%', ha='center', va='top',
                       fontsize=6, color=DEPL_COL, fontweight='bold')
    if elv_bt > 0:
        ax_bar_bt.text(0,  elv_bt + 0.01, f'{elv_bt*100:.1f}%', ha='center', va='bottom',
                       fontsize=6, color=ELEV_COL, fontweight='bold')

    ax_bar_bt.axhline(0, lw=0.7, color='#888')
    ax_bar_bt.set_xticks([0])
    ax_bar_bt.set_xticklabels([f'{ct_a}\nvs {ct_b}'], fontsize=7)
    ax_bar_bt.set_xlim(-0.6, 0.6)
    ax_bar_bt.set_title(f'Sig. between groups\n(n={n_feats_bt})', fontsize=7, fontweight='bold')

    # ── shared y-axis for both bar plots ──────────────────────────────────────
    shared_ymax = max(ydep_b, yelv_b, dep_bt + 0.05, elv_bt + 0.05)
    shared_fmt  = matplotlib.ticker.FuncFormatter(lambda x, _: f'{abs(x)*100:.0f}%')
    for ax in [ax_bar, ax_bar_bt]:
        ax.set_ylim(-shared_ymax, shared_ymax)
        ax.yaxis.set_major_formatter(shared_fmt)

    # ── VOLCANOES ─────────────────────────────────────────────────────────────
    xall    = vn['mean_of_sample_means']
    xpad    = (xall.max() - xall.min()) * 0.05
    xlims   = (xall.min() - xpad, xall.max() + xpad)
    yall    = -np.log10(vn['p_adj'].clip(lower=1e-10))
    ylims_v = (-0.2, yall.max() * 1.08)

    for ax, ct in [(ax_v1, ct_a), (ax_v2, ct_b)]:
        label  = f'{ct} | Init'
        col_ct = CANCER_PAL.get(ct, '#999')
        rows   = vn[vn['cancer_group'] == label].copy()
        rows['neglog10p']  = -np.log10(rows['p_adj'].clip(lower=1e-10))
        rows['in_between'] = rows['feature'].isin(between_feats)
        rows['bt_color']   = rows['feature'].apply(_between_color)

        n_sig = rows['significant'].sum()
        n_dep = (rows['significant'] & (rows['direction'] == 'depleted')).sum()
        n_elv = (rows['significant'] & (rows['direction'] == 'elevated')).sum()
        pct   = n_sig / n_feats_vn * 100

        ax.scatter(rows.loc[~rows['in_between'], 'mean_of_sample_means'],
                   rows.loc[~rows['in_between'], 'neglog10p'],
                   s=15, color=NS_COL, alpha=0.5, linewidths=0, zorder=2)
        for color in [DEPL_COL, ELEV_COL]:
            sub = rows[rows['in_between'] & (rows['bt_color'] == color)]
            ax.scatter(sub['mean_of_sample_means'], sub['neglog10p'],
                       s=15, color=color, alpha=0.88, linewidths=0.3,
                       edgecolors='white', zorder=4)

        ax.axhline(-np.log10(SIG_THRESH), lw=0.7, color='#888', linestyle='--')
        ax.axvline(0, lw=0.5, color='#ccc', linestyle='--')
        ax.set_xlim(xlims); ax.set_ylim(ylims_v)
        ax.set_xlabel('Mean of sample means', fontsize=7)
        ax.set_ylabel('-log₁₀(p_adj)', fontsize=7)
        ax.set_title(f'{ct}  ({pct:.1f}% sig, n={n_sig})\n↓ {n_dep}   ↑ {n_elv}',
                     fontsize=7, fontweight='bold', color=col_ct)
        ax.text(0.98, 0.02, f'n={n_feats_vn}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=5.5, color='#888')

    # ── STRIP (malignant only) ────────────────────────────────────────────────
    rng = np.random.default_rng(42)

    if extra_strip:
        extra_base_colors = plt.cm.Set2(np.linspace(0, 0.8, max(len(extra_groups_in_data), 1)))
        sg_color = {sg: extra_base_colors[i]
                    for i, sg in enumerate(extra_groups_in_data)}
    else:
        sg_color = {}

    malignant_groups = [
        (strip[f'mal_{ct_a}'], ct_a, CANCER_PAL.get(ct_a, '#999')),
        (strip[f'mal_{ct_b}'], ct_b, CANCER_PAL.get(ct_b, '#999')),
    ]

    if extra_strip:
        for key, (vals, sg, mc) in sorted(extra_strip.items()):
            col   = matplotlib.colors.to_hex(sg_color[sg])
            label = key.replace(f' | {mc}', f'\n{mc}')
            malignant_groups.append((vals, label, col))

    groups    = malignant_groups
    n_groups  = len(groups)
    all_strip = np.concatenate([g[0] for g in groups])
    ymin_s    = np.nanpercentile(all_strip, 0.5)
    ymax_s    = np.nanpercentile(all_strip, 99.5)
    ypad_s    = (ymax_s - ymin_s) * 0.05
    ylims_s   = (ymin_s - ypad_s, ymax_s + ypad_s)

    for pos, (vals, label, col) in enumerate(groups):
        jit = rng.uniform(-0.15, 0.15, (~is_bt).sum())
        ax_strip.scatter(pos + jit, vals[~is_bt],
                         s=3, color=NS_COL, alpha=0.4, linewidths=0, zorder=2)
        for color in [DEPL_COL, ELEV_COL]:
            mask = is_bt & (colors == color)
            jit2 = rng.uniform(-0.12, 0.12, mask.sum())
            ax_strip.scatter(pos + jit2, vals[mask],
                             s=15, color=color, alpha=0.88, linewidths=0, zorder=3)

    ax_strip.axhline(0, lw=0.7, color='#888', linestyle='--')
    ax_strip.set_xticks(range(n_groups))
    ax_strip.set_xticklabels([g[1] for g in groups], fontsize=5.5,
                              rotation=30, ha='right')
    ax_strip.set_xlim(-0.5, n_groups - 0.5)
    ax_strip.set_ylim(ylims_s)
    ax_strip.set_ylabel('Mean of sample means', fontsize=7)
    ax_strip.set_title(
        f'Malignant cell means\n'
        f'(blue/red = DE between groups, n={len(between_feats)})',
        fontsize=7, fontweight='bold')
    ax_strip.grid(axis='y', alpha=0.12, lw=0.4)
    ax_strip.text(0.02, 0.02, 'mean of per-sample means',
                  transform=ax_strip.transAxes, fontsize=5, color='#aaa', va='bottom')

    # ── legend ────────────────────────────────────────────────────────────────
    fig.legend(handles=[
        mpatches.Patch(color=NS_COL,   label='Not DE between groups'),
        mpatches.Patch(color=DEPL_COL, label=f'Depleted in {ct_a} vs {ct_b}'),
        mpatches.Patch(color=ELEV_COL, label=f'Elevated in {ct_a} vs {ct_b}'),
    ], loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.04), fontsize=6.5)

    fig.suptitle(f'{" + ".join(feat_types)} — significance overview',
                 fontsize=8, fontweight='bold')

    if out_path:
        fig.savefig(out_path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Saved: {out_path}")
    return fig

# ── dendrogram appearance ─────────────────────────────────────────────────────
DEND_LW      = 0.25     # tree line width (pt)
DEND_COLOR   = '#555555'
ROW_DEND_W   = 0.045    # width of the row-tree track (fraction of heatmap width) -> narrow
COL_DEND_H   = 0.11     # height of the col-tree track (fraction of heatmap height) -> narrow
 
 
def _thin_dend(ax, lw=DEND_LW, color=DEND_COLOR):
    """Force every dendrogram link to a thin line, robust across mpl/scipy versions."""
    for coll in ax.collections:
        coll.set_linewidth(lw)
        coll.set_color(color)
    for ln in ax.lines:
        ln.set_linewidth(lw)
        ln.set_color(color)
 
 
# separator shown between the two loci of a contact feature (e.g. chr10p–chr9p)
LOCUS_SEP = '\u2013'   # en dash; change to '-', ':' or '\u00d7' (×) if preferred
 
 
def _clean_feature_label(f):
    """Strip the analysis prefix and prettify a chromosome/arm contact feature.
    'contact_frac_OE_chr10_p__chr9_p' -> 'chr10p–chr9p'
    'loci_frac_chr7_p'                -> 'loci:chr7p'
    """
    s = f
    for pref in ('contact_frac_OE_', 'contacts_corr_', 'loci_frac_'):
        if s.startswith(pref):
            s = s[len(pref):]
            if pref == 'loci_frac_':
                s = 'loci:' + s
            break
    s = s.replace('__', LOCUS_SEP)   # separator between the two loci
    s = s.replace('_', '')           # chr10_p -> chr10p
    return s
 
 
def plot_annotated_heatmap(
    df: pd.DataFrame,
    sig_features: list,
    out_path: str = 'feature_heatmap_annotated.pdf',
    main_types: Optional[list] = None,
    extra_type_keywords: Optional[list] = None,
    figsize_width_per_feature: float = 0.13,
    figsize_base_height: float = 6.0,
    cmap: str = 'RdBu_r',
    vmax_pct: float = 95,
) -> plt.Figure:
    """Clustered annotated heatmap of significant features (see module docstring)."""
    sig_features = list(sig_features)
 
    # ── classify sample groups into main vs extra ─────────────────────────────
    all_groups = sorted(df['state'].unique())
    if extra_type_keywords:
        extra_types = [t for t in all_groups if any(kw in t for kw in extra_type_keywords)]
    else:
        extra_types = []
    if main_types is None:
        main_types = [t for t in all_groups if t not in extra_types]
 
    print(f"Main panel types:  {main_types}")
    print(f"Extra panel types: {extra_types}")
 
    # ── build mean matrix ─────────────────────────────────────────────────────
    def _build_matrix(group_list):
        rows = []
        for sg in group_list:
            sub = df[df['state'] == sg]
            for sid in sub['ROI'].unique():
                sub_s = sub[sub['ROI'] == sid]
                for cls in sub_s['class'].unique():
                    cells = sub_s[sub_s['class'] == cls]
                    means = cells[sig_features].mean()
                    rows.append({'ROI': sid, 'class': cls,
                                 'state': sg, **means.to_dict()})
        return pd.DataFrame(rows)
 
    main_mat  = _build_matrix(main_types)
    extra_mat = _build_matrix(extra_types) if extra_types else None
    has_extra = extra_mat is not None and len(extra_mat) > 0
 
    print(f"Main matrix:  {main_mat.shape[0]} rows × {len(sig_features)} features")
    if has_extra:
        print(f"Extra matrix: {extra_mat.shape[0]} rows × {len(sig_features)} features")
 
    # ── z-score and cluster ───────────────────────────────────────────────────
    def _zscore(data):
        mu = np.nanmean(data, axis=0, keepdims=True)
        sd = np.nanstd(data,  axis=0, keepdims=True)
        sd[sd == 0] = 1
        return (data - mu) / sd
 
    def _cluster(mat):
        data   = mat[sig_features].values.astype(float)
        data_z = _zscore(data)
        clean  = np.nan_to_num(data_z)
        r_link = hierarchy.linkage(pdist(clean,   'euclidean'), 'average')
        c_link = hierarchy.linkage(pdist(clean.T, 'euclidean'), 'average')
        r_ord  = hierarchy.leaves_list(r_link)
        c_ord  = hierarchy.leaves_list(c_link)
        return data_z, r_link, c_link, r_ord, c_ord
 
    data_z, row_link, col_link, row_ord, col_ord = _cluster(main_mat)
    data_ordered = data_z[np.ix_(row_ord, col_ord)]
    mat_ordered  = main_mat.iloc[row_ord].reset_index(drop=True)
    feat_ordered = [sig_features[i] for i in col_ord]
    feat_labels  = [_clean_feature_label(f) for f in feat_ordered]
 
    n_rows = data_ordered.shape[0]
    n_cols = data_ordered.shape[1]
    vmax   = float(np.nanpercentile(np.abs(data_ordered), vmax_pct))
    norm   = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
 
    # ── figure layout ─────────────────────────────────────────────────────────
    fig_w = max(14, n_cols * figsize_width_per_feature)
    fig_h = figsize_base_height + (2.5 if has_extra else 0)
    fig   = plt.figure(figsize=(fig_w, fig_h))
 
    if has_extra:
        n_extra_rows = len(extra_mat)
        outer = gridspec.GridSpec(
            2, 1, height_ratios=[n_rows, max(n_extra_rows, 2)],
            hspace=0.4, left=0.01, right=0.82, top=0.94, bottom=0.15)
    else:
        outer = gridspec.GridSpec(1, 1, left=0.01, right=0.82, top=0.94, bottom=0.15)
 
    # inner grid now 2x2:  [corner | col-dendro] / [row-dendro | heatmap]
    def _make_inner(subplot_spec):
        return gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=subplot_spec,
            height_ratios=[COL_DEND_H, 1],
            width_ratios=[ROW_DEND_W, 1],
            hspace=0.02, wspace=0.02)
 
    inner_main  = _make_inner(outer[0])
    fig.add_subplot(inner_main[0, 0]).axis('off')          # corner
    ax_col_dend = fig.add_subplot(inner_main[0, 1])
    ax_row_dend = fig.add_subplot(inner_main[1, 0])
    ax_heat     = fig.add_subplot(inner_main[1, 1])
 
    ax_cbar = fig.add_axes([0.855, 0.3, 0.012, 0.4])
 
    # ── column dendrogram (top) ───────────────────────────────────────────────
    with plt.rc_context({'lines.linewidth': DEND_LW}):
        hierarchy.dendrogram(col_link, ax=ax_col_dend,
                             link_color_func=lambda k: DEND_COLOR, no_labels=True)
    _thin_dend(ax_col_dend)
    ax_col_dend.axis('off')      # keep scipy's native xlim so leaves sit over columns
 
    # ── row dendrogram (left) ─────────────────────────────────────────────────
    with plt.rc_context({'lines.linewidth': DEND_LW}):
        hierarchy.dendrogram(row_link, ax=ax_row_dend, orientation='left',
                             link_color_func=lambda k: DEND_COLOR, no_labels=True)
    _thin_dend(ax_row_dend)
    ax_row_dend.invert_yaxis()   # match imshow(origin='upper'): row 0 at top
    ax_row_dend.axis('off')
 
    # ── main heatmap ──────────────────────────────────────────────────────────
    im = ax_heat.imshow(data_ordered, aspect='auto', cmap=cmap,
                        norm=norm, interpolation='nearest')
    ax_heat.set_xticks(range(n_cols))
    ax_heat.set_xticklabels(feat_labels, fontsize=3.5, rotation=90)
    ax_heat.set_yticks([])                        # row labels drawn on the right instead
    ax_heat.tick_params(length=0)
    for sp in ax_heat.spines.values():
        sp.set_visible(False)
    ax_heat.set_title(' & '.join(main_types), fontsize=7, fontweight='bold', loc='left', pad=3)
 
    # ── colorbar ──────────────────────────────────────────────────────────────
    cb = fig.colorbar(im, cax=ax_cbar, orientation='vertical')
    cb.ax.tick_params(labelsize=5)
    cb.set_label('Z-score', fontsize=5.5)
 
    # ── annotation strips + right-side row labels (heatmap data coords) ────────
    ann_step = n_cols * 0.015
    ann_x0   = n_cols + 0.3
 
    def _draw_annotations(heat_ax, mat_ord):
        for i, (_, row) in enumerate(mat_ord.iterrows()):
            heat_ax.add_patch(mpatches.Rectangle(
                (ann_x0, i - 0.5), ann_step, 1,
                color=CLASS_PAL.get(row['class'], FALLBACK_COLOR),
                lw=0, transform=heat_ax.transData, clip_on=False))
            heat_ax.add_patch(mpatches.Rectangle(
                (ann_x0 + ann_step * 1.4, i - 0.5), ann_step, 1,
                color=TYPE_PAL.get(row['state'], FALLBACK_COLOR),
                lw=0, transform=heat_ax.transData, clip_on=False))
            heat_ax.text(ann_x0 + ann_step * 2.9, i,
                         f"{row['ROI']}  {row['class']}",
                         ha='left', va='center', fontsize=6.5,
                         transform=heat_ax.transData, clip_on=False)
        heat_ax.text(ann_x0 + ann_step * 0.5, -0.7, 'class',
                     ha='center', va='bottom', fontsize=4.5, rotation=45,
                     color='#555', transform=heat_ax.transData, clip_on=False)
        heat_ax.text(ann_x0 + ann_step * 1.9, -0.7, 'group',
                     ha='center', va='bottom', fontsize=4.5, rotation=45,
                     color='#555', transform=heat_ax.transData, clip_on=False)
 
    _draw_annotations(ax_heat, mat_ordered)
 
    # ── extra panel ───────────────────────────────────────────────────────────
    if has_extra:
        inner_extra = _make_inner(outer[1])
        fig.add_subplot(inner_extra[0, 0]).axis('off')     # corner
        fig.add_subplot(inner_extra[0, 1]).axis('off')     # no col tree (shares main order)
        ax_ex_dend  = fig.add_subplot(inner_extra[1, 0])
        ax_ex_heat  = fig.add_subplot(inner_extra[1, 1])
 
        ex_data  = extra_mat[sig_features].values.astype(float)
        ex_z     = _zscore(ex_data)
        ex_r_lnk = hierarchy.linkage(pdist(np.nan_to_num(ex_z), 'euclidean'), 'average')
        ex_r_ord = hierarchy.leaves_list(ex_r_lnk)
        ex_ordered = ex_z[np.ix_(ex_r_ord, col_ord)]        # reuse main column order
        ex_mat_ord = extra_mat.iloc[ex_r_ord].reset_index(drop=True)
        n_ex = len(ex_mat_ord)
 
        with plt.rc_context({'lines.linewidth': DEND_LW}):
            hierarchy.dendrogram(ex_r_lnk, ax=ax_ex_dend, orientation='left',
                                 link_color_func=lambda k: DEND_COLOR, no_labels=True)
        _thin_dend(ax_ex_dend)
        ax_ex_dend.invert_yaxis()
        ax_ex_dend.axis('off')
 
        ax_ex_heat.imshow(ex_ordered, aspect='auto', cmap=cmap,
                          norm=norm, interpolation='nearest')
        ax_ex_heat.set_xticks(range(n_cols))
        ax_ex_heat.set_xticklabels(feat_labels, fontsize=3.5, rotation=90)
        ax_ex_heat.set_yticks([])
        ax_ex_heat.tick_params(length=0)
        for sp in ax_ex_heat.spines.values():
            sp.set_visible(False)
        ax_ex_heat.set_title(', '.join(extra_types), fontsize=7,
                             fontweight='bold', loc='left', pad=3)
        _draw_annotations(ax_ex_heat, ex_mat_ord)
 
    # ── legend ────────────────────────────────────────────────────────────────
    unique_classes = mat_ordered['class'].unique().tolist()
    unique_groups  = mat_ordered['state'].unique().tolist()
    if has_extra:
        unique_classes += extra_mat['class'].unique().tolist()
        unique_groups  += extra_mat['state'].unique().tolist()
    handles = (
        [mpatches.Patch(color=CLASS_PAL.get(c, FALLBACK_COLOR), label=f'class: {c}')
         for c in sorted(set(unique_classes))] +
        [mpatches.Patch(color=TYPE_PAL.get(g, FALLBACK_COLOR), label=f'group: {g}')
         for g in sorted(set(unique_groups))]
    )
    fig.legend(handles=handles, loc='center right', fontsize=6,
               bbox_to_anchor=(1.0, 0.5), ncol=1, title='Annotation',
               title_fontsize=6, frameon=False)
 
    fig.suptitle(f'Significant features — {len(sig_features)} features  |  z-scored per feature',
                 fontsize=8, fontweight='bold')
 
    if out_path:
        fig.savefig(out_path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Saved: {out_path}")
    return fig


def plot_updown_bars(df: pd.DataFrame,
                     threshold: float = 0.15,
                     figsize: tuple = None,
                     group_order: list = None,
                     out_path: str = None):
    """
    Plot a 100% stacked bar chart: each sample bar shows fraction of features
    that are up (red), down (blue), or unchanged (light gray).
    Samples are grouped by sample_group with a gap between groups.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns:
          - 'sample_id'    : sample identifier
          - 'sample_group' : group label
          - all remaining columns are treated as feature/data columns
    threshold : float
        Absolute value cutoff. Values > threshold are "up", < -threshold are "down".
    figsize : tuple, optional
        Figure size (width, height). Auto-sized if None.
    group_order : list, optional
        Explicit left-to-right order of group names. Defaults to appearance order in df.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    df = df.reset_index(drop=True).copy()

    meta_cols = ["ROI", "state"]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    data      = df[feat_cols].values
    n_feat    = data.shape[1]

    frac_up      = (data >  threshold).sum(axis=1) / n_feat
    frac_down    = (data < -threshold).sum(axis=1) / n_feat
    frac_neutral = 1.0 - frac_up - frac_down

    UP_COLOR      = "#c0392b"
    DOWN_COLOR    = "#2980b9"
    NEUTRAL_COLOR = "#d8d8d8"
    BAR_WIDTH     = 0.7
    GROUP_GAP     = 1.2

    # group order
    if group_order is None:
        groups = list(dict.fromkeys(df["state"].tolist()))
    else:
        groups = group_order

    group_palette = {
        g: c for g, c in zip(
            groups,
            ["#1d6fa4", "#c0582a", "#2e8b57",
             "#7b3fa0", "#b5860d", "#a0522d", "#4682b4"],
        )
    }

    # build plot order and x positions
    # within each group, sort by fraction_neutral descending (most unchanged first)
    plot_order    = []
    x_values      = []
    group_label_x = {}
    group_spans   = {}
    x_cursor = 0

    for g in groups:
        row_positions = df.index[df["state"] == g].tolist()
        # sort these positions by frac_neutral descending
        row_positions = sorted(row_positions, key=lambda p: frac_neutral[p], reverse=True)
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
        figsize = (max(8, n_samples * 0.24 + 2), 6)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#f8f7f4")
    ax.set_facecolor("#f8f7f4")

    # draw stacked bars: down (from 0 down to -frac_down), neutral (centered around 0),
    # up (from 0 up to frac_up) — but since we want 100% stacked we go bottom to top:
    # bottom segment = down (blue), middle = neutral (gray), top = up (red)
    # Stack from bottom=0: down | neutral | up, all positive, total = 1.0
    for i, pos in enumerate(plot_order):
        x   = x_values[i]
        fd  = frac_down[pos]
        fn  = frac_neutral[pos]
        fu  = frac_up[pos]

        # down at bottom (blue)
        ax.bar(x, fd, width=BAR_WIDTH, bottom=0,
               color=DOWN_COLOR, alpha=0.88, linewidth=0)
        # up in middle (red)
        ax.bar(x, fu, width=BAR_WIDTH, bottom=fd,
               color=UP_COLOR, alpha=0.88, linewidth=0)
        # neutral on top (gray)
        ax.bar(x, fn, width=BAR_WIDTH, bottom=fd + fu,
               color=NEUTRAL_COLOR, alpha=0.9, linewidth=0)

    # y-axis: 0–100%
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"],
                       fontsize=8, color="#666666")

    # group labels + separators
    for i, (g, (x0, x1)) in enumerate(group_spans.items()):
        col = group_palette.get(g, "#888888")
        if i > 0:
            ax.axvline(x0, color="#cccccc", linewidth=0.6, linestyle="--", zorder=0)
        ax.text(
            group_label_x[g], 1.15,
            g,
            ha="center", va="top",
            fontsize=9, fontweight="bold",
            color=col,
        )

    # x-axis sample labels
    ax.set_xticks(x_values)
    ax.set_xticklabels(
        df["ROI"].iloc[plot_order].tolist(),
        rotation=45, ha="right",
        fontsize=7, color="#444444",
    )

    ax.set_xlim(x_values.min() - 0.8, x_values.max() + 0.8)

    # spines
    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.tick_params(axis="both", length=0)

    # legend
    legend_handles = [
        mpatches.Patch(color=UP_COLOR,      alpha=0.88, label=f"Up (> +{threshold})"),
        mpatches.Patch(color=NEUTRAL_COLOR, alpha=0.90, label=f"Unchanged"),
        mpatches.Patch(color=DOWN_COLOR,    alpha=0.88, label=f"Down (< -{threshold})"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right", fontsize=8,
        frameon=True, framealpha=0.85,
        edgecolor="#cccccc",
    )

    ax.set_title(
        f"Feature direction per sample  |  threshold = +/-{threshold}",
        fontsize=10, color="#333333", pad=10,
    )

    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Saved: {out_path}")
    return fig
    
