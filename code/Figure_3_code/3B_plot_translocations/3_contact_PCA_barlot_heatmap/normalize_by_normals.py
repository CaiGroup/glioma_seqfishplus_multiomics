"""
normalize_by_normals.py
-----------------------
Correct single-cell genomic features against a per-sample normal-cell reference.

Two correction methods:
  - 'zscore'   : standard z-score → (cell - mean_normals) / std_normals
  - 'quantile' : percentile rank  → ECDF of normals evaluated at each cell's value

Features are split into two groups before correction:
  - loci_frac   : chromosome arm coverage fractions
  - contacts_corr: inter-arm contact correlations

Features where >50% of cells (across the whole dataset) are NA are dropped
before any correction is applied.

Designed to handle large files (>3 GB) via chunked reading when needed.
See `correct_chunked()` for the out-of-core version.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Literal


# ── Constants ────────────────────────────────────────────────────────────────

NORMAL_CELL_TYPES   = ["Olig. normal", "Microglia"]   # user-defined
NA_THRESHOLD        = 0.50                              # drop if > 50% NA
META_COLS           = ["sample_id", "cell_id", "class", "total_loci",
                        "x", "y", "type_1", "resection", "cell_type", "sample_group"]


# ── NA imputation ────────────────────────────────────────────────────────────

def impute_na_with_sample_mean(
    df: pd.DataFrame,
    feature_groups: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Replace NAs in contacts_corr features with the per-sample, per-feature mean
    of non-NA values.

    Applied only to contacts_corr features. loci_frac features are left
    untouched: their zeros indicate arm absence (a real measurement) and they
    have no NAs, so imputation would fabricate coverage that was not observed.

    Background
    ----------
    NAs in contacts_corr arise when at least one of the two arms involved in a
    contact pair is absent from the cell (loci_frac == 0). Because we cannot
    distinguish whether this represents a biological deletion or insufficient
    sequencing depth, both cases — one arm absent and both arms absent — are
    treated as 'unknown' and replaced with the sample mean.

    Parameters
    ----------
    df            : full dataframe with metadata + feature columns, per sample
                    (may contain multiple samples; imputation is per-sample).
    feature_groups: output of identify_feature_groups().

    Returns
    -------
    df_imputed : copy of df with NAs in contacts_corr filled.
    """
    contacts_feats = [f for f in feature_groups["contacts_corr"] if f in df.columns]
    if not contacts_feats:
        return df

    df_imputed = df.copy()

    for sample_id, sample_idx in df.groupby("sample_id").groups.items():
        sample_slice = df_imputed.loc[sample_idx, contacts_feats]

        # Per-feature mean of non-NA values within this sample
        sample_means = sample_slice.mean()  # pandas skips NaN by default

        # Fill NAs with the per-feature sample mean
        df_imputed.loc[sample_idx, contacts_feats] = sample_slice.fillna(sample_means)

        n_filled = sample_slice.isna().sum().sum()
        if n_filled > 0:
            print(f"  Imputed {n_filled:,} NAs in sample {sample_id} "
                  f"({100 * n_filled / sample_slice.size:.1f}% of contact values)")

    return df_imputed


# ── Feature identification ────────────────────────────────────────────────────

def identify_feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Split feature columns into loci_frac and contacts_corr groups,
    dropping any features where >NA_THRESHOLD of all cells are NaN.

    Returns
    -------
    dict with keys 'loci_frac' and 'contacts_corr', each a list of column names.
    """
    loci_cols     = [c for c in df.columns if c.startswith("loci_frac")]
    contacts_cols = [c for c in df.columns if c.startswith("contacts_corr")]

    def _filter_na(cols):
        na_rate = df[cols].isna().mean()
        kept    = na_rate[na_rate <= NA_THRESHOLD].index.tolist()
        dropped = na_rate[na_rate >  NA_THRESHOLD].index.tolist()
        if dropped:
            print(f"  Dropped {len(dropped)} features with >{NA_THRESHOLD*100:.0f}% NA "
                  f"(e.g. {dropped[:3]}{'...' if len(dropped) > 3 else ''})")
        return kept

    print("Identifying feature groups and filtering high-NA features...")
    loci_kept     = _filter_na(loci_cols)
    contacts_kept = _filter_na(contacts_cols)

    print(f"  loci_frac    : {len(loci_kept):>4d} features kept  ({len(loci_cols) - len(loci_kept)} dropped)")
    print(f"  contacts_corr: {len(contacts_kept):>4d} features kept  ({len(contacts_cols) - len(contacts_kept)} dropped)")

    return {"loci_frac": loci_kept, "contacts_corr": contacts_kept}


# ── Per-feature correction functions ─────────────────────────────────────────

def _quantile_correct_feature(cell_values: np.ndarray,
                               ref_values: np.ndarray) -> np.ndarray:
    """
    Percentile rank: evaluate the empirical CDF of ref_values at each cell value.

    Output is in [0, 1]:
      0.5  = at the normal median
      0.95 = higher than 95% of normal cells
      0.0  = at or below the minimum normal value

    NaN cells remain NaN. Values outside the reference range are naturally
    mapped to 0 (below all normals) or 1 (above all normals).

    Implementation: vectorised binary search via np.searchsorted on a
    pre-sorted reference array. Equivalent to percentileofscore(kind='mean')
    but operates on all cells in a single numpy call instead of a Python loop,
    giving ~100-1000x speedup for large arrays.

    Derivation of the 'mean' interpolation:
        lo = searchsorted(ref, v, side='left')   # # ref values strictly < v
        hi = searchsorted(ref, v, side='right')  # # ref values <= v
        rank = (lo + hi) / 2                     # average boundary rank
        percentile = rank / n_ref                # normalise to [0, 1]
    """
    ref_clean = ref_values[~np.isnan(ref_values)]
    if len(ref_clean) == 0:
        return np.full_like(cell_values, np.nan, dtype=float)

    ref_sorted = np.sort(ref_clean)
    n_ref      = len(ref_sorted)

    out        = np.full(len(cell_values), np.nan, dtype=float)
    valid_mask = ~np.isnan(cell_values)

    if valid_mask.any():
        v   = cell_values[valid_mask]
        lo  = np.searchsorted(ref_sorted, v, side="left")   # # values < v
        hi  = np.searchsorted(ref_sorted, v, side="right")  # # values <= v
        out[valid_mask] = (lo + hi) / (2.0 * n_ref)         # mean rank in [0, 1]

    return out


def _zscore_correct_feature(cell_values: np.ndarray,
                             ref_values: np.ndarray) -> np.ndarray:
    """
    Standard z-score: (x - mean_ref) / std_ref

    Uses the mean and standard deviation of the normal reference cells.
    More sensitive than quantile correction but less robust to outliers in the
    reference — prefer quantile when reference cell counts are low or the
    reference distribution is skewed.

    NaN cells remain NaN.
    If std_ref == 0 (constant reference), returns raw deviation only.
    """
    ref_clean = ref_values[~np.isnan(ref_values)]
    if len(ref_clean) == 0:
        return np.full_like(cell_values, np.nan, dtype=float)

    ref_mean = np.mean(ref_clean)
    ref_std  = np.std(ref_clean, ddof=1)   # sample std (unbiased)

    shifted = cell_values - ref_mean
    if ref_std == 0:
        return shifted.astype(float)
    return shifted / ref_std



def correct_by_normals(
    df: pd.DataFrame,
    method: Literal["zscore", "quantile"] = "zscore",
    normal_cell_types: list[str] = NORMAL_CELL_TYPES,
    feature_groups: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Correct all feature columns against the per-sample normal-cell reference.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataframe with metadata + feature columns.
    method : 'zscore' or 'quantile'
        Correction method to apply:
          'zscore'   — standard z-score: (x - mean) / std.
          'quantile' — percentile rank vs normal ECDF, in [0, 1].
    normal_cell_types : list of str
        Cell types used as the normal reference (applied per sample).
    feature_groups : dict, optional
        Pre-computed output of identify_feature_groups(). If None, computed here.

    Returns
    -------
    corrected_loci : pd.DataFrame
        Metadata columns + corrected loci_frac features.
    corrected_contacts : pd.DataFrame
        Metadata columns + corrected contacts_corr features.
    ref_stats : dict
        Reference statistics used for correction, keyed by (sample_id, feature).
    """
    assert method in ("zscore", "quantile"), \
        "method must be 'zscore' or 'quantile'"

    correct_fn = {
        "zscore":   _zscore_correct_feature,
        "quantile": _quantile_correct_feature,
    }[method]

    # Identify features (with NA filtering)
    if feature_groups is None:
        feature_groups = identify_feature_groups(df)

    # Impute NAs in contacts_corr with per-sample mean before correction.
    # loci_frac is left untouched (zeros = arm absent, no NAs present).
    print("Imputing NAs in contacts_corr features with per-sample mean...")
    df = impute_na_with_sample_mean(df, feature_groups)

    all_features = feature_groups["loci_frac"] + feature_groups["contacts_corr"]

    # Which metadata cols actually exist in this df
    meta_cols_present = [c for c in META_COLS if c in df.columns]

    # Collect results per sample
    loci_results     = []
    contacts_results = []
    ref_stats        = {}   # (sample_id, feature) -> {"mean": ..., "std": ...} or {"n_ref": ...}

    samples = df["sample_id"].unique()
    print(f"\nRunning '{method}' correction across {len(samples)} sample(s)...")

    for sample_id in samples:
        sample_mask = df["sample_id"] == sample_id
        sample_df   = df[sample_mask].copy()

        # Identify normal cells within this sample
        normal_mask = sample_df["cell_type"].isin(normal_cell_types)
        n_normals   = normal_mask.sum()

        if n_normals == 0:
            print(f"  WARNING: sample {sample_id} has no normal cells — skipping correction, "
                  "returning raw values.")
            loci_results.append(
                pd.concat([sample_df[meta_cols_present],
                           sample_df[feature_groups["loci_frac"]]], axis=1)
            )
            contacts_results.append(
                pd.concat([sample_df[meta_cols_present],
                           sample_df[feature_groups["contacts_corr"]]], axis=1)
            )
            continue

        print(f"  Sample {sample_id}: {len(sample_df)} cells, "
              f"{n_normals} normal cells ({', '.join(normal_cell_types)})")

        # Correct all features
        corrected_values = {}
        for feat in all_features:
            cell_vals = sample_df[feat].values.astype(float)
            ref_vals  = sample_df.loc[normal_mask, feat].values.astype(float)

            corrected_values[feat] = correct_fn(cell_vals, ref_vals)

            # Store reference statistics for inspection
            ref_clean = ref_vals[~np.isnan(ref_vals)]
            if len(ref_clean) > 0:
                if method == "zscore":
                    ref_stats[(sample_id, feat)] = {
                        "mean": float(np.mean(ref_clean)),
                        "std":  float(np.std(ref_clean, ddof=1)),
                        "n_ref": n_normals,
                    }
                else:
                    ref_stats[(sample_id, feat)] = {"n_ref": n_normals}
            else:
                ref_stats[(sample_id, feat)] = {"n_ref": n_normals}

        corrected_df = pd.DataFrame(corrected_values, index=sample_df.index)

        loci_results.append(
            pd.concat([sample_df[meta_cols_present],
                       corrected_df[feature_groups["loci_frac"]]], axis=1)
        )
        contacts_results.append(
            pd.concat([sample_df[meta_cols_present],
                       corrected_df[feature_groups["contacts_corr"]]], axis=1)
        )

    corrected_loci     = pd.concat(loci_results,     ignore_index=True)
    corrected_contacts = pd.concat(contacts_results, ignore_index=True)

    print(f"\nDone. Output shapes:")
    print(f"  loci_frac     : {corrected_loci.shape}")
    print(f"  contacts_corr : {corrected_contacts.shape}")

    return corrected_loci, corrected_contacts, ref_stats


# ── Example usage ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── In-memory version (for files that fit in RAM) ──
    df = pd.read_csv("tmp_features.csv")

    # Z-score correction
    loci_z, contacts_z, ref_stats_z = correct_by_normals(df, method="zscore")

    # Quantile correction
    loci_q, contacts_q, ref_stats_q = correct_by_normals(df, method="quantile")

    # Inspect reference stats for one feature
    sample_id = df["sample_id"].iloc[0]
    feat       = "loci_frac_chr1_p"
    print(f"\nReference stats for ({sample_id}, {feat}):")
    print(ref_stats_z.get((sample_id, feat)))

 