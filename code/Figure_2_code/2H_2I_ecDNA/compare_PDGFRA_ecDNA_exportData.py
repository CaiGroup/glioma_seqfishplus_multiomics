import glob
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import cKDTree

os.environ["PATH"] = os.path.expanduser("~/miniconda3/envs/fox_env24/bin") + ":" + os.environ["PATH"]
pd.options.mode.chained_assignment = None
plt.rcParams['svg.fonttype'] = 'none'

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = "/home/jjfox/resnick_ci2_gbm_/Data/DNA_data/"
PROBE_CSV = "/resnick/groups/CaiLab/jjfox/copied_from_central/DNA_seqFISH/human/24-12-19_human_genome_DNA_seqFISH_final_probes.csv"
CELLTYPE_CSV = "/resnick/groups/mthomson/ci2_data/CaiLab/gbm/dna_distance_data/V2_24_patients_Obj_mapped.csv"
CHROM_ARM_DICT_PKL = "chrom_arm_dict.pkl"
OUT_DIR = "../../../figures/Figure_S2/Figure_S2F"

PATIENT_IDS = ["3883", "5597"]
CELLTYPE_FILTER = "OPC"
TARGET_CHR = "chr7"
ECDNA_NAME = "EGFR"
TERRITORY_RADIUS = 1.0  # um


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
# Count ecDNA dots outside the target chromosome territory per cell
# ---------------------------------------------------------------------------
def count_ecdna_outside_territory(df, radius=TERRITORY_RADIUS):
    """For each cell, count ecDNA dots farther than `radius` from any target-chr dot."""
    opc_cells = df.loc[df["predicted_celltype"] == CELLTYPE_FILTER, "matched_label"].unique()
    records = []

    for cell in opc_cells:
        df_cell = df.loc[df["matched_label"] == cell]

        # Target chromosome dots (assigned to chromosome)
        chrom = df_cell.loc[df_cell["chromosome_true"] == TARGET_CHR, ["x", "y", "z"]]
        # ecDNA dots (unassigned)
        ecdna = df_cell.loc[
            (df_cell["chromosome_true"].isna()) & (df_cell["name"] == ECDNA_NAME),
            ["x", "y", "z"],
        ]

        if len(chrom) == 0 or len(ecdna) == 0:
            continue

        # Build KDTree from target-chr positions, query each ecDNA dot
        tree = cKDTree(chrom.values)
        dists, _ = tree.query(ecdna.values)

        n_outside = int((dists > radius).sum())
        n_total = len(ecdna)

        records.append({
            "cell": cell,
            "n_ecdna_total": n_total,
            "n_ecdna_outside": n_outside,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
chr_dict = build_chr_dict()
chrom_arm_dict = load_chrom_arm_dict()

results = []
for pid in PATIENT_IDS:
    df = load_patient(pid, chr_dict, chrom_arm_dict)
    counts = count_ecdna_outside_territory(df)
    counts["patient"] = pid
    results.append(counts)

results_df = pd.concat(results, ignore_index=True)

# ---------------------------------------------------------------------------
# Save plotted data
# ---------------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)

results_df["celltype"] = CELLTYPE_FILTER
results_df["target_chr"] = TARGET_CHR
results_df["ecDNA_name"] = ECDNA_NAME
results_df["territory_radius_um"] = TERRITORY_RADIUS

results_df.to_csv(
    f"{OUT_DIR}/{ECDNA_NAME}_outside_{TARGET_CHR}_violin_3883_5597_v2_data.csv", index=False
)

summary_df = (
    results_df.groupby("patient")["n_ecdna_outside"]
    .agg(
        n_cells="count",
        median="median",
        mean="mean",
        std="std",
        q25=lambda s: s.quantile(0.25),
        q75=lambda s: s.quantile(0.75),
        min="min",
        max="max",
    )
    .reset_index()
)
summary_df.to_csv(
    f"{OUT_DIR}/{ECDNA_NAME}_outside_{TARGET_CHR}_violin_3883_5597_v2_summary.csv", index=False
)

# ---------------------------------------------------------------------------
# Violin plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4, 5))
sns.violinplot(
    data=results_df,
    x="patient",
    y="n_ecdna_outside",
    ax=ax,
    inner="box",
    cut=0,
)
sns.stripplot(
    data=results_df,
    x="patient",
    y="n_ecdna_outside",
    ax=ax,
    color="black",
    size=3,
    alpha=0.4,
    jitter=True,
)
ax.set_xlabel("Patient")
ax.set_ylabel(f"{ECDNA_NAME} dots outside {TARGET_CHR} territory (>{TERRITORY_RADIUS} um)")
#ax.set_title(f"ecDNA outside {TARGET_CHR} territory ({CELLTYPE_FILTER} cells)")
sns.despine()
plt.tight_layout()

fig.savefig(f"{OUT_DIR}/{ECDNA_NAME}_outside_{TARGET_CHR}4_violin_3883_5597_v2.svg")
#fig.savefig(f"{OUT_DIR}/PDGFRA_outside_chr4_violin_3883_5597_v2.png", dpi=300)
#plt.show()

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
from scipy.stats import mannwhitneyu

group_3883 = results_df.loc[results_df["patient"] == "3883", "n_ecdna_outside"]
group_5597 = results_df.loc[results_df["patient"] == "5597", "n_ecdna_outside"]

stat, pval = mannwhitneyu(group_3883, group_5597, alternative="two-sided")
print(f"Mann-Whitney U test: U={stat:.1f}, p={pval:.4e}")
print(f"  3883 (n={len(group_3883)}): median={group_3883.median():.1f}, mean={group_3883.mean():.2f}")
print(f"  5597 (n={len(group_5597)}): median={group_5597.median():.1f}, mean={group_5597.mean():.2f}")

stats_df = pd.DataFrame([{
    "comparison": "3883_vs_5597",
    "test": "mannwhitneyu_two_sided",
    "U": stat,
    "p_value": pval,
    "n_3883": len(group_3883),
    "n_5597": len(group_5597),
    "median_3883": group_3883.median(),
    "median_5597": group_5597.median(),
    "mean_3883": group_3883.mean(),
    "mean_5597": group_5597.mean(),
}])
stats_df.to_csv(
    f"{OUT_DIR}/{ECDNA_NAME}_outside_{TARGET_CHR}_violin_3883_5597_v2_stats.csv", index=False
)
