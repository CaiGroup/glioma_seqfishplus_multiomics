import glob
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import cKDTree
import tqdm

os.environ["PATH"] = os.path.expanduser("~/miniconda3/envs/fox_env24/bin") + ":" + os.environ["PATH"]
pd.options.mode.chained_assignment = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = "/home/jjfox/resnick_ci2_gbm_/Data/DNA_data/"
PROBE_CSV = "/resnick/groups/CaiLab/jjfox/copied_from_central/DNA_seqFISH/human/24-12-19_human_genome_DNA_seqFISH_final_probes.csv"
CELLTYPE_CSV = "/resnick/groups/mthomson/ci2_data/CaiLab/gbm/dna_distance_data/V2_24_patients_Obj_mapped.csv"
CHROM_ARM_DICT_PKL = "chrom_arm_dict.pkl"
#OUT_DIR = "../../../figures/Figure_2/Figure_2F"
OUT_DIR = os.getcwd()

CELLTYPE_FILTER = "OPC"
TARGET_CHR = "chr7"
ECDNA_NAME = "EGFR"
TERRITORY_RADIUS = 1.0  # um

files = glob.glob(f"{DATA_PATH}/*assigned.csv")
samples = [i.split('DNA_data/')[1].split('_allTranscripts')[0] for i in files]
PATIENT_IDS = samples


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
# Count EGFR dots outside chr7 territory per cell
# ---------------------------------------------------------------------------
def count_egfr_outside_territory(df, radius=TERRITORY_RADIUS):
    """For each cell, count EGFR dots farther than `radius` from any chr7 dot."""
    opc_cells = df.loc[df["predicted_celltype"] == CELLTYPE_FILTER, "matched_label"].unique()
    records = []

    for cell in opc_cells:
        df_cell = df.loc[df["matched_label"] == cell]

        # chr7 dots (assigned to chromosome)
        chr7 = df_cell.loc[df_cell["chromosome_true"] == TARGET_CHR, ["x", "y", "z"]]
        # EGFR dots (unassigned / ecDNA)
        egfr = df_cell.loc[
            (df_cell["chromosome_true"].isna()) & (df_cell["name"] == ECDNA_NAME),
            ["x", "y", "z"],
        ]

        if len(chr7) == 0 or len(egfr) == 0:
            continue

        # Build KDTree from chr7 positions, query each EGFR dot
        tree = cKDTree(chr7.values)
        dists, _ = tree.query(egfr.values)

        n_outside = int((dists > radius).sum())
        n_total = len(egfr)

        records.append({
            "cell": cell,
            "n_egfr_total": n_total,
            "n_egfr_outside": n_outside,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
chr_dict = build_chr_dict()
chrom_arm_dict = load_chrom_arm_dict()

results = []
for pid in tqdm.tqdm(PATIENT_IDS):
    try:
        df = load_patient(pid, chr_dict, chrom_arm_dict)
        counts = count_egfr_outside_territory(df)
        counts["patient"] = pid
        results.append(counts)
    except Exception as e:
        print(f"Skipping patient {pid}: {e}")
        continue

results_df = pd.concat(results, ignore_index=True)

# ---------------------------------------------------------------------------
# Violin plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(max(4, len(results_df["patient"].unique()) * 1.5), 5))
sns.violinplot(
    data=results_df,
    x="patient",
    y="n_egfr_outside",
    ax=ax,
    inner="box",
    cut=0,
)
sns.stripplot(
    data=results_df,
    x="patient",
    y="n_egfr_outside",
    ax=ax,
    color="black",
    size=3,
    alpha=0.4,
    jitter=True,
)
ax.set_xlabel("Patient")
ax.set_ylabel(f"EGFR dots outside {TARGET_CHR} territory (>{TERRITORY_RADIUS} um)")
ax.set_title(f"ecDNA outside {TARGET_CHR} territory ({CELLTYPE_FILTER} cells)")
sns.despine()
plt.tight_layout()

os.makedirs(OUT_DIR, exist_ok=True)
#fig.savefig(f"{OUT_DIR}/egfr_outside_chr7_violin.svg")
fig.savefig(f"{OUT_DIR}/egfr_outside_chr7_violin.png", dpi=300)
#plt.show()