"""The discovery cohort: 18 sections (22 ROIs) of 15 patients, 8 future MT and 7 future stable.

Spatial neighbours are found within one ROI; a patient's ROIs are combined afterwards by summing counts.
Run every script from the root of this folder.
"""
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial import cKDTree

ADATA = './Notebook_1_adata/adata.h5ad'
LR_DATABASE = './Notebook_4_pairs_fdr/data/Nichenetr_database.csv'
MT, STABLE = 'LGG_MT', 'LGG_noMT'
SECTIONS = ['2122', '2411_1', '2411_2', '2683', '2789', '2830', '3209', '3557', '4494_1', '4494_2',
            '3364', '4523', '4668', '4970', '5272_1', '5272_2', '2994', '3891']
TUMOR_STATES = ['OPC', 'OPC.like Prolif.', 'AC.like']
TUMOR, OLIG = 'tumor', 'Olig. normal'
RADIUS = 50.0


def load_cohort():
    """Cohort cells in h5ad order, with section, roi, patient_id, arm, celltype_raw and celltype (tumour states merged)."""
    adata = sc.read_h5ad(ADATA)
    adata = adata[adata.obs['ID'].astype(str).isin(SECTIONS)].copy()
    obs = adata.obs
    obs['section'] = obs['ID'].astype(str)
    obs['roi'] = obs['ROI'].astype(str)
    obs['patient'] = obs['patient'].astype(str)
    obs['patient_id'] = obs['section'].str.split('_').str[0]
    obs['arm'] = obs['type_1'].astype(str)
    obs['celltype_raw'] = obs['predicted.celltype'].astype(str)
    obs['celltype'] = obs['celltype_raw'].where(~obs['celltype_raw'].isin(TUMOR_STATES), TUMOR)
    return adata


def patient_table(adata):
    """One row per patient, MT first, sorted by patient_id within each arm."""
    patients = adata.obs[['patient_id', 'patient', 'arm']].drop_duplicates()
    return patients.sort_values(['arm', 'patient_id']).reset_index(drop=True)


def lr_table(genes):
    """NicheNet ligand -> receptor pairs with both genes in the panel, plus ENPP2 -> LPAR1."""
    lr = pd.read_csv(LR_DATABASE)[['from', 'to']].drop_duplicates()
    lr = lr[lr['from'].isin(genes) & lr['to'].isin(genes)]
    lr = pd.concat([lr, pd.DataFrame([['ENPP2', 'LPAR1']], columns=['from', 'to'])]).drop_duplicates()
    return lr.rename(columns={'from': 'ligand', 'to': 'receptor'}).reset_index(drop=True)


def cell_pairs(xy_senders, xy_receivers):
    """Sender and receiver index of every cell pair within 50 um; a cell is its own neighbour if the two sets are one."""
    neighbours = cKDTree(xy_receivers).query_ball_point(xy_senders, r=RADIUS)
    n = np.fromiter(map(len, neighbours), np.int64, len(neighbours))
    if n.sum() == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.repeat(np.arange(len(xy_senders)), n), np.concatenate(neighbours).astype(np.int64)
