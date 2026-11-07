"""ENPP2 -> LPAR1 without cell types (Fig S10D). Every cell is a sender and a receiver: a cell pair is two cells within
50 um in one ROI, counted in both directions, and each cell is also its own partner. A patient's value is the share of its
cell pairs with an ENPP2+ sender and an LPAR1+ receiver (>= 1 count each), ROIs pooled by summing.

Output: source_data/enpp2_lpar1_no_celltype.csv
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, './_shared')
from cohort import cell_pairs, load_cohort, patient_table

adata = load_cohort()
obs = adata.obs
roi = obs['roi'].to_numpy()
xy = np.asarray(adata.obsm['spatial'], dtype=float)
enpp2 = np.asarray(adata[:, 'ENPP2'].layers['counts'].todense()).ravel() >= 1
lpar1 = np.asarray(adata[:, 'LPAR1'].layers['counts'].todense()).ravel() >= 1

total = {}
for r in pd.unique(roi):
    idx = np.where(roi == r)[0]
    i, j = cell_pairs(xy[idx], xy[idx])
    p = obs['patient_id'].iloc[idx[0]]
    total[p] = total.get(p, 0) + np.array([(enpp2[idx][i] & lpar1[idx][j]).sum(), len(i)])

d = patient_table(adata)
d['n_expr'] = [total[p][0] for p in d.patient_id]
d['n_total'] = [total[p][1] for p in d.patient_id]
d['box_frac'] = d.n_expr / d.n_total
d.to_csv('./Notebook_7_no_celltype/source_data/enpp2_lpar1_no_celltype.csv', index=False)
