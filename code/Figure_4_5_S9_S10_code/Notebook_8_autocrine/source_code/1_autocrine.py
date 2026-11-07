"""Autocrine only (Fig S10E): the fraction of all cells of a type that are ENPP2+ LPAR1+ and have no ENPP2+ or LPAR1+ cell of
the type within 50 um (no other cell of the type there, or only ENPP2- LPAR1- ones). A patient's ROIs are pooled by summing.

Output: source_data/autocrine_only.csv
"""
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, './_shared')
from cohort import OLIG, RADIUS, TUMOR, load_cohort, patient_table

adata = load_cohort()
obs = adata.obs
roi, celltype = obs['roi'].to_numpy(), obs['celltype'].to_numpy()
xy = np.asarray(adata.obsm['spatial'], dtype=float)
enpp2 = np.asarray(adata[:, 'ENPP2'].layers['counts'].todense()).ravel() >= 1
lpar1 = np.asarray(adata[:, 'LPAR1'].layers['counts'].todense()).ravel() >= 1

tables = []
for ct in (TUMOR, OLIG):
    total = {}
    for r in pd.unique(roi):
        idx = np.where((roi == r) & (celltype == ct))[0]
        if len(idx) == 0:
            continue
        u, v = cKDTree(xy[idx]).query_pairs(r=RADIUS, output_type='ndarray').T
        i, j = np.r_[u, v], np.r_[v, u]
        e, l = enpp2[idx], lpar1[idx]
        has_positive_neighbour = np.zeros(len(idx), bool)
        has_positive_neighbour[i[e[j] | l[j]]] = True
        p = obs['patient_id'].iloc[idx[0]]
        total[p] = total.get(p, 0) + np.array([len(idx), (e & l).sum(), (e & l & ~has_positive_neighbour).sum()])
    d = patient_table(adata).assign(celltype=ct)
    d['n_cells'] = [total[p][0] for p in d.patient_id]
    d['n_double_pos'] = [total[p][1] for p in d.patient_id]
    d['n_double_alone'] = [total[p][2] for p in d.patient_id]
    d['frac_autocrine_only'] = d.n_double_alone / d.n_cells
    tables.append(d)
pd.concat(tables).to_csv('./Notebook_8_autocrine/source_data/autocrine_only.csv', index=False)
