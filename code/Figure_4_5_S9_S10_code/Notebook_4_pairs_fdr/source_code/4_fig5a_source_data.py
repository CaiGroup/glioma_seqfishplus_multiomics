"""Source data of Fig 5A, sections 4970 (future stable) and 2683 (future MT).

Per cell: position, cell type, whether it is an Olig. normal or tumour cell in an Olig. normal - tumour cell pair within
50 um (in_pair), and in a pair of an ENPP2+ Olig. normal cell and an LPAR1+ tumour cell (in_cross_pair).
log2(obs/exp): the number of Olig. normal - tumour cell pairs within 50 um over its mean after shuffling the cell-type
labels 500 times within the section.

Output: source_data/fig5a_cells.csv, source_data/fig5a_stats.csv
"""
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, './_shared')
from cohort import OLIG, RADIUS, TUMOR, cell_pairs, load_cohort

NB = './Notebook_4_pairs_fdr'
N_SHUFFLES = 500

adata = load_cohort()
cells, stats = [], []
for section in ['4970', '2683']:
    a = adata[adata.obs['section'] == section]
    xy = np.asarray(a.obsm['spatial'], dtype=float)
    celltype = a.obs['celltype'].to_numpy()
    enpp2 = np.asarray(a[:, 'ENPP2'].layers['counts'].todense()).ravel() >= 1
    lpar1 = np.asarray(a[:, 'LPAR1'].layers['counts'].todense()).ravel() >= 1

    o, t = np.where(celltype == OLIG)[0], np.where(celltype == TUMOR)[0]
    i, j = cell_pairs(xy[o], xy[t])
    i, j = o[i], t[j]
    cross = enpp2[i] & lpar1[j]
    in_pair, in_cross_pair = np.zeros(len(a), bool), np.zeros(len(a), bool)
    in_pair[np.r_[i, j]] = True
    in_cross_pair[np.r_[i[cross], j[cross]]] = True
    cells.append(pd.DataFrame(dict(section=section, x=xy[:, 0], y=xy[:, 1], celltype=celltype, ENPP2_pos=enpp2,
                                   LPAR1_pos=lpar1, in_pair=in_pair, in_cross_pair=in_cross_pair)))

    u, v = cKDTree(xy).query_pairs(r=RADIUS, output_type='ndarray').T
    rng = np.random.default_rng(0)
    expected = 0
    for _ in range(N_SHUFFLES):
        shuffled = rng.permutation(celltype)
        is_olig, is_tumor = shuffled == OLIG, shuffled == TUMOR
        expected += ((is_olig[u] & is_tumor[v]) | (is_tumor[u] & is_olig[v])).sum() / N_SHUFFLES
    stats.append(dict(section=section, n_tumor=len(t), n_olig=len(o), n_pairs=len(i), n_cross_pairs=int(cross.sum()),
                      cross_fraction=cross.mean(), log2_obs_exp=np.log2(len(i) / expected)))

pd.concat(cells, ignore_index=True).to_csv(f'{NB}/source_data/fig5a_cells.csv', index=False, float_format='%.6g')
pd.DataFrame(stats).to_csv(f'{NB}/source_data/fig5a_stats.csv', index=False, float_format='%.6g')
print(pd.DataFrame(stats).to_string(index=False), flush=True)
