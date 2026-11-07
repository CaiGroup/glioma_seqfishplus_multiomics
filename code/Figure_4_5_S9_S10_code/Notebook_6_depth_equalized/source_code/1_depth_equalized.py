"""Pair 08 (Olig. normal ENPP2 -> tumour LPAR1) before and after every ROI is brought to the same transcripts per cell (Fig S10C).

The shallowest ROI (5272_2) has 396 transcripts per cell. Each cell keeps round(transcripts x 396 / mean of its ROI) of its
transcripts, and the genes are redrawn by multivariate hypergeometric. One draw (seed 1) covers the 9 genes of the 23 pairs;
the random sequence depends on this gene set.

Output: source_data/pair08_raw_vs_equalized.csv
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, './_shared')
from cohort import cell_pairs, load_cohort, patient_table

SEED = 1


def thin(counts, depth, target, rng):
    """Multivariate hypergeometric draw of the gene columns of `counts`, cell by cell, one gene after the other."""
    out = np.zeros(counts.shape, np.int64)
    remaining_depth, remaining_target = depth.copy(), target.copy()
    for j in range(counts.shape[1]):
        c = counts[:, j]
        ok = (remaining_target > 0) & (remaining_depth > 0) & (c > 0)
        draw = rng.hypergeometric(c[ok], np.maximum(remaining_depth[ok] - c[ok], 0),
                                  np.minimum(remaining_target[ok], remaining_depth[ok]))
        out[ok, j] = draw
        remaining_target[ok] -= draw
        remaining_depth = np.maximum(remaining_depth - c, 0)
    return out


pairs = pd.read_csv('./Notebook_4_pairs_fdr/source_data/pairs23.csv', dtype={'pair_id': str})
pair = pairs[pairs.pair_id == '08'].iloc[0]
genes = sorted(set(pairs.gene_1) | set(pairs.gene_2))

adata = load_cohort()
obs = adata.obs
roi, celltype = obs['roi'].to_numpy(), obs['celltype'].to_numpy()
xy = np.asarray(adata.obsm['spatial'], dtype=float)
counts_all = adata.layers['counts'].tocsr()
depth = np.asarray(counts_all.sum(1)).ravel().astype(np.int64)
counts = counts_all[:, adata.var_names.get_indexer(genes)].toarray().astype(np.int64)

mean_depth = {r: depth[roi == r].mean() for r in pd.unique(roi)}
lowest = min(mean_depth.values())
rate = np.array([lowest / mean_depth[r] for r in roi])
target = np.minimum(np.rint(depth * rate).astype(np.int64), depth)
print('transcripts per cell:', round(lowest, 1), flush=True)
equalized = thin(counts, depth, target, np.random.default_rng(SEED))

rows = []
for version, values in (('raw', counts), ('depth_equalized', equalized)):
    sender_pos = values[:, genes.index(pair.gene_1)] >= 1
    receiver_pos = values[:, genes.index(pair.gene_2)] >= 1
    total = {}
    for r in pd.unique(roi):
        idx = np.where(roi == r)[0]
        s, t = idx[celltype[idx] == pair.source], idx[celltype[idx] == pair.target]
        if len(s) and len(t):
            i, j = cell_pairs(xy[s], xy[t])
            p = obs['patient_id'].iloc[idx[0]]
            total[p] = total.get(p, 0) + np.array([(sender_pos[s][i] & receiver_pos[t][j]).sum(), len(i)])
    for q in patient_table(adata).itertuples():
        n_expr, n_total = total[q.patient_id]
        rows.append(dict(version=version, patient_id=q.patient_id, patient=q.patient, arm=q.arm,
                         box_frac=n_expr / n_total, n_expr=n_expr, n_total=n_total))
pd.DataFrame(rows).to_csv('./Notebook_6_depth_equalized/source_data/pair08_raw_vs_equalized.csv', index=False)
