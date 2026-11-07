"""Cross-expression of every ligand-receptor gene pair in every cell-type pair, future MT v future stable patients.

A cell pair is a sender and a receiver within 50 um in one ROI. A patient's value is the share of its cell pairs with
>= 1 ligand count in the sender and >= 1 receptor count in the receiver, ROIs pooled by summing. 770 gene pairs and 154
ordered cell-type pairs: the ten types with tumour merged, plus each tumour state with each non-tumour type. A cell-type
pair is tested when >= 3 patients per arm have cell pairs of it. Two-sided Mann-Whitney.

Output: result/fractions.pkl (per cell-type pair: gene pairs x patients), result/lr_tests.csv (one row per test)
"""
import pickle
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, './_shared')
sys.path.insert(0, './Notebook_4_pairs_fdr/source_code')
from cohort import MT, TUMOR, TUMOR_STATES, cell_pairs, load_cohort, lr_table
from fastmw import mw_two_sided

OUT = './Notebook_4_pairs_fdr/result'

adata = load_cohort()
obs = adata.obs
genes = adata.var_names
positive = adata.layers['counts'].toarray() >= 1
xy = np.asarray(adata.obsm['spatial'], dtype=float)
roi = obs['roi'].to_numpy()
grouped, raw = obs['celltype'].to_numpy(), obs['celltype_raw'].to_numpy()
patient_of_roi = dict(zip(obs['roi'], obs['patient_id']))
arm_of = dict(zip(obs['patient_id'], obs['arm']))

lr = lr_table(set(genes))
lig, rec = genes.get_indexer(lr.ligand), genes.get_indexer(lr.receptor)
order = np.lexsort((rec, lig))
lig, rec = lig[order], rec[order]

types = sorted(set(grouped))
others = [t for t in types if t != TUMOR]
celltype_pairs = ([(a, b) for a in types for b in types] + [(a, b) for a in others for b in TUMOR_STATES]
                  + [(a, b) for a in TUMOR_STATES for b in others])

fractions, tests = [], []
for task, (sender, receiver) in enumerate(celltype_pairs, start=1):
    celltype = raw if sender in TUMOR_STATES or receiver in TUMOR_STATES else grouped
    coexpressed, n_pairs = {}, {}
    for r in pd.unique(roi):
        idx = np.where(roi == r)[0]
        s, t = idx[celltype[idx] == sender], idx[celltype[idx] == receiver]
        if len(s) == 0 or len(t) == 0:
            continue
        rows, cols = cell_pairs(xy[s], xy[t])
        if len(rows) == 0:
            continue
        adjacency = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(s), len(t)))
        counts = positive[s].astype(np.float32).T @ (adjacency @ positive[t].astype(np.float32))
        p = patient_of_roi[r]
        coexpressed[p] = coexpressed.get(p, 0) + counts.astype(np.float64)
        n_pairs[p] = n_pairs.get(p, 0) + len(rows)
    patients = sorted(coexpressed)
    is_mt = np.array([arm_of[p] == MT for p in patients], dtype=bool)
    if is_mt.sum() < 3 or (~is_mt).sum() < 3:
        print(task, sender, '->', receiver, 'not tested', flush=True)
        continue
    F = np.stack([coexpressed[p][lig, rec] / n_pairs[p] for p in patients], axis=1).astype(np.float32)
    pvalue = mw_two_sided(F, is_mt)
    fractions.append(dict(task=task, source=sender, target=receiver, patients=patients, is_mt=is_mt, F=F, p=pvalue))
    tests.append(pd.DataFrame(dict(source=sender, gene_1=genes[lig], target=receiver, gene_2=genes[rec],
                                   mt_mean=F[:, is_mt].mean(1).astype(np.float64),
                                   stable_mean=F[:, ~is_mt].mean(1).astype(np.float64),
                                   n_nonzero_MT=(F[:, is_mt] > 0).sum(1), n_nonzero_stable=(F[:, ~is_mt] > 0).sum(1),
                                   n_MT=int(is_mt.sum()), n_stable=int((~is_mt).sum()), p_value=pvalue)))
    print(task, sender, '->', receiver, flush=True)

with open(f'{OUT}/fractions.pkl', 'wb') as f:
    pickle.dump(dict(gene_1=genes[lig].to_numpy(), gene_2=genes[rec].to_numpy(), celltype_pairs=fractions), f, protocol=4)
pd.concat(tests, ignore_index=True).to_csv(f'{OUT}/lr_tests.csv', index=False)
