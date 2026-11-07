"""The 23 cell pairs: eligible tests (see 2_permutation_fdr.py) with p < 1e-3, leaving out the 10 of 33 that resolve a
single tumour state. Per patient: the fraction in the boxplots and the two marginals in the scatter plots of Fig 5B, 5D, 5E.

box_frac = co-expressed / all 50 um cell pairs. src_frac = ligand+ senders / senders with a receiver within 50 um.
tgt_frac = receptor+ receivers / receivers with a sender within 50 um. A patient's ROIs are pooled by summing.

Output: source_data/pairs23.csv, source_data/pairs23_per_patient.csv
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, './_shared')
from cohort import TUMOR_STATES, cell_pairs, load_cohort, patient_table

NB = './Notebook_4_pairs_fdr'

tests = pd.read_csv(f'{NB}/result/lr_tests.csv')
hits = tests[(tests.n_nonzero_MT >= 3) & (tests.n_nonzero_stable >= 3)
             & (np.maximum(tests.mt_mean, tests.stable_mean) > 0.05) & (tests.p_value < 1e-3)]
print('rejections:', len(hits), flush=True)
pairs = hits[~hits.source.isin(TUMOR_STATES) & ~hits.target.isin(TUMOR_STATES)]
pairs = pairs.sort_values(['p_value', 'source', 'target', 'gene_1', 'gene_2']).reset_index(drop=True)
pairs.insert(0, 'pair_id', [f'{i + 1:02d}' for i in range(len(pairs))])
pairs.to_csv(f'{NB}/source_data/pairs23.csv', index=False)

adata = load_cohort()
obs = adata.obs
celltype, roi = obs['celltype'].to_numpy(), obs['roi'].to_numpy()
xy = np.asarray(adata.obsm['spatial'], dtype=float)
patient_of_roi = dict(zip(obs['roi'], obs['patient_id']))
patients = patient_table(adata)
counts = adata.layers['counts'].tocsc()

rows = []
for p in pairs.itertuples():
    sender_pos = np.asarray(counts[:, adata.var_names.get_loc(p.gene_1)].todense()).ravel() >= 1
    receiver_pos = np.asarray(counts[:, adata.var_names.get_loc(p.gene_2)].todense()).ravel() >= 1
    total = {pid: np.zeros(6, np.int64) for pid in patients.patient_id}
    for r in pd.unique(roi):
        idx = np.where(roi == r)[0]
        s, t = idx[celltype[idx] == p.source], idx[celltype[idx] == p.target]
        if len(s) == 0 or len(t) == 0:
            continue
        i, j = cell_pairs(xy[s], xy[t])
        has_receiver, has_sender = np.zeros(len(s), bool), np.zeros(len(t), bool)
        has_receiver[i], has_sender[j] = True, True
        total[patient_of_roi[r]] += [(sender_pos[s][i] & receiver_pos[t][j]).sum(), len(i), has_receiver.sum(),
                                     has_sender.sum(), (sender_pos[s] & has_receiver).sum(),
                                     (receiver_pos[t] & has_sender).sum()]
    for q in patients.itertuples():
        n_expr, n_total, n_src, n_tgt, n_src_pos, n_tgt_pos = total[q.patient_id]
        rows.append(dict(pair_id=p.pair_id, source=p.source, gene_1=p.gene_1, target=p.target, gene_2=p.gene_2,
                         patient_id=q.patient_id, patient=q.patient, arm=q.arm,
                         box_frac=n_expr / n_total if n_total else np.nan,
                         src_frac=n_src_pos / n_src if n_src else np.nan,
                         tgt_frac=n_tgt_pos / n_tgt if n_tgt else np.nan,
                         n_expr=n_expr, n_total=n_total, n_src_elig=n_src, n_tgt_elig=n_tgt,
                         n_src_elig_pos=n_src_pos, n_tgt_elig_pos=n_tgt_pos))
pd.DataFrame(rows).to_csv(f'{NB}/source_data/pairs23_per_patient.csv', index=False)
