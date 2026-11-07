"""High v low risk cells, paired over patients, every gene in every cell type (Fig 4D, 4E; partner genes of Fig S10A).

For each patient and cell type: mean log1p(CP10k) of the high minus the low cells. Tumour is split per state and
weighted by the state's cell number; a patient (tumour: a state) counts when both groups have >= 10 cells.
Two-sided Wilcoxon signed-rank over patients, BH over the genes of each cell type.

Output: source_data/de_details.csv, source_data/per_patient_gene_means.csv
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, rankdata, wilcoxon

sys.path.insert(0, './_shared')
from cohort import MT, load_cohort, patient_table

OUT = './Notebook_3_high_low_risk_DE/source_data'
MIN_CELLS = 10

adata = load_cohort()
coh = adata.obs.reset_index(drop=True)
X = adata.X.tocsr().astype(np.float64)
genes = adata.var_names.to_numpy().astype(str)
patients = patient_table(adata)
arm_of = dict(zip(patients.patient, patients.arm))
pid_of = dict(zip(patients.patient, patients.patient_id))
label = pd.read_csv(f'{OUT}/high_low_percell.csv', usecols=['high_low']).high_low.to_numpy()


def paired_test(diff):
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        return np.nan, ''
    ranks = rankdata(np.abs(nonzero))
    up, down = ranks[nonzero > 0].sum(), ranks[nonzero < 0].sum()
    direction = 'higher_in_high_risk' if up > down else 'lower_in_high_risk' if down > up else ''
    return float(wilcoxon(diff, alternative='two-sided').pvalue), direction


details, means = [], []
for celltype in sorted(coh.celltype.unique()):
    high, low, used = [], [], []
    for p in patients.patient:
        rows = np.where((coh.patient == p) & (coh.celltype == celltype))[0]
        parts = []
        for idx in coh.iloc[rows].groupby('celltype_raw').indices.values():
            r = rows[idx]
            hi, lo = r[label[r] == 'high'], r[label[r] == 'low']
            if len(hi) >= MIN_CELLS and len(lo) >= MIN_CELLS:
                parts.append((len(r), hi, lo))
        if parts:
            w = np.array([n for n, _, _ in parts], float) / sum(n for n, _, _ in parts)
            mean_high = sum(wi * np.asarray(X[hi].mean(0)).ravel() for wi, (_, hi, _) in zip(w, parts))
            mean_low = sum(wi * np.asarray(X[lo].mean(0)).ravel() for wi, (_, _, lo) in zip(w, parts))
            high.append(mean_high)
            low.append(mean_low)
            used.append(p)
            means.append(pd.DataFrame(dict(patient_id=pid_of[p], arm=arm_of[p], celltype=celltype, gene=genes,
                                           mean_high=mean_high, mean_low=mean_low)))
    if not used:
        continue
    high, low = np.vstack(high), np.vstack(low)
    diff = high - low
    tests = [paired_test(diff[:, j]) for j in range(len(genes))]
    pvalue = np.array([t[0] for t in tests])
    bh = np.full_like(pvalue, np.nan)
    bh[np.isfinite(pvalue)] = false_discovery_control(pvalue[np.isfinite(pvalue)], method='bh')
    is_mt = np.array([arm_of[u] == MT for u in used])
    details.append(pd.DataFrame(dict(gene=genes, celltype=celltype, n_patients=len(used), n_MT=int(is_mt.sum()),
                                     n_stable=int((~is_mt).sum()), n_up=(diff > 0).sum(0), n_down=(diff < 0).sum(0),
                                     mean_high=high.mean(0), mean_low=low.mean(0), median_diff=np.median(diff, 0),
                                     direction=[t[1] for t in tests], p=pvalue, p_bh=bh)))
    print(celltype, len(used), 'patients, BH < 0.05:', int((bh < 0.05).sum()), flush=True)

pd.concat(details, ignore_index=True).to_csv(f'{OUT}/de_details.csv', index=False, float_format='%.6g')
pd.concat(means, ignore_index=True).to_csv(f'{OUT}/per_patient_gene_means.csv', index=False, float_format='%.6g')
