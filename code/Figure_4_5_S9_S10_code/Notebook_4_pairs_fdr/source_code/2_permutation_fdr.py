"""Permutation FDR of the cell-pair tests.

A test is eligible when >= 3 MT and >= 3 stable patients have a nonzero fraction and the larger arm mean is > 0.05. The
eligible set stays fixed at the real labels; the MT and stable labels of the patients are shuffled 1000 times and every
eligible test is repeated. FDR(alpha) = mean rejections over the shuffles / rejections with the real labels.

Output: source_data/fdr_table.csv (alpha = 1e-3: R_obs 33, FDR .245; alpha = 4e-4: R_obs 21, FDR .124)
"""
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, './Notebook_4_pairs_fdr/source_code')
from fastmw import mw_two_sided

NB = './Notebook_4_pairs_fdr'
N_SHUFFLES = 1000
SEED = 20260908
ALPHAS = np.array([1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3, 2e-3, 5e-3, 1e-2, 1.5e-2, 2e-2, 3e-2, 4e-2,
                   5e-2, 7e-2, 1e-1, 1.5e-1, 2e-1])

with open(f'{NB}/result/fractions.pkl', 'rb') as f:
    celltype_pairs = pickle.load(f)['celltype_pairs']

r_obs = np.zeros(len(ALPHAS), np.int64)
v_null = np.zeros((N_SHUFFLES, len(ALPHAS)), np.int64)
n_eligible = 0
for d in celltype_pairs:
    F, is_mt = d['F'], d['is_mt']
    eligible = (((F[:, is_mt] > 0).sum(1) >= 3) & ((F[:, ~is_mt] > 0).sum(1) >= 3)
                & (np.maximum(F[:, is_mt].mean(1), F[:, ~is_mt].mean(1)) > 0.05))
    if not eligible.any():
        continue
    n_eligible += int(eligible.sum())
    r_obs += (d['p'][eligible, None] < ALPHAS).sum(0)
    tested = np.ascontiguousarray(F[eligible])
    n_patients, n_mt = F.shape[1], int(is_mt.sum())
    for s in range(N_SHUFFLES):
        permutation = np.random.default_rng([SEED, d['task'], s]).permutation(n_patients)
        shuffled = np.zeros(n_patients, bool)
        shuffled[permutation[:n_mt]] = True
        v_null[s] += (mw_two_sided(tested, shuffled)[:, None] < ALPHAS).sum(0)

table = pd.DataFrame(dict(alpha=ALPHAS, R_obs=r_obs, V_null_mean=v_null.mean(0), V_null_std=v_null.std(0), n_eligible=n_eligible))
table['FDR'] = np.where(r_obs > 0, table.V_null_mean / np.maximum(r_obs, 1), np.nan)
table.to_csv(f'{NB}/source_data/fdr_table.csv', index=False, float_format='%.6g')
print(table.to_string(index=False), flush=True)
