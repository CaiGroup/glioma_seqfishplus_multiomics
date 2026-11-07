"""High and low risk cells: within each patient, high = step-0 risk above the median of all the patient's cells.

Output: source_data/high_low_percell.csv
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, './_shared')
from cohort import load_cohort

adata = load_cohort()
risk = np.load('./Notebook_2_simulation_risk_score/result/risk_score.npy')[:, 0].astype(np.float64)
obs = adata.obs

d = obs[['section', 'roi', 'patient_id', 'patient', 'arm', 'celltype_raw', 'celltype']].copy()
d[['x', 'y']] = adata.obsm['spatial']
d['risk_step0'] = risk
median = pd.Series(risk).groupby(obs['patient'].to_numpy()).transform('median').to_numpy()
d['high_low'] = np.where(risk > median, 'high', 'low')
d.to_csv('./Notebook_3_high_low_risk_DE/source_data/high_low_percell.csv', index=False, float_format='%.6g')
print(d.groupby('celltype').high_low.value_counts().unstack(), flush=True)
