"""Source data of Fig 4B, 4C and S9.

Output: source_data/risk_by_patient.csv (mean risk of each patient at each step),
        source_data/percell_risk_<section>.csv (per-cell risk, steps 0-6, sections shown in Fig 4C and S9)
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, './_shared')
from cohort import load_cohort, patient_table

NB = './Notebook_2_simulation_risk_score'
SECTIONS_SHOWN = ['4970', '2683', '2122', '3557', '2830', '5272_2', '4668', '4523']

adata = load_cohort()
obs = adata.obs
risk = np.load(f'{NB}/result/risk_score.npy')

rows = []
for p in patient_table(adata).itertuples():
    cells = (obs['patient_id'] == p.patient_id).to_numpy()
    rows += [dict(patient_id=p.patient_id, patient=p.patient, arm=p.arm, step=s, mean_risk=risk[cells, s].mean())
             for s in range(risk.shape[1])]
pd.DataFrame(rows).to_csv(f'{NB}/source_data/risk_by_patient.csv', index=False, float_format='%.6g')

for section in SECTIONS_SHOWN:
    cells = (obs['section'] == section).to_numpy()
    d = pd.DataFrame(adata.obsm['spatial'][cells], columns=['x', 'y'])
    for s in range(7):
        d[f'risk_step{s}'] = risk[cells, s]
    d.to_csv(f'{NB}/source_data/percell_risk_{section}.csv', index=False, float_format='%.6g')
