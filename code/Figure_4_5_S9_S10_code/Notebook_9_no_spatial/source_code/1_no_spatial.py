"""Tumour genes that separate future MT from future stable patients without spatial information (Fig S10F).

Per patient: the share of its tumour cells with >= 1 count. Two-sided Mann-Whitney, 8 v 7 patients. q: BH over all genes
that differ between patients. q_5pct: BH over those detected in >= 5% of the cohort's tumour cells (detection, both arms pooled).

Output: source_data/tumor_genes.csv, source_data/tumor_genes_per_patient.csv (the genes with q_5pct < 0.05)
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, mannwhitneyu

sys.path.insert(0, './_shared')
from cohort import MT, TUMOR, load_cohort, patient_table

OUT = './Notebook_9_no_spatial/source_data'

adata = load_cohort()
tumor = adata[adata.obs['celltype'] == TUMOR]
X = tumor.X.tocsr()
genes = tumor.var_names.to_numpy().astype(str)
pid = tumor.obs['patient_id'].to_numpy()
patients = patient_table(adata)
is_mt = (patients.arm == MT).to_numpy()

share = np.vstack([np.asarray((X[pid == p] > 0).mean(0)).ravel() for p in patients.patient_id])
pvalue = np.array([mannwhitneyu(share[is_mt, j], share[~is_mt, j]).pvalue if np.ptp(share[:, j]) > 0 else np.nan
                   for j in range(len(genes))])
d = pd.DataFrame(dict(gene=genes, MT_mean=share[is_mt].mean(0), stable_mean=share[~is_mt].mean(0),
                      detection=np.asarray((X > 0).mean(0)).ravel(), p=pvalue))
tested = d.p.notna()
d.loc[tested, 'q'] = false_discovery_control(d.p[tested], method='bh')
detected = tested & (d.detection >= 0.05)
d.loc[detected, 'q_5pct'] = false_discovery_control(d.p[detected], method='bh')
d = d.sort_values(['p', 'gene']).reset_index(drop=True)
d.to_csv(f'{OUT}/tumor_genes.csv', index=False, float_format='%.6g')

hits = d.gene[d.q_5pct < 0.05]
print(d[d.q_5pct < 0.05].to_string(index=False), flush=True)
per_patient = pd.DataFrame(share[:, np.isin(genes, hits)], columns=genes[np.isin(genes, hits)])
per_patient = pd.concat([patients, per_patient], axis=1).melt(['patient_id', 'patient', 'arm'], var_name='gene', value_name='share_positive')
per_patient.to_csv(f'{OUT}/tumor_genes_per_patient.csv', index=False, float_format='%.6g')
