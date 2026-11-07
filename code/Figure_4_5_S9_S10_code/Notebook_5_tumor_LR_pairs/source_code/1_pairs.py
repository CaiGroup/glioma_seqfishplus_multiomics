"""Ligand-receptor cell pairs built from the high v low risk DE of the tumour cells, tested MT v stable (Fig S10A).

Each tumour gene with BH < 0.05 that is a ligand or receptor (NicheNet, plus ENPP2 -> LPAR1) is paired with its partner
gene in every cell type where the partner has raw p < 0.05 in the same direction. Tumour gene = ligand: tumour -> partner
type; receptor: partner type -> tumour. p is the patient-level Mann-Whitney of Notebook 4; p_bh is BH over all tested
pairs. Kept: pairs where >= 5 MT and >= 5 stable patients each have >= 100 touching cells of both cell types (cells with a
partner cell within 50 um).

Output: source_data/pairs_tested.csv
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

sys.path.insert(0, './_shared')
from cohort import MT, STABLE, TUMOR, TUMOR_STATES, cell_pairs, load_cohort, lr_table

KEY = ['source', 'gene_1', 'target', 'gene_2']
MIN_CELLS, MIN_PATIENTS = 100, 5

de = pd.read_csv('./Notebook_3_high_low_risk_DE/source_data/de_details.csv')
lr = lr_table(set(de.gene))
partner = de[de.p < 0.05].set_index(['celltype', 'gene'])
hits = de[(de.celltype == TUMOR) & (de.p_bh < 0.05)].set_index('gene')

rows = []
for gene, hit in hits.iterrows():
    for role, partner_genes in (('ligand', lr.receptor[lr.ligand == gene]), ('receptor', lr.ligand[lr.receptor == gene])):
        for other in partner_genes:
            for celltype in de.celltype.unique():
                if (celltype, other) not in partner.index or partner.loc[(celltype, other), 'direction'] != hit.direction:
                    continue
                src, lig, tgt, rec = (TUMOR, gene, celltype, other) if role == 'ligand' else (celltype, other, TUMOR, gene)
                rows.append(dict(source=src, gene_1=lig, target=tgt, gene_2=rec, tumor_gene=gene, tumor_gene_role=role,
                                 direction=hit.direction, tumor_gene_bh=hit.p_bh, partner_celltype=celltype,
                                 partner_gene=other, partner_p=partner.loc[(celltype, other), 'p']))

tests = pd.read_csv('./Notebook_4_pairs_fdr/result/lr_tests.csv').rename(columns={'p_value': 'p_mannwhitney'})
tests = tests[~tests.source.isin(TUMOR_STATES) & ~tests.target.isin(TUMOR_STATES)]
tests = tests[KEY + ['mt_mean', 'stable_mean', 'n_MT', 'n_stable', 'p_mannwhitney']]
pairs = pd.DataFrame(rows).drop_duplicates(KEY).merge(tests, on=KEY, how='left')
tested = pairs.p_mannwhitney.notna()
pairs.loc[tested, 'p_bh'] = false_discovery_control(pairs.p_mannwhitney[tested], method='bh')
pairs = pairs.sort_values(['p_mannwhitney', 'source', 'target', 'gene_1', 'gene_2']).reset_index(drop=True)
pairs.insert(0, 'pair_id', [f'{i + 1:03d}' for i in range(len(pairs))])

adata = load_cohort()
obs = adata.obs
celltype, roi = obs['celltype'].to_numpy(), obs['roi'].to_numpy()
xy = np.asarray(adata.obsm['spatial'], dtype=float)
patient_of_roi = dict(zip(obs['roi'], obs['patient_id']))
arm_of = dict(zip(obs['patient_id'], obs['arm']))

kept_patients = {}
for src, tgt in set(zip(pairs.source, pairs.target)):
    touching = {}
    for r in pd.unique(roi):
        idx = np.where(roi == r)[0]
        s, t = idx[celltype[idx] == src], idx[celltype[idx] == tgt]
        if len(s) and len(t):
            i, j = cell_pairs(xy[s], xy[t])
            touching[patient_of_roi[r]] = touching.get(patient_of_roi[r], 0) + np.array([np.unique(i).size, np.unique(j).size])
    enough = [arm_of[p] for p, n in touching.items() if n.min() >= MIN_CELLS]
    kept_patients[src, tgt] = (enough.count(MT), enough.count(STABLE))

pairs['MT_patients_cells_ge_100'] = [kept_patients[k][0] for k in zip(pairs.source, pairs.target)]
pairs['stable_patients_cells_ge_100'] = [kept_patients[k][1] for k in zip(pairs.source, pairs.target)]
keep = (pairs.MT_patients_cells_ge_100 >= MIN_PATIENTS) & (pairs.stable_patients_cells_ge_100 >= MIN_PATIENTS)
print(f'{len(pairs)} pairs, {int(tested.sum())} tested, {int(keep.sum())} kept', flush=True)
pairs[keep].to_csv('./Notebook_5_tumor_LR_pairs/source_data/pairs_tested.csv', index=False, float_format='%.6g')
