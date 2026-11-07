"""Risk score classifier: a random forest on the 1150-gene expression, trained on the measured tissue and the 30
simulation steps, with 5 folds that hold out whole patients. Risk = out-of-fold P(future MT) of every cell at every step.

Output: result/risk_score.npy (cells x steps 0-30, cohort order), result/clf_checkpoint/rf_fold{1-5}.joblib
"""
import os
import pickle
import sys

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, './_shared')
from cohort import MT, STABLE, load_cohort

SIM = './Notebook_2_simulation_risk_score/data/simulation'
OUT = './Notebook_2_simulation_risk_score/result'
STEPS = list(range(31))
os.makedirs(f'{OUT}/clf_checkpoint', exist_ok=True)

adata = load_cohort()
patient_of_cell = adata.obs['patient'].to_numpy()
arm = adata.obs['arm'].to_numpy()
y = (arm == MT).astype(int)
patients = sorted(np.unique(patient_of_cell).tolist())

rois = sorted(adata.obs['roi'].unique())
row_of = {name: i for i, name in enumerate(adata.obs_names)}
order = np.concatenate([[row_of[c] for c in np.load(f'{SIM}/sample_{roi}/cells.npy').astype(str)] for roi in rois])
inverse = np.empty(len(order), np.int64)
inverse[order] = np.arange(len(order))


def expression(step):
    """Cells x genes at one step, cohort order. Step 0 is the measured tissue."""
    if step == 0:
        return sp.csr_matrix(adata.X, dtype=np.float32)
    blocks = []
    for roi in rois:
        with open(f'{SIM}/sample_{roi}/step_{step}.pkl', 'rb') as f:
            blocks.append(sp.csr_matrix(pickle.load(f), dtype=np.float32))
    return sp.csr_matrix(sp.vstack(blocks).tocsr()[inverse], dtype=np.float32)


rng = np.random.default_rng(42)
fold_of = {}
for label in (MT, STABLE):
    members = sorted(set(patient_of_cell[arm == label]))
    rng.shuffle(members)
    fold_of.update({p: i % 5 for i, p in enumerate(members)})
test_cells = [np.where(np.isin(patient_of_cell, [p for p in patients if fold_of[p] == k]))[0] for k in range(5)]
train_cells = [np.where(~np.isin(patient_of_cell, [p for p in patients if fold_of[p] == k]))[0] for k in range(5)]

models = []
for k, train in enumerate(train_cells):
    rng = np.random.default_rng(42 + k)
    groups = [train[patient_of_cell[train] == p] for p in patients if (patient_of_cell[train] == p).any()]
    n = min(map(len, groups))
    X_pool, y_pool = [], []
    for step in STEPS:
        idx = np.concatenate([rng.choice(g, size=n, replace=False) for g in groups])
        rng.shuffle(idx)
        X_pool.append(expression(step)[idx].toarray().astype(np.float32))
        y_pool.append(y[idx])
    clf = RandomForestClassifier(n_estimators=300, max_depth=10, n_jobs=-1, random_state=42)
    clf.fit(np.concatenate(X_pool), np.concatenate(y_pool))
    joblib.dump(clf, f'{OUT}/clf_checkpoint/rf_fold{k + 1}.joblib')
    models.append(clf)
    print('fold', k + 1, 'trained', flush=True)

risk = np.zeros((adata.n_obs, len(STEPS)), np.float32)
for j, step in enumerate(STEPS):
    X = expression(step)
    for clf, test in zip(models, test_cells):
        risk[test, j] = clf.predict_proba(X[test].toarray().astype(np.float32))[:, 1].astype(np.float32)
np.save(f'{OUT}/risk_score.npy', risk)
print('risk', risk.shape, flush=True)
