"""30-step forward simulation of one ROI with the finetuned CIFM: no gene is perturbed, and at each step every cell is
re-predicted from its neighbourhood, with coordinates and graph held fixed. Dropout thresholds are estimated once, on the
measured tissue. Run with 2_simulate_30steps.sh from the root of this folder: python 2_simulate_30steps.py <roi_index> <gpu_index>

Output: data/simulation/sample_<roi>/ with cells.npy (cell order), thresholds.npy, step_{1-30}.pkl (cells x genes, csr)
"""
import os
import pickle
import sys
import tempfile

ROIS = ['2122', '2411_1', '2411_2', '2683', '2789', '2830', '3209', '3557', '4494_1', '4494_2', '3364', '4523', '4668', '4970',
        '5272_1', '5272_2', '2994_1', '2994_2', '3891_1', '3891_2', '3891_3', '3891_4']
roi, gpu = ROIS[int(sys.argv[1])], sys.argv[2]

miopen = os.path.join(tempfile.gettempdir(), f"miopen-{os.environ.get('SLURM_JOB_ID', 'local')}-{gpu}")
os.environ['MIOPEN_USER_DB_PATH'] = os.environ['MIOPEN_CUSTOM_CACHE_DIR'] = miopen
os.makedirs(miopen, exist_ok=True)

import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch

sys.path.append('../../..')
from virtualtissue_cifm.virtualtissue.virtualtissue import VirtualTissue

model_dir = os.path.abspath('./Notebook_2_simulation_risk_score/model/ModelV2-GBMSeqFISH-Finetuned-090926')
out = f'./Notebook_2_simulation_risk_score/data/simulation/sample_{roi}'
os.makedirs(out, exist_ok=True)

adata = sc.read_h5ad('./Notebook_1_adata/adata.h5ad')
adata = adata[adata.obs['ROI'].astype(str) == roi].copy()
adata.X = adata.X.tocsr().astype(np.float32)
np.save(f'{out}/cells.npy', adata.obs_names.to_numpy().astype(str))
channel2ensembl = torch.load('./Notebook_1_adata/channel2ensembl.pt', weights_only=False)

model = VirtualTissue(model_name=model_dir, device=f'cuda:{gpu}')
model.load_tissue_adata(adata.copy(), adata_batch_key='ROI', channel2ensembl_ids_target=channel2ensembl,
                        filter_unmatched_channels=False, microenvironment_knn_radius=15,
                        microenvironment_sampling_khop=2, num_cell_per_samp_thres=100)

thresholds = model.estimate_dropout_threshold_basedon_genecountperchannel(batch_size=64, num_workers=32)
thresholds[thresholds < 0.05] = 0.05
thresholds[thresholds > 0.95] = 0.95
np.save(f'{out}/thresholds.npy', thresholds)

update_all_genes = np.ones(adata.n_vars, dtype=bool)
for step in range(1, 31):
    expression = model.update_cells_insitu(batch_size=64, num_workers=32, threshold=thresholds,
                                           threshold_per_cellorchannel='channel',
                                           mask_gene_channel_to_update=update_all_genes, renormalize_transform=True,
                                           update_adata_dataset_expressions=True)
    with open(f'{out}/step_{step}.pkl', 'wb') as f:
        pickle.dump(sp.csr_matrix(expression, dtype=np.float32), f)
