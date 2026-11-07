"""Finetune CIFM (100M parameters, 23M cells) on all 42 samples, with no clinical labels: 20 epochs, learning rate 1e-6,
batch size 2 on each of 8 GPUs. Run with 1_finetune.sh from the root of this folder.

Output: model/ModelV2-GBMSeqFISH-Finetuned-090926 (the checkpoint is not kept in this folder)
"""
import os
import sys
import tempfile

os.environ.setdefault('PYTORCH_HIP_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
miopen = os.path.join(tempfile.gettempdir(), f"miopen-{os.environ.get('SLURM_JOB_ID', 'local')}-{os.environ.get('SLURM_PROCID', '0')}")
os.environ['MIOPEN_USER_DB_PATH'] = os.environ['MIOPEN_CUSTOM_CACHE_DIR'] = miopen
os.makedirs(miopen, exist_ok=True)

import scanpy as sc
import torch

sys.path.append('../../..')
from virtualtissue_cifm.virtualtissue.virtualtissue import VirtualTissue

base_model = os.path.abspath('../../../virtualtissue_cifm/model_checkpoints/CIFM-ModelV2#100M-DataV1#23M')
save_dir = os.path.abspath('./Notebook_2_simulation_risk_score/model/ModelV2-GBMSeqFISH-Finetuned-090926')
os.makedirs(save_dir, exist_ok=True)

adata = sc.read_h5ad('./Notebook_1_adata/adata.h5ad')
channel2ensembl = torch.load('./Notebook_1_adata/channel2ensembl.pt', weights_only=False)

model = VirtualTissue(model_name=base_model, device='cuda')
model.finetune_on_adata(adata, adata_batch_key='ROI', channel2ensembl_ids_target=channel2ensembl,
                        microenvironment_knn_radius=15, microenvironment_sampling_khop=2, num_cell_thres=100,
                        batch_size=2, num_workers=8, num_epoch=20, learning_rate=1e-6, resume_from_checkpoint=None,
                        save_finetunecheckpoint_dir=save_dir, gpu_devices=list(range(torch.cuda.device_count())))
