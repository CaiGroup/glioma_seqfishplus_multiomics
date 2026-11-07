#!/bin/bash
#SBATCH --job-name=job
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH -t 1-12:00:00
#SBATCH -p mi3258x

# from the root of this folder: sbatch ./Notebook_2_simulation_risk_score/source_code/1_finetune.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate CIFM
srun --kill-on-bad-exit=1 python -u ./Notebook_2_simulation_risk_score/source_code/1_finetune.py
