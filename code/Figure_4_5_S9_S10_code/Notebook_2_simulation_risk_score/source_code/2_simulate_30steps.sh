#!/bin/bash
#SBATCH --job-name=job
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH -t 06:00:00
#SBATCH -p mi2104x
#SBATCH --array=0-5

# from the root of this folder: sbatch ./Notebook_2_simulation_risk_score/source_code/2_simulate_30steps.sh
# each array task runs 4 ROIs, one per GPU
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate CIFM
for gpu in 0 1 2 3; do
    roi=$(( SLURM_ARRAY_TASK_ID * 4 + gpu ))
    if [ "$roi" -lt 22 ]; then
        python -u ./Notebook_2_simulation_risk_score/source_code/2_simulate_30steps.py "$roi" "$gpu" &
    fi
done
wait
