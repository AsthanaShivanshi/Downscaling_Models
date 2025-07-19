#!/bin/bash
#SBATCH --job-name=tmax_EQM_BC_EQM
#SBATCH --output=logs/bc/EQM_tmax_job_output-%j.txt
#SBATCH --error=logs/bc/EQM_tmax_job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --mem=256G
#SBATCH --partition=cpu

module load python
source environment.sh

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd BC_methods

echo "EQM for all cells started"
python EQM_allcells.py --n_jobs $SLURM_CPUS_PER_TASK
echo "EQM for all cells finished"