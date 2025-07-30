#!/bin/bash
#SBATCH --job-name=EQM_Zürich
#SBATCH --output=logs/bc/EQM_zürich_output-%j.txt
#SBATCH --error=logs/bc/EQM_zürich_job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --mem=128G
#SBATCH --partition=cpu

module load python
source environment.sh

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd BC_methods

echo "EQM for single cell started"
python EQM_single_cell_Zürich.py --n_jobs $SLURM_CPUS_PER_TASK
echo "EQM for single cell finished"