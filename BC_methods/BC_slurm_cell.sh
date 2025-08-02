#!/bin/bash
#SBATCH --job-name=EQM_3cities_Manual
#SBATCH --output=logs/bc/EQM_3cities_output-%j.txt
#SBATCH --error=logs/bc/EQM_3cities_job_error-%j.txt
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

echo "EQM for 2 cities started"
python EQM_single_cell_2cities_Manual.py --n_jobs $SLURM_CPUS_PER_TASK
echo "EQM for 2 cities finished using SBCK method"