#!/bin/bash
#SBATCH --job-name=BC_EQM
#SBATCH --output=logs/bc/job_output-%j.txt
#SBATCH --error=logs/bc/job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --mem=500G
#SBATCH --partition=cpu

module load python
source environment.sh

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd BC_methods

echo "Starting EQM"
python EQM.py --n_jobs $SLURM_CPUS_PER_TASK
echo "EQM finished"