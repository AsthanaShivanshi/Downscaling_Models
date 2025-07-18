#!/bin/bash
#SBATCH --job-name=temp_BC_EQM
#SBATCH --output=logs/bc/temp_job_output-%j.txt
#SBATCH --error=logs/bc/temp_job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=200G
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