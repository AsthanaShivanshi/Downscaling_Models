#!/bin/bash
#SBATCH --job-name=DASK_AllCells
#SBATCH --output=logs/bc/DASK_AllCells_output-%j.txt
#SBATCH --error=logs/bc/DASK_AllCells_job_error-%j.txt
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

echo "EQM for All Cells started"
python EQM_allcells_DASK.py --n_jobs $SLURM_CPUS_PER_TASK
echo "EQM for All Cells finished"