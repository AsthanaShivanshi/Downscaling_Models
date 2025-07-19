#!/bin/bash
#SBATCH --job-name=single_gridcell_tmax_BC_EQM
#SBATCH --output=logs/bc/tmax_job_output-%j.txt
#SBATCH --error=logs/bc/tmax_job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=200G
#SBATCH --partition=cpu

module load python
source environment.sh

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK #1 in this case
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK #1 in this case, no parallelisatn
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK #1 in this case

cd BC_methods

echo "EQM for single cell started"
python EQM_single_cell_Zurich.py --n_jobs $SLURM_CPUS_PER_TASK
echo "EQM finished"