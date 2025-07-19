#!/bin/bash
#SBATCH --job-name=temp_QDM_BC_EQM
#SBATCH --output=logs/bc/QDM_temp_job_output-%j.txt
#SBATCH --error=logs/bc/QDM_temp_job_error-%j.txt
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

echo "QDM for single cell started"
python QDM_single_cell_Zurich.py --n_jobs $SLURM_CPUS_PER_TASK
echo "QDM for single cell finished"