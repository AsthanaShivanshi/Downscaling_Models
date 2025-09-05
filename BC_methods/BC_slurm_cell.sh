#!/bin/bash
#SBATCH --job-name=BC_DTC/EQM_cities_TminD
#SBATCH --output=logs/bc/BC_tmin_output-%j.txt
#SBATCH --error=logs/bc/BC_tmin_job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --partition=cpu

module load python
source environment.sh


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd BC_methods

echo "EQM for single cell started"
python EQM_single_cell_cities.py --city Zurich --lat 47.3769 --lon 8.5417

#echo "DOTC for single cell started"
#python DOTC_BC_single_cell.py --city Zurich --lat 47.3769 --lon 8.5417
