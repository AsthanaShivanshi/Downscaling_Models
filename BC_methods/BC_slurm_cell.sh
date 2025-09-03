#!/bin/bash
#SBATCH --job-name=EQM_3cities_TmaxD
#SBATCH --output=logs/bc/EQM_tmax_output-%j.txt
#SBATCH --error=logs/bc/EQM_tmax_job_error-%j.txt
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

echo "EQM for 2 cities started"
python EQM_single_cell_cities_Manual.py --city Zurich --lat 47.3769 --lon 8.5417
python EQM_single_cell_cities_Manual.py --city Geneva --lat 46.2044 --lon 6.1432
python EQM_single_cell_cities_Manual.py --city Locarno --lat 46.1670 --lon 8.7943
