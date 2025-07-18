#!/bin/bash
#SBATCH --job-name=BC_EQM
#SBATCH --output=logs/bc/job_output-%j.txt
#SBATCH --error=logs/bc/job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=256G
#SBATCH --partition=cpu

module load python
source environment.sh

cd BC_methods

python EQM.py
echo "Empirical Quantile Mapping finished."