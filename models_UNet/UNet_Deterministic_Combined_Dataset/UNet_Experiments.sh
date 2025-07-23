#!/bin/bash
#SBATCH --job-name=UNet_Deterministic_Combined_Dataset
#SBATCH --output=logs/combined/job_output-%j.txt
#SBATCH --error=logs/combined/job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --mem=500G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

mkdir -p logs

echo "Loading python module"
module load python

echo "Sourcing environment"
source environment.sh
echo "Environment sourced."

cd models_UNet/UNet_Deterministic_Combined_Dataset

export WANDB_MODE=online
export PYTHONUNBUFFERED=1
which python
python --version
echo "Starting Main.py for combined dataset..."
#python #Main.py --quick_test

#For full training, remove the --quick_test flag, with 1000 samples, a smoke test
python Main.py         #Main.py
echo "Main.py for combined dataset finished."