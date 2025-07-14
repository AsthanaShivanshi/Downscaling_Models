#!/bin/bash
#SBATCH --job-name=UNet_Deterministic_Combined_Dataset
#SBATCH --output=logs/combined/job_output-%j.txt
#SBATCH --error=logs/combined/job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --mem=500G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -x
mkdir -p logs

echo "Loading python module"
module load python

echo "Sourcing environment"
source environment.sh
echo "Environment sourced."

cd models_UNet/UNet_Deterministic_Combined_Dataset

export WANDB_MODE="online"
export PYTHONUNBUFFERED=1
which python
python --version
echo "Starting optuna_optimisation.py..."
#python Main.py --quick_test

#For full training, remove the --quick_test flag
python optuna_optimisation.py
echo "optuna_optimisation.py finished."