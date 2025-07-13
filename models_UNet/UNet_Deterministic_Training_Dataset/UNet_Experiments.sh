#!/bin/bash
#SBATCH --job-name=Full_UNet_Deterministic_Training_Dataset
#SBATCH --output=logs/training/full/job_output-%j.txt
#SBATCH --error=logs/training/full/job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --mem=256G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

mkdir -p logs

echo "Loading python module"
module load python

echo "Sourcing environment"
source environment.sh
echo "Environment sourced."

cd models_UNet/UNet_Deterministic_Training_Dataset

export WANDB_MODE="online"
export PYTHONUNBUFFERED=1
which python
python --version
echo "Starting Optuna Optimisation..."
#python #Main.py --quick_test

#For full training, remove the --quick_test flag, with 1000 samples, a smoke test
python optuna_optimisation.py         #Main.py
echo "Optuna optimisation finished."