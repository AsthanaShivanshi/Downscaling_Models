#!/bin/bash
#SBATCH --job-name=UNet_Deterministic_Training_Dataset_100_samples
#SBATCH --output=logs/job_output-%j.txt
#SBATCH --error=logs/job_error-%j.txt
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

cd /work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/models_UNet/UNet_Deterministic_Training_Dataset

export WANDB_MODE="online"

echo "Starting Main.py..."
python Main.py --quick_test

#For full training, remove the --quick_test flag
# python Main.py
echo "Main.py finished."