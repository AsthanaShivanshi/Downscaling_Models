#!/bin/bash
#SBATCH --job-name=Training_Model_Downscaling
#SBATCH --output=logs/model_downscaling_logs/all_timesteps_%j.out
#SBATCH --error=logs/model_downscaling_logs/all_timesteps_%j.err
#SBATCH --time=16:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G

mkdir -p logs/model_downscaling_logs
module load python
source environment.sh
cd models_UNet/UNet_Deterministic_Training_Dataset_Optim_Weights

python Inference_Model.py

echo "Inference done, test loss printed"
