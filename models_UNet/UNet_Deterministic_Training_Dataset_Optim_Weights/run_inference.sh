#!/bin/bash
#SBATCH --job-name=Single_timestep_test
#SBATCH --output=logs/model_downscaling_logs/single_timestep_%j.out
#SBATCH --error=logs/model_downscaling_logs/single_timestep_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G

mkdir -p logs/model_downscaling_logs
module load python
source environment.sh
cd models_UNet/UNet_Deterministic_Training_Dataset_Optim_Weights

python Inference_Model_Timestep.py

echo "Inference done, test loss printed"
