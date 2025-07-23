#!/bin/bash
#SBATCH --job-name=UNet_inference_combined_dataset
#SBATCH --output=logs/combined/UNet_inference_%j.out
#SBATCH --error=logs/combined/UNet_inference_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

mkdir -p logs/training
module load python
source environment.sh
cd models_UNet/UNet_Deterministic_Training_Dataset_Optim_Weights

python Inference.py

echo "Inference done, test loss printed"
