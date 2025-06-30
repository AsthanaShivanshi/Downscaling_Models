#!/bin/bash
#SBATCH --job-name=UNet_inference_training_dataset
#SBATCH --output=logs/FULL_training/UNet_inference_%j.out
#SBATCH --error=logs/FULL_training/UNet_inference_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

mkdir -p logs/FULL_training
module load python
source environment.sh

cd models_UNet/UNet_Deterministic_Training_Dataset

python inference.py

echo "Inference has finished, test loss printed"