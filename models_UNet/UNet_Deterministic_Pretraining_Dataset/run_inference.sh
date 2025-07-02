#!/bin/bash
#SBATCH --job-name=UNet_inference_pretraining_dataset
#SBATCH --output=logs/pretraining/UNet_pretraining_inference_%j.out
#SBATCH --error=logs/pretraining/UNet_pretraining_inference_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

mkdir -p logs/pretraining
module load python
source environment.sh

cd models_UNet/UNet_Deterministic_Pretraining_Dataset

python Inference.py

echo "Inference has finished, test loss printed"