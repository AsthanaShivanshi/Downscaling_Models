#!/bin/bash
#SBATCH --job-name=UNet_Deterministic_Pretraining_Dataset_100_samples
#SBATCH --output=logs/job_output-%j.txt
#SBATCH --error=logs/job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
              
              
module load python

source environment.sh

#Directory containing the pipeline
cd models_UNet/UNet_Deterministic_Pretraining_Dataset

export WANDB_MODE="online"

#For quick test module uncomment
python Main.py --quick_test

#For full training module utilisation uncomment

#python Main.py