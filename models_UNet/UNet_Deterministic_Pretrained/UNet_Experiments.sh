#!/bin/bash
#SBATCH --job-name=UNet_training_100_samples_CyclicLR
#SBATCH --output=job_output-CyclicLR-%j.txt 
#SBATCH --error=job_error-CyclicLR%j.txt  
#SBATCH --ntasks=1             
#SBATCH --cpus-per-task=4     
#SBATCH --time=3-00:00:00         
#SBATCH --mem=64G  
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
              
              
module load python

source environment.sh

#Directory containing the pipeline
cd ML_models/UNet_Deterministic

export WANDB_MODE="online"

#For quick test module uncomment
python Main.py --quick_test

#For full training modzle utilisattion uncomment

#python Main.py