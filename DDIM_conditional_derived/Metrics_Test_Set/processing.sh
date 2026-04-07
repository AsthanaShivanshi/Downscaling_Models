#!/bin/bash
#SBATCH --job-name=denoising_steps_SR_metrics
#SBATCH --output=logs/denoising_steps_SR_metrics_%j.log
#SBATCH --error=logs/denoising_steps_SR_metrics_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1   
#SBATCH --mem=128G
#SBATCH --time=10:00:00

source diffscaler.sh
module load cdo


python rmse.py
python ssim.py

