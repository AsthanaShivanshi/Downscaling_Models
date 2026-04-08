#!/bin/bash
#SBATCH --job-name=denoising_steps_SR_metrics
#SBATCH --output=DDIM_conditional_derived/Metrics_Test_Set/logs/denoising_steps_SR_metrics_%j.log
#SBATCH --error=DDIM_conditional_derived/Metrics_Test_Set/logs/denoising_steps_SR_metrics_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1   
#SBATCH --mem=128G
#SBATCH --time=12:00:00

source diffscaler.sh
module load cdo


#python DDIM_conditional_derived/Metrics_Test_Set/rmse.py
python DDIM_conditional_derived/Metrics_Test_Set/ssim.py


