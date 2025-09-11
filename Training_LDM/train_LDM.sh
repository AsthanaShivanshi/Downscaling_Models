#!/bin/bash
#SBATCH --job-name=LDm_res_training
#SBATCH --output=logs/training_LDm_res/job_output-%j.txt
#SBATCH --error=logs/training_LDm_res/job_error-%j.txt
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

eval "$(micromamba shell hook -s bash)"
micromamba activate /work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/.micromamba/envs/diffscaler

cd Training_LDM

export WANDB_MODE=online
export PYTHONUNBUFFERED=1
which python
python --version

python train.py  
