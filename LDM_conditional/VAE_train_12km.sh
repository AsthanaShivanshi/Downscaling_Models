#!/bin/bash
#SBATCH --job-name=VAE_MAE_12km_Sweep
#SBATCH --output=logs/ckpts_LDM/VAE/Log_VAEMAESweep_job_output-%j.txt
#SBATCH --error=logs/ckpts_LDM/VAE/Log_VAEMAESweep_job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source ../diffscaler.sh

export PYTHONPATH="$PROJECT_DIR"
mkdir -p logs/ckpts_LDM/VAE/

cd "$PROJECT_DIR"
export WANDB_MODE=online
export WANDB_START_METHOD=thread
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

which python

python -c "import wandb; print(wandb.__version__)"

python train.py -m --config-name VAE_bivariate_config vae.lr=0.001,0.01,0.0001 vae.kl_weight=0.01,0.001