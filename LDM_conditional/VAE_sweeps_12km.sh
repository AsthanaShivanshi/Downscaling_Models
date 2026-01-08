#!/bin/bash
#SBATCH --job-name=VAE_MAE_12km_Sweep
#SBATCH --output=logs/ckpts_LDM/VAE/Log_VAEMAESweep_job_output-%j.txt
#SBATCH --error=logs/ckpts_LDM/VAE/Log_VAEMAESweep_job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --mem=256G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source diffscaler.sh
export PYTHONPATH="$PROJECT_DIR"
mkdir -p logs/ckpts_LDM/VAE/

cd "$PROJECT_DIR"
export WANDB_MODE=online
export WANDB_START_METHOD=thread
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

which python
python -c "import wandb; print(wandb.__version__)"

#vae sweep :mae sweep

python LDM_conditional/train.py --multirun --config-name VAE_bivariate_config.yaml \
  vae.latent_dim=16,32,64 \
  vae.kl_weight=0.001,0.01,0.1 \


#HP search based on 
#having a large enough bottleneck that your reconstruction is good. 
#having a small enough bottleneck and a large enough KL term that the z variables are not overly correlated
