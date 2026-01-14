#!/bin/bash
#SBATCH --job-name=VAE_MAE_12km_Sweep
#SBATCH --output=logs/ckpts_LDM/VAE/Log_VAEMAESweep_job_output-%j.txt
#SBATCH --error=logs/ckpts_LDM/VAE/Log_VAEMAESweep_job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=128G
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

#vae sweep :mae sweep

for latent_dim in 32 64 128; do
  for kl_weight in 0.001 0.01 0.1; do
    sbatch --job-name=VAE_ld${latent_dim}_kl${kl_weight} \
      --output=logs/ckpts_LDM/VAE/vae_ld${latent_dim}_kl${kl_weight}_%j.out \
      --error=logs/ckpts_LDM/VAE/vae_ld${latent_dim}_kl${kl_weight}_%j.err \
      --ntasks=1 --cpus-per-task=4 --time=12:00:00 --mem=128G --partition=gpu --gres=gpu:1 \
      --wrap="source ../diffscaler.sh && export PYTHONPATH='$PROJECT_DIR' && cd '$PROJECT_DIR' && \
        export WANDB_MODE=online && export WANDB_START_METHOD=thread && export PYTHONUNBUFFERED=1 && export HYDRA_FULL_ERROR=1 && \
        python LDM_conditional/train.py --config-name VAE_bivariate_config.yaml vae.latent_dim=${latent_dim} vae.kl_weight=${kl_weight}"
  done
done


#HP search based on 
#having a large enough bottleneck that your reconstruction is good. 
#having a small enough bottleneck and a large enough KL term that the z variables are not overly correlated
