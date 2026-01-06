#!/bin/bash
#SBATCH --job-name=Yeo_Johnson_12km_UNet_MAE_Sweep
#SBATCH --output=logs/ckpts_LDM/UNet/UNetMAESweep_job_output-%j.txt
#SBATCH --error=logs/ckpts_LDM/UNet/UNetMAESweep_job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --mem=256G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source diffscaler.sh
export PYTHONPATH="$PROJECT_DIR"
mkdir -p logs/ckpts_LDM/UNet/

cd "$PROJECT_DIR"
export WANDB_MODE=online
export WANDB_START_METHOD=thread
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

which python
python -c "import wandb; print(wandb.__version__)"



#unet sweep :mae sweep

python LDM_conditional/train.py --multirun --config-name UNet_bivariate_config_yeo_johnson.yaml \
  model.use_crps_channels="[0,1]" \
  lr_scheduler.factor=0.50,0.75 \
  model.lr=0.001,0.01



#python LDM_conditional/train.py --multirun --config-name VAE_bivariate_config.yaml \
  #vae.latent_dim=8,16,32,64,128 \
  #vae.kl_weight=0.001,0.01,0.1
