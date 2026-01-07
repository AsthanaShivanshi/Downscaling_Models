#!/bin/bash
#SBATCH --job-name=Log_MAE_12km_UNet_MAE_Sweep
#SBATCH --output=logs/ckpts_LDM/UNet/Log_UNetMAESweep_job_output-%j.txt
#SBATCH --error=logs/ckpts_LDM/UNet/Log_UNetMAESweep_job_error-%j.txt
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

python LDM_conditional/train.py --multirun --config-name UNet_bivariate_config.yaml \
  model.precip_loss_weight=1.0,3.0,5.0,7.0,10.0 \
  model.use_crps_channels="[0,1]"\
  model.lr=0.001,0.01



#python LDM_conditional/train.py --multirun --config-name VAE_bivariate_config.yaml \
  #vae.latent_dim=8,16,32,64,128 \
  #vae.kl_weight=0.001,0.01,0.1
