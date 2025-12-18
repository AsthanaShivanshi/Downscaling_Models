#!/bin/bash
#SBATCH --job-name=10km_ldm_LDM
#SBATCH --output=logs/ckpts_LDM/10km/job_output-%j.txt
#SBATCH --error=logs/ckpts_LDM/10km/job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --mem=256G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source diffscaler.sh
export PYTHONPATH="$PROJECT_DIR"
mkdir -p logs/ckpts_LDM_optimised/10km


cd "$PROJECT_DIR"
export WANDB_MODE=online
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
which python
python -c "import wandb; print(wandb.__version__)"

# Training the Unet
#python LDM_conditional/train.py --config-name UNet_config.yaml

# Training the VAE
#python LDM_conditional/train.py --config-name VAE_config.yaml   

# Training the LDM
python LDM_conditional/train.py --config-name LDM_config.yaml


#hydra sweeps for multiruns

# Training the Unet with a sweep
#python LDM_conditional/train.py --multirun --config-name UNet_config.yaml \
  #experiment.batch_size=16,32 \
  #lr_scheduler.factor=0.50,0.75 \
  #model.lr=0.001,0.01 \
  #lr_scheduler.patience=5,10