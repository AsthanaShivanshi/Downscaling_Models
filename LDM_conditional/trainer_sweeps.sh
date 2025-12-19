#!/bin/bash
#SBATCH --job-name=UNet_Sweep
#SBATCH --output=logs/ckpts_LDM_optimised/10km/job_output-%j.txt
#SBATCH --error=logs/ckpts_LDM_optimised/10km/job_error-%j.txt
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
export WANDB_START_METHOD=thread
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
which python
python -c "import wandb; print(wandb.__version__)"

#hydra sweeps for multiruns

# Sweep for hyperparameter tuning

#unet sweep
python LDM_conditional/train.py --multirun --config-name UNet_config.yaml \
  experiment.batch_size=16,32 \
  lr_scheduler.factor=0.50,0.75 \
  model.lr=0.001,0.01 \
  lr_scheduler.patience=5,10



#vae sweep

"""python LDM_conditional/train.py --multirun --config-name VAE_config.yaml \
  'encoder.levels=2,3,4' \
  'encoder.min_ch=16,32,64' \
  'encoder.ch_mult=4,8,16' \
  'decoder.in_dim=64,128,256' \
  'decoder.levels=2,3,4' \
  'decoder.min_ch=16,32,64' \
  model.kl_weight=0.001,0.01,0.005,0.1\
  hydra.sweeper.params="zip(encoder.levels,encoder.min_ch,encoder.ch_mult,decoder.in_dim,decoder.levels,decoder.min_ch)"""