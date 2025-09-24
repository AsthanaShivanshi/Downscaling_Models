#!/bin/bash
#SBATCH --job-name=ckpts_LDM
#SBATCH --output=logs/ckpts_LDM/job_output-%j.txt
#SBATCH --error=logs/ckpts_LDM/job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --mem=256G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source diffscaler.sh
export PYTHONPATH="$PROJECT_DIR"
mkdir -p logs

cd "$PROJECT_DIR"
export WANDB_MODE=online
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
which python
python -c "import wandb; print(wandb.__version__)"

# Training the Unet
#python Training_LDM/train.py --config-name UNet_config.yaml

# Training the VAE
#python Training_LDM/train.py --config-name VAE_config.yaml   

# Training the LDM
python Training_LDM/train.py --config-name LDM_config.yaml
