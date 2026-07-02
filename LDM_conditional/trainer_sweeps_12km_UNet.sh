#!/bin/bash
#SBATCH --job-name=BICUBIC_UNet_MixedLoss_Sweep
#SBATCH --output=logs/UNet/BICUBIC_UNetMixedLossSweep%j.log
#SBATCH --error=logs/UNet/BICUBIC_UNetMixedLossSweep%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01-12:00:00
#SBATCH --mem=128G
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:1

source ../diffscaler.sh
export PYTHONPATH="$PROJECT_DIR"
cd LDM_conditional

which python
python -c "import wandb; print(wandb.__version__)"

python train_UNet_bicubic.py --config-name UNet_bivariate_config_12km_bicubic 
#python train_UNet_bilinear.py --config-name UNet_bivariate_config_12km_bilinear
