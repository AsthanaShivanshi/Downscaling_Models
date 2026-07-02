#!/bin/bash
#SBATCH --job-name=BICUBIC_UNet_inference
#SBATCH --output=logs/UNet/UNet_inference_BICUBIC_job_output-%j.log
#SBATCH --error=logs/UNet/UNet_inference_BICUBIC_job_error-%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source ../diffscaler.sh
export PYTHONPATH="$PROJECT_DIR"

export WANDB_MODE=online
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

which python
python -c "import wandb; print(wandb.__version__)"

#python LDM_conditional/Inference_UNet_bicubic.py

python LDM_conditional/Inference_UNet_bilinear.py
