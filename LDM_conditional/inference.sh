#!/bin/bash
#SBATCH --job-name=VAE_LDM_inference
#SBATCH --output=logs/ckpts_LDM/_VAE_LDM_inference-job_output-%j.txt
#SBATCH --error=logs/ckpts_LDM/_VAE_LDM_inference-job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --mem=256G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source ../diffscaler.sh
export PYTHONPATH="$PROJECT_DIR"

export WANDB_MODE=online
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

cd LDM_conditional
mkdir -p outputs


which python
python -c "import wandb; print(wandb.__version__)"

#python LDM_conditional/inference_single_frame_hierarchy.py --idx 25
#python LDM_conditional/inference_single_frame_hierarchy.py --idx 10
#python LDM_conditional/inference_single_frame_hierarchy.py --idx 20
#python LDM_conditional/inference_single_frame_hierarchy.py --idx 5

#python choosing_ckpt.py

python choosing_latentdim_SR.py