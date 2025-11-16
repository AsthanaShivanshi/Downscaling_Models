#!/bin/bash
#SBATCH --job-name=dOTC_inference_model_run
#SBATCH --output=logs/bc_unet_inference/job_output-%j.txt
#SBATCH --error=logs/bc_unet_inference/job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --mem=256G
#SBATCH --partition=cpu #Use GPU while LDM sampling  
##SBATCH --gres=gpu:1

source diffscaler.sh
export PYTHONPATH="$PROJECT_DIR"
mkdir -p logs/bc_unet_inference


cd "$PROJECT_DIR"
export WANDB_MODE=online
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1


which python
python -c "import wandb; print(wandb.__version__)"

#python Training_LDM/inference_all_frames.py #--n_samples 10

python Training_LDM/inference_model_UNet_LDM_copy.py #--n_samples 10