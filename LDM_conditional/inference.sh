#!/bin/bash
#SBATCH --job-name=UNet_LDM_inference
#SBATCH --output=logs/ckpts_LDM/_UNet_LDM_inference-job_output-%j.txt
#SBATCH --error=logs/ckpts_LDM/_UNet_LDM_inference-job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --mem=256G
#SBATCH --partition=gpu #Using GPU while LDM sampling tbs. 
#SBATCH --gres=gpu:1

source ../diffscaler.sh
export PYTHONPATH="$PROJECT_DIR"

export WANDB_MODE=online
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1


which python
python -c "import wandb; print(wandb.__version__)"

python DDIM_conditional_derived/inference_single_frame_hierarchy.py --idx 25
python DDIM_conditional_derived/inference_single_frame_hierarchy.py --idx 10
python DDIM_conditional_derived/inference_single_frame_hierarchy.py --idx 20
python DDIM_conditional_derived/inference_single_frame_hierarchy.py --idx 5

