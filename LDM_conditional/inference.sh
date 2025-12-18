#!/bin/bash
#SBATCH --job-name=single_run_inference_testset_LDM
#SBATCH --output=logs/ckpts_LDM/single_LDM_testset_inference-job_output-%j.txt
#SBATCH --error=logs/ckpts_LDM/single_LDM_testset_inference-job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --mem=256G
#SBATCH --partition=gpu #Use GPU while LDM sampling  
#SBATCH --gres=gpu:1

source diffscaler.sh
export PYTHONPATH="$PROJECT_DIR"
mkdir -p logs/ckpts_LDM

export WANDB_MODE=online
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1


which python
python -c "import wandb; print(wandb.__version__)"

python LDM_conditional/inference_all_frames.py
