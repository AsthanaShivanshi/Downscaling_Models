#!/bin/bash
#SBATCH --job-name=L1_BILINEAR_DDIM_quadratic
#SBATCH --output=logs/ckpts_DDIM/L1_BILINEAR_DDIM_quadratic_job_output-%j.log
#SBATCH --error=logs/ckpts_DDIM/L1_BILINEAR_DDIM_quadratic_job_error-%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=256G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1


source ../diffscaler.sh
export PYTHONPATH="$PROJECT_DIR"
mkdir -p logs/ckpts_DDIM/


cd "$PROJECT_DIR"
export WANDB_MODE=online
export WANDB_START_METHOD=thread
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

which python
python -c "import wandb; print(wandb.__version__)"

python DDIM_conditional_derived/train.py --config-name DDIM_bivariate_config_bilinear.yaml 