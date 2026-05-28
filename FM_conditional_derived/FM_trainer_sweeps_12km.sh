#!/bin/bash
#SBATCH --job-name=L2_VPFM_noise_sweep_12km
#SBATCH --output=FM_conditional_derived/logs/ckpts_FM/L2_VPFM_noise_sweep_job_output-%j.txt
#SBATCH --error=FM_conditional_derived/logs/ckpts_FM/L2_VPFM_noise_sweep_job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01-23:00:00
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1


source diffscaler.sh

export PYTHONPATH=$(pwd)
export WANDB_MODE=online
export WANDB_START_METHOD=thread
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_PROJECT=FM_sweep_12km
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True



which python
python -c "import wandb; print(wandb.__version__)"


python FM_conditional_derived/train.py --config-path configs --config-name FM_bivariate_config.yaml