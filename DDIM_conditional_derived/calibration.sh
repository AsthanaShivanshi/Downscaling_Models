#!/bin/bash
#SBATCH --job-name=Calibration
#SBATCH --output=DDIM_conditional_derived/logs/ckpts_DDIM/etas_calibration-job_output-%j.txt
#SBATCH --error=DDIM_conditional_derived/logs/ckpts_DDIM/etas_calibration-job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --mem=256G
#SBATCH --partition=cpu
#SBATCH --gres=gpu:0

source diffscaler.sh
export PYTHONPATH="$PROJECT_DIR"

export WANDB_MODE=online
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1


which python
python -c "import wandb; print(wandb.__version__)"


python DDIM_conditional_derived/Empirical_PIT.py