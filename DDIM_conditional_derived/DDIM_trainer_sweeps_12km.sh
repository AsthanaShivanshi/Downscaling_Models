#!/bin/bash
#SBATCH --job-name=single_run_DDIM_12km
#SBATCH --output=logs/ckpts_DDIM/SingleRun_DDPM_job_output-%j.txt
#SBATCH --error=logs/ckpts_DDIM/SingleRun_DDPM_job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --mem=256G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source diffscaler.sh
export PYTHONPATH="$PROJECT_DIR"
mkdir -p logs/ckpts_DDIM/

cd "$PROJECT_DIR"
export WANDB_MODE=online
export WANDB_START_METHOD=thread
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

which python
python -c "import wandb; print(wandb.__version__)"



#DDPM : eta=0.0
python DDIM_conditional_derived/train.py  --config-name DDIM_bivariate_config.yaml \
  model.lr=1e-3 \
  model.timesteps=1000\
  model.beta_schedule="linear" \
  sampler.ddim_eta=0.0