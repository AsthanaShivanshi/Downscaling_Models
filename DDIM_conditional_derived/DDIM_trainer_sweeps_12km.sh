#!/bin/bash
#SBATCH --job-name=param_sweep_run_DDIM_12km
#SBATCH --output=logs/ckpts_DDIM/paramSweep_DDIM_job_output-%j.txt
#SBATCH --error=logs/ckpts_DDIM/paramSweep_DDIM_job_error-%j.txt
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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

which python
python -c "import wandb; print(wandb.__version__)"


python DDIM_conditional_derived/train.py --multirun --config-name DDIM_bivariate_config.yaml \
  model.parameterization=eps,x0,v \
  model.loss_type=l2\
  model.timesteps=500,1000 \
  model.beta_schedule=linear,cosine \