#!/bin/bash
#SBATCH --job-name=LDM_sweep12km_UNet_Sweep
#SBATCH --output=logs/ckpts_LDM_optimised/12km/LDM_sweep12km_UNet_Sweep_job_output-%j.txt
#SBATCH --error=logs/ckpts_LDM_optimised/12km/LDM_sweep12km_UNet_Sweep_job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --mem=256G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source diffscaler.sh
export PYTHONPATH="$PROJECT_DIR"
mkdir -p logs/ckpts_LDM_optimised/12km

cd "$PROJECT_DIR"
export WANDB_MODE=online
export WANDB_START_METHOD=thread
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

which python
python -c "import wandb; print(wandb.__version__)"

#ldm sweep

python LDM_conditional/train_LDM.py -m --config-name LDM_bivariate_config.yaml \
  model.parameterization=eps,v,x0 \
  sampler.schedule=linear,cosine
