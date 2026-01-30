#!/bin/bash
#SBATCH --job-name=linear_run_DDIM_12km
#SBATCH --output=logs/ckpts_DDIM/linear_DDIM_job_output-%j.txt
#SBATCH --error=logs/ckpts_DDIM/linear_DDIM_job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
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


for beta_schedule in linear; do
  sbatch --job-name=DDIM_${beta_schedule} \
    --output=logs/ckpts_DDIM/DDIM_${beta_schedule}_%j.out \
    --error=logs/ckpts_DDIM/DDIM_${beta_schedule}_%j.err \
    --ntasks=1 --cpus-per-task=4 --time=3-00:00:00 --mem=256G --partition=gpu --gres=gpu:1 \
    --wrap="source ../diffscaler.sh && export PYTHONPATH='$PROJECT_DIR' && cd '$PROJECT_DIR' && \
      export WANDB_MODE=online && export WANDB_START_METHOD=thread && export PYTHONUNBUFFERED=1 && export HYDRA_FULL_ERROR=1 && \
      python DDIM_conditional_derived/train.py --config-name DDIM_bivariate_config.yaml model.parameterization=v model.beta_schedule=${beta_schedule}"
done