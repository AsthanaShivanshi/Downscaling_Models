#!/bin/bash
#SBATCH --job-name=FM_sweep_12km
#SBATCH --output=FM_conditional_derived/logs/ckpts_FM/FM_sweep_job_output-%j.txt
#SBATCH --error=FM_conditional_derived/logs/ckpts_FM/FM_sweep_job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=256G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1



#TBR from DM dir : AsthanaSh

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

for loss in l2; do
  sbatch --job-name=FM_${loss} \
    --output=FM_conditional_derived/logs/ckpts_FM/FM_${loss}_%j.out \
    --error=FM_conditional_derived/logs/ckpts_FM/FM_${loss}_%j.err \
    --ntasks=1 --cpus-per-task=1 --time=3-00:00:00 --mem=256G --partition=gpu --gres=gpu:1 \
    --wrap="source ../diffscaler.sh && python FM_conditional_derived/train.py --config-path configs --config-name FM_bivariate_config.yaml model.loss_type=${loss}"
done