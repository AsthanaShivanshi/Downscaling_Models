#!/bin/bash
#SBATCH --job-name=LDM_sweep12km_UNet_Sweep
#SBATCH --output=logs/ckpts_LDM/LDM/LDM_sweep12km_UNet_Sweep_job_output-%j.txt
#SBATCH --error=logs/ckpts_LDM/LDM/LDM_sweep12km_UNet_Sweep_job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --mem=256G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source ../diffscaler.sh
export PYTHONPATH="$PROJECT_DIR"
cd LDM_conditional

mkdir -p logs/ckpts_LDM/

export WANDB_MODE=online
export WANDB_START_METHOD=thread
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

which python
python -c "import wandb; print(wandb.__version__)"

python train_LDM.py --multirun --config-name LDM_bivariate_config.yaml \
  model.beta_schedule=cosine model.cosine_s=8e-3 model.loss_type=l1
python train_LDM.py --multirun --config-name LDM_bivariate_config.yaml \
  model.beta_schedule=cosine model.cosine_s=1e-2 model.loss_type=l2
python train_LDM.py --multirun --config-name LDM_bivariate_config.yaml \
  model.beta_schedule=quadratic model.linear_end=1e-3 model.loss_type=l1
python train_LDM.py --multirun --config-name LDM_bivariate_config.yaml \
  model.beta_schedule=quadratic model.linear_end=5e-3 model.loss_type=l2
python train_LDM.py --multirun --config-name LDM_bivariate_config.yaml \
  model.beta_schedule=cosine model.cosine_s=8e-3 model.loss_type=l2
python train_LDM.py --multirun --config-name LDM_bivariate_config.yaml \
  model.beta_schedule=cosine model.cosine_s=1e-2 model.loss_type=l1


# Each sweep corresponds to a different combination of beta schedule and loss type,,, -- is crucial to separate diff runs
# total runs =6

#VAE being used : VAE_levels_latentdim_64_klweight_0.01_checkpoint.ckpt