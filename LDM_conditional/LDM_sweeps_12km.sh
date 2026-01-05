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


#unet sweep :mse and huber losses 
#python LDM_conditional/train.py --multirun --config-name UNet_bivariate_Huber_MSE.yaml \
  #lr_scheduler.factor=0.50,0.75 \
  #model.lr=0.001,0.01\
  #model.huber_delta=0.2,0.6,0.8,1.0


#vae sweep
#python LDM_conditional/train.py --multirun --config-name VAE_bivariate_config.yaml \
  #model.latent_dim=8,16,32,64,128\
  #model.kl_weight=0.0001,0.001,0.01


#ldm sweep

python LDM_conditional/train_LDM.py -m --config-name LDM_bivariate_12km_config.yaml \
  model.lr=0.001,0.1 \
  model.parameterization=eps,v,x0 \
  model.noise_schedule=linear,cosine