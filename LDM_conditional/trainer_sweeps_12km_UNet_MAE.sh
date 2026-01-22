#!/bin/bash

##EOF: heredoc for passing block to sbatch instead of creating separate scripts. 

for config in UNet_bivariate_config_24km.yaml UNet_bivariate_config_36km.yaml UNet_bivariate_config_48km.yaml; do
  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${config%.yaml}_Log_MAE_UNet_MAE_Sweep
#SBATCH --output=logs/ckpts_LDM/UNet/${config%.yaml}_Log_UNetMAESweep_job_output-%j.txt
#SBATCH --error=logs/ckpts_LDM/UNet/${config%.yaml}_Log_UNetMAESweep_job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=15:00:00
#SBATCH --mem=256G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source ../diffscaler.sh
mkdir -p logs/ckpts_LDM/UNet/${config%.yaml}

cd "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models"
export PYTHONPATH="/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models"
export WANDB_MODE=online
export WANDB_START_METHOD=thread
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

which python
python -c "import wandb; print(wandb.__version__)"


#unet sweep :mae sweep

python LDM_conditional/train.py --multirun --config-name ${config} \
  model.precip_loss_weight=5.0\
  model.use_crps_channels="[0,1]"\
  model.lr=0.001
EOF
done