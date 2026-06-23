#!/bin/bash

config="UNet_bivariate_config_12km"
mkdir -p logs/UNet/

for weight in 1.0 3.0 5.0; do
  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${config}_Log_UNet_MixedLoss_Sweep_${weight}
#SBATCH --output=logs/UNet/${config}_Log_UNetMixedLossSweep_${weight}_job_output-%j.txt
#SBATCH --error=logs/UNet/${config}_Log_UNetMixedLossSweep_${weight}_job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --mem=128G
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:1

source ../diffscaler.sh

cd "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models"

export PYTHONPATH="/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models"
export WANDB_MODE=online
export WANDB_START_METHOD=thread
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

which python
python -c "import wandb; print(wandb.__version__)"

python LDM_conditional/train.py --config-name ${config} \
  model.precip_loss_weight=${weight}
EOF
done