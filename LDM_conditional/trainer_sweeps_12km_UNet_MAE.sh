#!/bin/bash


config="UNet_bivariate_config_12km.yaml"

for weight in 1.0 3.0 5.0 7.0; do
  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${config%.yaml}_Log_MAE_UNet_MAE_Sweep_${weight}
#SBATCH --output=logs/ckpts_LDM/UNet/${config%.yaml}_Log_UNetMAESweep_${weight}_job_output-%j.txt
#SBATCH --error=logs/ckpts_LDM/UNet/${config%.yaml}_Log_UNetMAESweep_${weight}_job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=15:00:00
#SBATCH --mem=256G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source ../diffscaler.sh
mkdir -p logs/ckpts_LDM/UNet/

cd "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models"
export PYTHONPATH="/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models"
export WANDB_MODE=online
export WANDB_START_METHOD=thread
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

which python
python -c "import wandb; print(wandb.__version__)"


#unet sweep :mae sweep

python LDM_conditional/train.py --config-name UNet_bivariate_config_12km.yaml \
  model.precip_loss_weight=${weight} \
  model.lr=0.001
EOF
done