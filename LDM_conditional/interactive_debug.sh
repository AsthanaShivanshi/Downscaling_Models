#!/bin/bash

source ../diffscaler.sh

cd /work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models || exit 1

export PYTHONPATH=/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models
export WANDB_MODE=offline
export WANDB_START_METHOD=thread
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

python LDM_conditional/train.py --config-name UNet_bivariate_config_12km 
