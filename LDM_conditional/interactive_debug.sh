#!/bin/bash

source ../diffscaler.sh
export PYTHONPATH="$PROJECT_DIR"

which python
python -c "import wandb; print(wandb.__version__)"

#python train_UNet_bicubic.py --config-name UNet_bivariate_config_12km_bicubic 
#python train_UNet_bilinear.py --config-name UNet_bivariate_config_12km_bilinear


#For inference 
python LDM_conditional/Inference_UNet_bicubic.py
