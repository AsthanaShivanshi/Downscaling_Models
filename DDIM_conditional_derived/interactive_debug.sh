#!/bin/bash

source ../diffscaler.sh
export PYTHONPATH="$PROJECT_DIR"
cd "$PROJECT_DIR"

export WANDB_MODE=online
export WANDB_START_METHOD=thread
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python DDIM_conditional_derived/train.py \
  --config-name DDIM_bivariate_config.yaml \
  model.parameterization=v \
  model.beta_schedule=quadratic