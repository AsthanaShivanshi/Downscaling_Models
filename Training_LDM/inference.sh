#!/bin/bash
#SBATCH --job-name=Testset_LDM_inference_model_run
#SBATCH --output=logs/testset_ldm_inference/job_output-%j.txt
#SBATCH --error=logs/testset_ldm_inference/job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --mem=256G
#SBATCH --partition=cpu   #Change to GPU when running model inference ####
##SBATCH --gres=gpu:1 #Used for model inference,,,,not used for test set inference ####

source diffscaler.sh
export PYTHONPATH="$PROJECT_DIR"
mkdir -p logs/testset_ldm_inference


cd "$PROJECT_DIR"
export WANDB_MODE=online
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
which python
python -c "import wandb; print(wandb.__version__)"

python Training_LDM/inference_all_frames.py --n_samples 10

#python Training_LDM/inference_model_UNet_LDM.py --n_samples 15