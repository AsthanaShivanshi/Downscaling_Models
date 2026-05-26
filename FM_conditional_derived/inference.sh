#!/bin/bash
#SBATCH --job-name=_Inference_Test_Set_CFM
#SBATCH --output=FM_conditional_derived/logs/inference/_Inference_Test_Set_CFM-job_output-%j.txt
#SBATCH --error=FM_conditional_derived/logs/inference/_Inference_Test_Set_CFM-job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=128G
#SBATCH --partition=gpu #Using GPU while CFM sampling
#SBATCH --gres=gpu:1

source diffscaler.sh

export PYTHONPATH="$PROJECT_DIR"

export WANDB_MODE=online
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1


which python
python -c "import wandb; print(wandb.__version__)"



#For  inference on the entire test set

python FM_conditional_derived/outputs_inference/inference_allframes_etaxx.py


#experiemntal : single frame tests 
#python FM_conditional_derived/inference_single_frame_hierarchy.py --idx 10 --num_steps 30 --num_samples 6


