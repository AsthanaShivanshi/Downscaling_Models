#!/bin/bash
#SBATCH --job-name=_Single_Frame_Inference_Test_Set_CFM
#SBATCH --output=FM_conditional_derived/logs/inference/_Single_Frame_Inference_Test_Set_CFM-job_output-%j.txt
#SBATCH --error=FM_conditional_derived/logs/inference/_Single_Frame_Inference_Test_Set_CFM-job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --mem=32G
#SBATCH --partition=gpu #Using GPU while CFM/VPFM sampling
#SBATCH --gres=gpu:1

source diffscaler.sh

export PYTHONPATH="$PROJECT_DIR"

export WANDB_MODE=online
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1


which python
python -c "import wandb; print(wandb.__version__)"



#For  inference on the entire test set

#python FM_conditional_derived/inference_allframes_fm.py


#experiemntal : single frame tests 
python FM_conditional_derived/inference_single_frame_hierarchy.py --idx 15 --num_steps 30 --num_samples 1

