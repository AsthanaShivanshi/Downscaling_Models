#!/bin/bash
#SBATCH --job-name=AllFrames_DDIM_inference
#SBATCH --output=DDIM_conditional_derived/logs/ckpts_DDIM/2samples_AllFrames_DDIM_inference-job_output-%j.txt
#SBATCH --error=DDIM_conditional_derived/logs/ckpts_DDIM/2samples_AllFrames_DDIM_inference-job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --mem=256G
#SBATCH --partition=gpu #Using GPU while DDIM sampling tbs. 
#SBATCH --gres=gpu:1

source diffscaler.sh
export PYTHONPATH="$PROJECT_DIR"

export WANDB_MODE=online
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1


which python
python -c "import wandb; print(wandb.__version__)"

#python DDIM_conditional_derived/inference_single_frame_hierarchy.py --idx 12 --sampling_steps 250 500 750 999
#python DDIM_conditional_derived/inference_single_frame_hierarchy.py --idx 25 --sampling_steps 250 500 750 999
#python DDIM_conditional_derived/inference_single_frame_hierarchy.py --idx 31 --sampling_steps 250 500 750 999

#Single run determinisitc DDIm inference ... all frames. 
python DDIM_conditional_derived/inference_allframes_eta0.py