#!/bin/bash
#SBATCH --job-name=Single_frame_Inference_Test_Set_CFM
#SBATCH --output=FM_conditional_derived/logs/inference/Single_frame_Inference_Test_Set_CFM-job_output-%j.txt
#SBATCH --error=FM_conditional_derived/logs/inference/Single_frame_Inference_Test_Set_CFM-job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --mem=64G
#SBATCH --partition=gpu #Using GPU while DDIM sampling tbs. 
#SBATCH --gres=gpu:1
##SBATCH --array=0-29 Commented out for now #For every N_steps,, N_sample combo (for experiemnt_Nsteps_Nsamples.py) : 30 combos in total.

source diffscaler.sh

export PYTHONPATH="$PROJECT_DIR"

export WANDB_MODE=online
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1


which python
python -c "import wandb; print(wandb.__version__)"



#For normal infernece on test set
#python DDIM_conditional_derived/inference_allframes_etaxx.py

python FM_conditional_derived/inference_single_frame_hierarchy.py --idx 25 --steps 10 --num_samples 1
