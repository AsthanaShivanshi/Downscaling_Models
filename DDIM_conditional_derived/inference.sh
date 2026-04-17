#!/bin/bash
#SBATCH --job-name=Experiment_Inference_Time
#SBATCH --output=DDIM_conditional_derived/logs/ckpts_DDIM/Experiment_Inference_Time-job_output-%j.txt
#SBATCH --error=DDIM_conditional_derived/logs/ckpts_DDIM/Experiment_Inference_Time-job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --mem=256G
#SBATCH --partition=gpu #Using GPU while DDIM sampling tbs. 
#SBATCH --gres=gpu:1
#SBATCH --array=0-29 #For every N_steps,, N_sample combo

source diffscaler.sh
export PYTHONPATH="$PROJECT_DIR"

export WANDB_MODE=online
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1


which python
python -c "import wandb; print(wandb.__version__)"


#  S and num_samples from param_grid.txt 
PARAMS=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" DDIM_conditional_derived/param_grid.txt)
S=$(echo $PARAMS | awk '{print $1}')
NUM_SAMPLES=$(echo $PARAMS | awk '{print $2}')

python DDIM_conditional_derived/experiment_Nsteps_Nsamples.py --S $S --num_samples $NUM_SAMPLES
