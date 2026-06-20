#!/bin/bash
#SBATCH --job-name=Inference_Test_Set_DDPM
#SBATCH --output=DDIM_conditional_derived/logs/ckpts_DDPM/Inference_Test_Set_DDPM-job_output-%j.txt
#SBATCH --error=DDIM_conditional_derived/logs/ckpts_DDPM/Inference_Test_Set_DDPM-job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=15:00:00
#SBATCH --mem=128G
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:1
#SBATCH --array=0-47 #For every N_steps,, N_sample combo (for experiemnt_Nsteps_Nsamples.py) : 48 combos in total.

source diffscaler.sh
export PYTHONPATH="$PROJECT_DIR"



python -c "import wandb; print(wandb.__version__)"


PARAMS=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" DDIM_conditional_derived/param_grid.txt)
S=$(echo "$PARAMS" | awk '{print $1}')
NUM_SAMPLES=$(echo "$PARAMS" | awk '{print $2}')

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "S=$S NUM_SAMPLES=$NUM_SAMPLES"


python DDIM_conditional_derived/experiment_Nsteps_Nsamples.py --S $S --num_samples $NUM_SAMPLES


#Normal infernece
#python DDIM_conditional_derived/inference_allframes_etaxx.py
