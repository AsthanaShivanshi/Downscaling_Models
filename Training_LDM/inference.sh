#!/bin/bash
#SBATCH --job-name=LDM_inference_testset
#SBATCH --output=logs/testset_inference/job_output-%j.txt
#SBATCH --error=logs/testset_inference/job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=23:00:00
#SBATCH --mem=256G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source diffscaler.sh
export PYTHONPATH="$PROJECT_DIR"
mkdir -p logs/testset_inference

cd "$PROJECT_DIR"
export WANDB_MODE=online
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
which python
python -c "import wandb; print(wandb.__version__)"

python Training_LDM/inference_all_frames.py --n_samples 20 --output testset_samples_LDM_high_res.npz