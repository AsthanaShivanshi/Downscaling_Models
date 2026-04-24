#!/bin/bash
#SBATCH --job-name=RMSE_SSIM_Metrics_Test_Set
#SBATCH --output=DDIM_conditional_derived/logs/Metrics_Test_Set/rmse_ssim_test_set_job_%j.log
#SBATCH --error=DDIM_conditional_derived/logs/Metrics_Test_Set/rmse_ssim_test_set_job_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --partition=cpu
#SBATCH --gres=gpu:0

source diffscaler.sh
export PYTHONPATH="$PROJECT_DIR"
module load cdo
export WANDB_MODE=online
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1


which python
python -c "import wandb; print(wandb.__version__)"




python DDIM_conditional_derived/Metrics_Test_Set/cobweb/ssim_precip.py
python DDIM_conditional_derived/Metrics_Test_Set/cobweb/ssim_temp.py



python DDIM_conditional_derived/Metrics_Test_Set/cobweb/rmse_precip.py
python DDIM_conditional_derived/Metrics_Test_Set/cobweb/rmse_temp.py



