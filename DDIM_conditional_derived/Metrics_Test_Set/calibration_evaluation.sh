#!/bin/bash
#SBATCH --job-name=Kiviat_job
#SBATCH --output=DDIM_conditional_derived/logs/Metrics_Test_Set/kiviat_job_%j.log
#SBATCH --error=DDIM_conditional_derived/logs/Metrics_Test_Set/kiviat_job_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --mem=256G
#SBATCH --partition=cpu
#SBATCH --gres=gpu:0

source diffscaler.sh
export PYTHONPATH="$PROJECT_DIR"

export WANDB_MODE=online
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1


which python
python -c "import wandb; print(wandb.__version__)"



#python DDIM_conditional_derived/Metrics_Test_Set/Empirical_PIT_precip_gridcell.py --target_lat 47.3769 --target_lon 8.5417 --city Zurich
#python DDIM_conditional_derived/Metrics_Test_Set/Empirical_PIT_precip_gridcell.py --target_lat 46.1670 --target_lon 8.7943 --city Locarno
#python DDIM_conditional_derived/Metrics_Test_Set/Empirical_PIT_precip_gridcell.py --target_lat 46.9480 --target_lon 7.4474 --city Bern

#python DDIM_conditional_derived/Metrics_Test_Set/Empirical_PIT_temp_gridcell.py --target_lat 47.3769 --target_lon 8.5417 --city Zurich
#python DDIM_conditional_derived/Metrics_Test_Set/Empirical_PIT_temp_gridcell.py --target_lat 46.1670 --target_lon 8.7943 --city Locarno
#python DDIM_conditional_derived/Metrics_Test_Set/Empirical_PIT_temp_gridcell.py --target_lat 46.9480 --target_lon 7.4474 --city Bern

#python DDIM_conditional_derived/Metrics_Test_Set/Empirical_PIT_precip.py 
#python DDIM_conditional_derived/Metrics_Test_Set/Empirical_PIT_temp.py


python DDIM_conditional_derived/Metrics_Test_Set/csv_metrics_poster_plot_cobweb.py