#!/bin/bash
#SBATCH --job-name=Plot_Frame
#SBATCH --output=DDIM_conditional_derived/logs/Metrics_Test_Set/plot_frame_model_run_%j.log
#SBATCH --error=DDIM_conditional_derived/logs/Metrics_Test_Set/plot_frame_model_run_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
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


cd $PROJECT_DIR

#python DDIM_conditional_derived/Metrics_Test_Set/cobweb/ssim_precip.py


#python DDIM_conditional_derived/Metrics_Test_Set/cobweb/ssim_temp.py



#python DDIM_conditional_derived/Metrics_Test_Set/cobweb/rmse_precip.py


#python DDIM_conditional_derived/Metrics_Test_Set/cobweb/rmse_temp.py




#Generating random samples for dates.. 


#  50 random dates in 2027

for date in $(shuf -i 1-365 -n 50); do
    day_of_year=$(printf "%03d" $date)



    plot_date=$(date -d "2027-01-01 +$((date-1)) days" "+%Y-%m-%d")
    echo "Plotting for $plot_date"
    python DDIM_conditional_derived/Metrics_Test_Set/cobweb/plot_frame_model_run.py --date "$plot_date"
done

