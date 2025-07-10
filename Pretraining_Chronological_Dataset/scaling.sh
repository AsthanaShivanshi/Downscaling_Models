#!/bin/bash
#SBATCH --job-name=scaling
#SBATCH --chdir=/work/FAC/FGSE/IDYST/tbeucler/downscaling
#SBATCH --output=sasthana/Downscaling/Downscaling_Models/logs/scaling_%A_%a.log
#SBATCH --error=sasthana/Downscaling/Downscaling_Models/logs/scaling_%A_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=500G
#SBATCH --time=3-00:00:00

source sasthana/Downscaling/Downscaling_Models/environment.sh
export PROJ_LIB="$ENVIRONMENT/share/proj"
export HDF5_USE_FILE_LOCKING=FALSE
cd /work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/Pretraining_Chronological_Dataset
python /work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/Pretraining_Chronological_Dataset/scale_inputs_targets_for_test.py