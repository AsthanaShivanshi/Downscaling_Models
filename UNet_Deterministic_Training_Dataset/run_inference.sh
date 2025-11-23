#!/bin/bash
#SBATCH --job-name=UNet_inference_model_EQM
#SBATCH --output=logs/UNet_inference_model_EQM/UNet_inference_%j.out
#SBATCH --error=logs/UNet_inference_model_EQM/UNet_inference_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --partition=cpu

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G

mkdir -p logs/UNet_inference_model_EQM
module load python
source environment.sh

cd /work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/UNet_Deterministic_Training_Dataset

#python Inference_Test_Set.py


#python Inference_Model_dOTC.py --validation_1981_2010
python Inference_Model_EQM_QDM.py  --validation_1981_2010


#echo "Inference done, test set predictions saved"
echo "Inference done, model run predictions saved"