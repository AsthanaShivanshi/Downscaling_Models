#!/bin/bash
# environment.sh

export BASE_DIR="/work/FAC/FGSE/IDYST/tbeucler/downscaling"
export ENVIRONMENT="${BASE_DIR}/sasthana/MyPythonEnvNew"

module load micromamba
eval "$(micromamba shell hook --shell=bash)"
export PROJ_LIB="/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/MyPythonEnvNew/share/proj"
micromamba activate "$ENVIRONMENT"

echo $PROJ_LIB
cd "$BASE_DIR/sasthana/Downscaling/Downscaling_Models/"