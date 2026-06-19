#!/bin/bash
# environment.sh

export BASE_DIR="/work/FAC/FGSE/IDYST/tbeucler/downscaling"
export ENVIRONMENT="${BASE_DIR}/sasthana/MyPythonEnvNew"

module load mamba
eval "$(mamba shell hook --shell=bash)"
export PROJ_LIB="/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/MyPythonEnvNew/share/proj"
mamba activate "$ENVIRONMENT"

echo $PROJ_LIB
cd "$BASE_DIR/sasthana/Downscaling/Downscaling_Models/"