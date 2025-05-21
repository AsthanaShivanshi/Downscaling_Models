#!/bin/bash
# environment.sh

ENVIRONMENT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/MyPythonEnvNew"

module load micromamba
eval "($micromamba shell hook --shell=bash)"

micromamba activate "$ENVIRONMENT"