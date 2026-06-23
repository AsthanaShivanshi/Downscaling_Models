#!/bin/bash

source diffscaler.sh
export PYTHONPATH="$PROJECT_DIR"

MPLBACKEND=Agg python DDIM_conditional_derived/inference_single_frame_hierarchy.py --idx 25 --sampling_steps 100
