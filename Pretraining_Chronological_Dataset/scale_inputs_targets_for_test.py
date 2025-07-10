#!/usr/bin/env python3
import xarray as xr
import numpy as np
import json
import os

base_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/Pretraining_Chronological_Dataset"
combined_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/Combined_Chronological_Dataset"

def minmax_scale(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)

def meanstd_scale(x, mean, std):
    return (x - mean) / std

def scale_and_save(var, scaling_type, scaling_params_file, input_file, target_file, input_out, target_out):
    print(f"Processing {var}...")
    with open(scaling_params_file, "r") as f:
        params = json.load(f)
    # Scaling input
    ds_input = xr.open_dataset(input_file)
    ds_input_scaled = ds_input.copy()
    if scaling_type == "minmax":
        ds_input_scaled[var] = minmax_scale(ds_input[var], params["min"], params["max"])
    else:
        ds_input_scaled[var] = meanstd_scale(ds_input[var], params["mean"], params["std"])
    ds_input_scaled.to_netcdf(input_out)
    print(f"Saved scaled input: {input_out}")
    # Scaling target
    ds_target = xr.open_dataset(target_file)
    ds_target_scaled = ds_target.copy()
    if scaling_type == "minmax":
        ds_target_scaled[var] = minmax_scale(ds_target[var], params["min"], params["max"])
    else:
        ds_target_scaled[var] = meanstd_scale(ds_target[var], params["mean"], params["std"])
    ds_target_scaled.to_netcdf(target_out)
    print(f"Saved scaled target: {target_out}")

if __name__ == "__main__":
    scale_and_save(
        var="precip",
        scaling_type="minmax",
        scaling_params_file=f"{base_dir}/precip_scaling_params_chronological.json",
        input_file=f"{combined_dir}/precip_test_step3_interp.nc",
        target_file=f"{combined_dir}/precip_test_step1_latlon.nc",
        input_out=f"{base_dir}/precip_input_test_chronological_scaled.nc",
        target_out=f"{base_dir}/precip_target_test_chronological_scaled.nc"
    )
    scale_and_save(
        var="temp",
        scaling_type="meanstd",
        scaling_params_file=f"{base_dir}/temp_scaling_params_chronological.json",
        input_file=f"{combined_dir}/temp_test_step3_interp.nc",
        target_file=f"{combined_dir}/temp_test_step1_latlon.nc",
        input_out=f"{base_dir}/temp_input_test_chronological_scaled.nc",
        target_out=f"{base_dir}/temp_target_test_chronological_scaled.nc"
    )
    scale_and_save(
        var="tmin",
        scaling_type="meanstd",
        scaling_params_file=f"{base_dir}/tmin_scaling_params_chronological.json",
        input_file=f"{combined_dir}/tmin_test_step3_interp.nc",
        target_file=f"{combined_dir}/tmin_test_step1_latlon.nc",
        input_out=f"{base_dir}/tmin_input_test_chronological_scaled.nc",
        target_out=f"{base_dir}/tmin_target_test_chronological_scaled.nc"
    )
    scale_and_save(
        var="tmax",
        scaling_type="meanstd",
        scaling_params_file=f"{base_dir}/tmax_scaling_params_chronological.json",
        input_file=f"{combined_dir}/tmax_test_step3_interp.nc",
        target_file=f"{combined_dir}/tmax_test_step1_latlon.nc",
        input_out=f"{base_dir}/tmax_input_test_chronological_scaled.nc",
        target_out=f"{base_dir}/tmax_target_test_chronological_scaled.nc"
    )