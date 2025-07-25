import os
import yaml
import torch
import xarray as xr
import numpy as np
import json
import sys
from UNet import UNet
import rioxarray

BASE_DIR = "/work/FAC/FGSE/IDYST/tbeucler/downscaling"
EQM_DIR = os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/BC_Model_Runs/EQM")
SCALING_DIR = os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/Training_Chronological_Dataset")
MODEL_PATH = os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/models_UNet/UNet_Deterministic_Training_Dataset_Optim_Weights/training_model_huber_weights.pth")
CONFIG_PATH = os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/models_UNet/UNet_Deterministic_Training_Dataset_Optim_Weights/config.yaml")
ELEVATION_PATH = os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/elevation.tif")

var_names = ["precip", "temp", "tmin", "tmax"]
eqm_files = {
    "precip": "eqm_precip_r01.nc",
    "temp": "eqm_temp_r01.nc",
    "tmin": "eqm_tmin_r01.nc",
    "tmax": "eqm_tmax_r01.nc"
}
scaling_param_map = {
    "precip": "RhiresD",
    "temp": "TabsD",
    "tmin": "TminD",
    "tmax": "TmaxD"
}
scaling_param_files = {
    var: f"{scaling_param_map[var]}_scaling_params_chronological.json"
    for var in var_names
}

def scale_precip(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)

def scale_temp(x, mean, std):
    return (x - mean) / std

def descale_precip(x, min_val, max_val):
    return x * (max_val - min_val) + min_val

def descale_temp(x, mean, std):
    return x * std + mean

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)
training_checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model_instance = UNet(in_channels=5, out_channels=4)
model_instance.load_state_dict(training_checkpoint["model_state_dict"])
model_instance.eval()

inputs_scaled = []
coords = None
for var in var_names:
    eqm_path = os.path.join(EQM_DIR, eqm_files[var])
    ds = xr.open_dataset(eqm_path)
    arr = ds[var].values if var in ds else ds[list(ds.data_vars)[0]].values
    params = json.load(open(os.path.join(SCALING_DIR, scaling_param_files[var])))
    if var == "precip":
        arr_scaled = scale_precip(arr, params["min"], params["max"])
    else:
        arr_scaled = scale_temp(arr, params["mean"], params["std"])
    inputs_scaled.append(arr_scaled)
    if coords is None:
        coords = {
            "time": ds.time.values,
            "lat": ds.lat.values,
            "lon": ds.lon.values
        }

inputs_scaled = np.stack(inputs_scaled, axis=1)


elevation_da = rioxarray.open_rasterio(ELEVATION_PATH)
if elevation_da.ndim == 3:
    elevation_da = elevation_da.isel(band=0)
if set(elevation_da.dims) == {"x", "y"}:
    elevation_da = elevation_da.rename({"x": "lon", "y": "lat"})

if "lat" not in elevation_da.coords or "lon" not in elevation_da.coords:
    raster_lat = np.linspace(coords["lat"].min(), coords["lat"].max(), elevation_da.shape[0])
    raster_lon = np.linspace(coords["lon"].min(), coords["lon"].max(), elevation_da.shape[1])
    elevation_da = elevation_da.assign_coords(
        lat=("lat", raster_lat),
        lon=("lon", raster_lon)
    )

if elevation_da.shape == (len(coords["lat"]), len(coords["lon"])):
    elev_resampled = elevation_da.values
else:
    elev_resampled = elevation_da.interp(
        lat=coords["lat"],
        lon=coords["lon"]
    ).values

if elev_resampled.ndim == 2:
    elev_resampled = elev_resampled[None, :, :]
elev_resampled = np.repeat(elev_resampled, inputs_scaled.shape[0], axis=0)
inputs_scaled = np.concatenate([inputs_scaled, elev_resampled[:, None, :, :]], axis=1)

# Downscaling
all_preds = []
with torch.no_grad():
    for t in range(inputs_scaled.shape[0]):
        input_tensor = torch.tensor(inputs_scaled[t:t+1], dtype=torch.float32)
        output = model_instance(input_tensor)
        all_preds.append(output.squeeze(0).cpu().numpy())
all_preds = np.stack(all_preds)


for i, var in enumerate(var_names):
    params = json.load(open(os.path.join(SCALING_DIR, scaling_param_files[var])))
    if var == "precip":
        arr_denorm = descale_precip(all_preds[:, i, :, :], params["min"], params["max"])
    else:
        arr_denorm = descale_temp(all_preds[:, i, :, :], params["mean"], params["std"])
    da = xr.DataArray(
        arr_denorm,
        dims=("time", "lat", "lon"),
        coords=coords,
        name=var
    )
    out_path = os.path.join(EQM_DIR, f"eqm_{var}_downscaled_r01.nc")
    da.to_netcdf(out_path)
    print(f"Saved downscaled {var} to {out_path}")