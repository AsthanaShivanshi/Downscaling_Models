import os
import yaml
import torch
import xarray as xr
import numpy as np
import json
from UNet import UNet
import rioxarray
from skimage.transform import resize
from directories import (
    EQM_DIR, SCALING_DIR, MODEL_PATH, CONFIG_PATH, ELEVATION_PATH
)

var_names = ["precip", "temp", "tmin", "tmax"]
eqm_files = {
    "precip": "eqm_precip_r01.nc",
    "temp": "eqm_temp_r01.nc",
    "tmin": "eqm_tmin_r01.nc",
    "tmax": "eqm_tmax_r01.nc"
}
scaling_param_map = {
    "precip": "precip",
    "temp": "temp",
    "tmin": "tmin",
    "tmax": "tmax"
}
scaling_param_files = {
    var: f"combined_{scaling_param_map[var]}_scaling_params_chronological.json"
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
    directories = yaml.safe_load(f)
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
    print(f"{var} input shape: {arr.shape}, min: {np.nanmin(arr)}, max: {np.nanmax(arr)}, mean: {np.nanmean(arr)}")
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
        eqm_lat = ds.lat.values
        eqm_lon = ds.lon.values

inputs_scaled = np.stack(inputs_scaled, axis=1)

elevation_da = rioxarray.open_rasterio(ELEVATION_PATH)
if elevation_da.ndim == 3:
    elevation_da = elevation_da.isel(band=0)
elev_array = elevation_da.values
if elev_array.shape == (len(eqm_lon), len(eqm_lat)):
    elev_array = elev_array.T
    print("Transposed elev_array shape:", elev_array.shape)
target_shape = (len(eqm_lat), inputs_scaled.shape[3])
if elev_array.shape != target_shape:
    elev_array = resize(
        elev_array,
        target_shape,
        order=1,
        preserve_range=True,
        anti_aliasing=True
    )
    print("Resized elev_array shape:", elev_array.shape)
elev_array = elev_array.astype(np.float32)
print("Final elev_array stats: min:", np.nanmin(elev_array), "max:", np.nanmax(elev_array), "mean:", np.nanmean(elev_array))

lat_1d = np.asarray(eqm_lat)
lon_1d = np.asarray(eqm_lon)
if lat_1d.ndim > 1:
    lat_1d = lat_1d[:, 0]
if lon_1d.ndim > 1:
    lon_1d = lon_1d[0, :]

n_time = inputs_scaled.shape[0]
n_lat = len(lat_1d)
n_lon = len(lon_1d)
outputs_all = {var: np.empty((n_time, n_lat, n_lon), dtype=np.float32) for var in var_names}

input_nan_mask = np.any(np.isnan(inputs_scaled), axis=1)
inputs_scaled = np.nan_to_num(inputs_scaled, nan=0.0)

for t in range(n_time):
    input_t = inputs_scaled[t]
    elev_t = elev_array[None, :, :]
    input_t = np.concatenate([input_t, elev_t], axis=0)
    print(f"Step {t}: input_t shape: {input_t.shape}, min: {np.nanmin(input_t)}, max: {np.nanmax(input_t)}, mean: {np.nanmean(input_t)}")
    input_tensor = torch.tensor(input_t[None], dtype=torch.float32)
    output = model_instance(input_tensor).cpu().detach().numpy()[0]
    print(f"Step {t}: output shape: {output.shape}, min: {np.nanmin(output)}, max: {np.nanmax(output)}, mean: {np.nanmean(output)}")

    for i, var in enumerate(var_names):
        params = json.load(open(os.path.join(SCALING_DIR, scaling_param_files[var])))
        if var == "precip":
            arr_denorm = descale_precip(output[i], params["min"], params["max"])
        else:
            arr_denorm = descale_temp(output[i], params["mean"], params["std"])
        arr_denorm[input_nan_mask[t]] = np.nan
        print(f"Step {t}, var {var}: arr_denorm shape: {arr_denorm.shape}, min: {np.nanmin(arr_denorm)}, max: {np.nanmax(arr_denorm)}, mean: {np.nanmean(arr_denorm)}")
        outputs_all[var][t] = arr_denorm

for var in var_names:
    print(f"Saving {var}: shape {outputs_all[var].shape}, min: {np.nanmin(outputs_all[var])}, max: {np.nanmax(outputs_all[var])}, mean: {np.nanmean(outputs_all[var])}")
    da = xr.DataArray(
        outputs_all[var],
        dims=("time", "lat", "lon"),
        coords={
            "time": coords["time"],
            "lat": lat_1d,
            "lon": lon_1d
        },
        name=var
    )
#lat lon have to be added as vars for cdo compatibility for downstream tasks
    ds = da.to_dataset()
    lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
    ds["lat2d"] = (("lat", "lon"), lat2d)
    ds["lon2d"] = (("lat", "lon"), lon2d)
    out_path = os.path.join(EQM_DIR, f"combined_model_eqm_{var}_downscaled_r01.nc")
    ds.to_netcdf(out_path, mode="w")
