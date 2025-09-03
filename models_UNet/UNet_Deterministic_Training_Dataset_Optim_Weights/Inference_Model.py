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
    EQM_DIR, SCALING_DIR, MODEL_PATH, CONFIG_PATH, ELEVATION_PATH, DATASETS_TRAINING_DIR
)

var_names = ["precip", "temp", "tmin", "tmax"]

eqm_files = {
    "precip": "precip_QM_BC_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099_r01.nc",
    "temp": "temp_QM_BC_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099_r01.nc",
    "tmin": "tmin_QM_BC_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099_r01.nc",
    "tmax": "tmax_QM_BC_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099_r01.nc"
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

    # Bicubically interp
    if var == "precip":
        #Ref files
        ref_ds = xr.open_dataset(f"{DATASETS_TRAINING_DIR}/RhiresD_step3_interp.nc")
    else:
        ref_ds = xr.open_dataset(f"{DATASETS_TRAINING_DIR}/TabsD_step3_interp.nc")
    ref_lat = ref_ds.lat.values
    ref_lon = ref_ds.lon.values

    arr_interp = xr.DataArray(
        arr,
        dims=("time", "lat", "lon"),
        coords={"time": ds.time, "lat": ds.lat, "lon": ds.lon}
    ).interp(
        lat=ref_lat, lon=ref_lon, method="cubic"
    ).values
    print(f"{var} interpolated shape: {arr_interp.shape}, min: {np.nanmin(arr_interp)}, max: {np.nanmax(arr_interp)}, mean: {np.nanmean(arr_interp)}")

    params = json.load(open(os.path.join(SCALING_DIR, scaling_param_files[var])))

    if var == "precip":
        arr_scaled = scale_precip(arr_interp, params["min"], params["max"])
    else:
        arr_scaled = scale_temp(arr_interp, params["mean"], params["std"])

    inputs_scaled.append(arr_scaled)

    if coords is None:
        coords = {
            "time": ds.time.values,
            "lat": ref_lat,
            "lon": ref_lon
        }
        eqm_lat = ref_lat
        eqm_lon = ref_lon

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
    # For correct structure
    eqm_path = os.path.join(EQM_DIR, eqm_files[var])
    ds_in = xr.open_dataset(eqm_path)

    dims = ds_in[var].dims  # ('time', 'N', 'E')
    coords = {dim: ds_in[dim] for dim in dims if dim in ds_in.coords}

    da = xr.DataArray(
        outputs_all[var],
        dims=dims,
        coords=coords,
        name=var,
        attrs=ds_in[var].attrs
    )

    ds_out = da.to_dataset()

    if "time_bnds" in ds_in.variables:
        ds_out["time_bnds"] = ds_in["time_bnds"]

    ds_out.attrs = ds_in.attrs

    out_path = os.path.join(EQM_DIR, f"TRAINING_QM_BC_{var}_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099_downscaled_r01.nc")
    ds_out.to_netcdf(out_path, mode="w")
