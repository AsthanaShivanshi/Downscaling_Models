import os
import torch
import xarray as xr
import numpy as np
import json
import sys
import argparse
from UNet import UNet
from directories import BASE_DIR, SCALING_DIR, ELEVATION_PATH
from skimage.transform import resize
from tqdm import trange
import yaml

device = torch.device("cpu")

# Load config for preprocessing and nan handling
config_path = os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/UNet_Deterministic_Training_Dataset/config.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)
handle_nan = config.get("preprocessing", {}).get("nan_to_num", True)
nan_value = config.get("preprocessing", {}).get("nan_value", 0.0)

parser = argparse.ArgumentParser()
parser.add_argument('--validation_1981_2010', action='store_true', help='Limit downscaling to 1981-2010')
args = parser.parse_args()

def scale_precip_log(x, mean, std, epsilon):
    print("Precip input min/max before log:", np.nanmin(x), np.nanmax(x))
    x_log = np.log(x + epsilon)
    print("Precip log min/max before standardize:", np.nanmin(x_log), np.nanmax(x_log))
    x_scaled = (x_log - mean) / std
    print("Precip scaled min/max:", np.nanmin(x_scaled), np.nanmax(x_scaled))
    return x_scaled

def descale_precip_log(x, mean, std, epsilon):
    x_unscaled = x * std + mean
    print("Precip log range before exp (descale):", np.nanmin(x_unscaled), np.nanmax(x_unscaled))
    x_unscaled = np.clip(x_unscaled, -10, 10)  # Prevent overflow
    x_exp = np.exp(x_unscaled) - epsilon
    print("Precip final min/max after exp (descale):", np.nanmin(x_exp), np.nanmax(x_exp))
    return x_exp

def scale_temp(x, mean, std):
    return (x - mean) / std

def descale_temp(x, mean, std):
    return x * std + mean

sys.path.append(os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/UNet_Deterministic_Training_Dataset"))
model_path = os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/UNet_Deterministic_Training_Dataset/training_model_huber.pth")
training_checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model_instance = UNet(in_channels=5, out_channels=4)
model_instance.load_state_dict(training_checkpoint["model_state_dict"])
model_instance.to(device)
model_instance.eval()




temp_path    = os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/BC_Model_Runs/EQM/temp_BC_bicubic_r01.nc")
precip_path  = os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/BC_Model_Runs/EQM/precip_BC_bicubic_r01.nc")
tmin_path    = os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/BC_Model_Runs/EQM/tmin_BC_bicubic_r01.nc")
tmax_path    = os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/BC_Model_Runs/EQM/tmax_BC_bicubic_r01.nc")

temp_ds   = xr.open_dataset(temp_path)
precip_ds = xr.open_dataset(precip_path)
tmin_ds   = xr.open_dataset(tmin_path)
tmax_ds   = xr.open_dataset(tmax_path)



if args.validation_1981_2010:
    start_date = "1981-01-01"
    end_date = "2010-12-31"
    temp_ds   = temp_ds.sel(time=slice(start_date, end_date))
    precip_ds = precip_ds.sel(time=slice(start_date, end_date))
    tmin_ds   = tmin_ds.sel(time=slice(start_date, end_date))
    tmax_ds   = tmax_ds.sel(time=slice(start_date, end_date))




n_time = temp_ds["temp"].shape[0]
target_shape = temp_ds["temp"].shape[-2:]

if args.validation_1981_2010:
    n_save = n_time
    output_filename = "EQM_ModelRun_Downscaled_Predictions_Validation_1981_2010.nc"
else:
    n_save = min(50, n_time)
    output_filename = "EQM_ModelRun_Downscaled_Predictions_Debug_50.nc"



precip_out = np.zeros((n_save, target_shape[0], target_shape[1]), dtype=np.float32)
temp_out   = np.zeros_like(precip_out)
tmin_out   = np.zeros_like(precip_out)
tmax_out   = np.zeros_like(precip_out)




rhiresd_params = json.load(open(os.path.join(SCALING_DIR, "RhiresD_scaling_params_chronological.json")))
tabsd_params   = json.load(open(os.path.join(SCALING_DIR, "TabsD_scaling_params_chronological.json")))
tmind_params   = json.load(open(os.path.join(SCALING_DIR, "TminD_scaling_params_chronological.json")))
tmaxd_params   = json.load(open(os.path.join(SCALING_DIR, "TmaxD_scaling_params_chronological.json")))

scaling_params = {
    "RhiresD": rhiresd_params,
    "TabsD": tabsd_params,
    "TminD": tmind_params,
    "TmaxD": tmaxd_params
}

elevation_da = xr.open_dataarray(ELEVATION_PATH)

elevation_2d = np.squeeze(elevation_da.values.astype(np.float32))
if elevation_2d.shape != target_shape:
    elevation_2d = resize(
        elevation_2d,
        target_shape,
        order=1,
        preserve_range=True,
        anti_aliasing=True
    ).astype(np.float32)
elevation_da.close()


for t in trange(n_save, desc="Downscaling frames", file=sys.stdout):

    orig_precip = precip_ds["precip"].isel(time=t).values
    orig_mask = np.isnan(orig_precip)

    # Set negative precip to zero, handle NaNs using config
    precip = np.where(orig_precip < 0, 0, orig_precip)
    if handle_nan:
        precip = np.nan_to_num(precip, nan=nan_value)

    temp  = temp_ds["temp"].isel(time=t).values
    tmin  = tmin_ds["tmin"].isel(time=t).values
    tmax  = tmax_ds["tmax"].isel(time=t).values
    if handle_nan:
        temp  = np.nan_to_num(temp, nan=nan_value)
        tmin  = np.nan_to_num(tmin, nan=nan_value)
        tmax  = np.nan_to_num(tmax, nan=nan_value)

    elev = elevation_2d
    if elev.shape != precip.shape:
        elev = resize(elev, precip.shape, order=1, preserve_range=True, anti_aliasing=True)
    elev = elev.astype(np.float32)

    rhiresd = scale_precip_log(precip, rhiresd_params["mean"], rhiresd_params["std"], rhiresd_params["epsilon"])
    tabsd   = scale_temp(temp, tabsd_params["mean"], tabsd_params["std"])
    tmind   = scale_temp(tmin, tmind_params["mean"], tmind_params["std"])
    tmaxd   = scale_temp(tmax, tmaxd_params["mean"], tmaxd_params["std"])

    input_tensor = torch.tensor(np.stack([rhiresd, tabsd, tmind, tmaxd, elev])).float()
    frame_input = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model_instance(frame_input)
        output_np = output.cpu().numpy()[0]



    # Descale and restore original NaNs
    precip_out_frame = descale_precip_log(output_np[0], **scaling_params["RhiresD"])
    precip_out_frame = np.maximum(precip_out_frame, 0)
    precip_out_frame[orig_mask] = np.nan
    precip_out[t] = precip_out_frame

    temp_out_frame = descale_temp(output_np[1], scaling_params["TabsD"]["mean"], scaling_params["TabsD"]["std"])
    temp_out_frame[orig_mask] = np.nan
    temp_out[t] = temp_out_frame

    tmin_out_frame = descale_temp(output_np[2], scaling_params["TminD"]["mean"], scaling_params["TminD"]["std"])
    tmin_out_frame[orig_mask] = np.nan
    tmin_out[t] = tmin_out_frame

    tmax_out_frame = descale_temp(output_np[3], scaling_params["TmaxD"]["mean"], scaling_params["TmaxD"]["std"])
    tmax_out_frame[orig_mask] = np.nan
    tmax_out[t] = tmax_out_frame



var_names = ["precip", "temp", "tmin", "tmax"]
out_arrays = [precip_out, temp_out, tmin_out, tmax_out]

pred_vars = {}
for i, var in enumerate(var_names):
    pred_vars[var] = xr.DataArray(
        out_arrays[i],
        dims=("time", "N", "E"),
        coords={
            "time": temp_ds.time.values[:n_save],
            "N": temp_ds.N.values,
            "E": temp_ds.E.values,
            "lat": (("N", "E"), temp_ds.lat.values),
            "lon": (("N", "E"), temp_ds.lon.values),
        },
        name=var
    )


pred_ds = xr.Dataset(pred_vars)

if "units" in temp_ds.time.attrs:
    pred_ds["time"].attrs["units"] = temp_ds.time.attrs["units"]
if "calendar" in temp_ds.time.attrs:
    pred_ds["time"].attrs["calendar"] = temp_ds.time.attrs["calendar"]

pred_ds.to_netcdf(output_filename)