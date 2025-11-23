import os
import torch
import xarray as xr
import numpy as np
import json
import sys
import argparse
import yaml
from UNet import UNet
from directories import BASE_DIR, SCALING_DIR, ELEVATION_PATH
from skimage.transform import resize
from tqdm import trange

device = torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--validation_1981_2010', action='store_true', help='Limit downscaling to 1981-2010')
args = parser.parse_args()



def scale_temp(x, mean, std):

    return (x - mean) / std

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
    x_exp = np.exp(x_unscaled) - epsilon
    print("Precip final min/max after exp (descale):", np.nanmin(x_exp), np.nanmax(x_exp))
    return x_exp

def descale_temp(x, mean, std):
    return x * std + mean



config_path = os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/UNet_Deterministic_Training_Dataset/config.yaml")


with open(config_path) as f:
    config = yaml.safe_load(f)
input_var_names = ["precip", "temp", "tmin", "tmax"]
handle_nan = config.get("preprocessing", {}).get("nan_to_num", True)
nan_value = config.get("preprocessing", {}).get("nan_value", 0.0)

sys.path.append(os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/UNet_Deterministic_Training_Dataset"))
model_path = os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/UNet_Deterministic_Training_Dataset/trial_no_trial_best_model.pth")
training_checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model_instance = UNet(in_channels=5, out_channels=4)
model_instance.load_state_dict(training_checkpoint["model_state_dict"])
model_instance.to(device)
model_instance.eval()

dotc_input_path = os.path.join(
    BASE_DIR,
    "sasthana/Downscaling/Downscaling_Models/BC_Model_Runs/dOTC/precip_temp_tmin_tmax_bicubic_r01.nc"
)
if os.path.exists(dotc_input_path):
    dotc_input = xr.open_dataset(dotc_input_path)

    if args.validation_1981_2010:
        start_date = "1981-01-01"
        end_date = "2010-12-31"
        dotc_input = dotc_input.sel(time=slice(start_date, end_date))

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
    elevation_2d = elevation_da.values.astype(np.float32)
    elevation_2d = np.squeeze(elevation_2d)
    target_shape = dotc_input["precip"].shape[-2:]
    if elevation_2d.shape != target_shape:
        elevation_2d = resize(
            elevation_2d,
            target_shape,
            order=1,
            preserve_range=True,
            anti_aliasing=True
        ).astype(np.float32)
    elevation_da.close()

    n_time = dotc_input["precip"].shape[0]
    if args.validation_1981_2010:
        n_save = n_time
    else:
        n_save = min(50, n_time)  # Only process first 50 timesteps::: DEBUG

    precip_out = np.zeros((n_save, target_shape[0], target_shape[1]), dtype=np.float32)
    temp_out   = np.zeros_like(precip_out)
    tmin_out   = np.zeros_like(precip_out)
    tmax_out   = np.zeros_like(precip_out)

    for t in trange(n_save, desc="Downscaling frames", file=sys.stdout):
        # Get original precip and mask
        orig_precip = dotc_input["precip"].isel(time=t).values
        orig_mask = np.isnan(orig_precip)

        # Set negative precip to zero, handle NaNs
        precip = np.where(orig_precip < 0, 0, orig_precip)
        if handle_nan:
            precip = np.nan_to_num(precip, nan=nan_value)

        # Prepare other variables
        temp  = dotc_input["temp"].isel(time=t).values
        tmin  = dotc_input["tmin"].isel(time=t).values
        tmax  = dotc_input["tmax"].isel(time=t).values
        if handle_nan:
            temp  = np.nan_to_num(temp, nan=nan_value)
            tmin  = np.nan_to_num(tmin, nan=nan_value)
            tmax  = np.nan_to_num(tmax, nan=nan_value)

        elev = elevation_2d
        if elev.shape != precip.shape:
            elev = resize(elev, precip.shape, order=1, preserve_range=True, anti_aliasing=True)
        elev = elev.astype(np.float32)

        # Stack and scale
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

        
    print("precip_out shape:", precip_out.shape)
    print("N shape:", dotc_input["N"].shape)
    print("E shape:", dotc_input["E"].shape)
    print("lat shape:", dotc_input["lat"].shape)
    print("lon shape:", dotc_input["lon"].shape)
    print("time shape:", dotc_input["time"].values[:n_save].shape)
    print("precip_out min/max:", np.nanmin(precip_out), np.nanmax(precip_out))

    pred_ds = xr.Dataset(
        {
            "precip": (("time", "N", "E"), precip_out),
            "temp":   (("time", "N", "E"), temp_out),
            "tmin":   (("time", "N", "E"), tmin_out),
            "tmax":   (("time", "N", "E"), tmax_out),
        },
        coords={
            "time": dotc_input["time"].values[:n_save],
            "N": dotc_input["N"].values,
            "E": dotc_input["E"].values,
            "lat": (("N", "E"), dotc_input["lat"].values),
            "lon": (("N", "E"), dotc_input["lon"].values),
        }
    )

    if "units" in dotc_input.time.attrs:
        pred_ds["time"].attrs["units"] = dotc_input.time.attrs["units"]
    if "calendar" in dotc_input.time.attrs:
        pred_ds["time"].attrs["calendar"] = dotc_input.time.attrs["calendar"]

    if args.validation_1981_2010:
        output_filename = "dOTC_ModelRun_Downscaled_Predictions_Validation_1981_2010.nc"
    else:
        output_filename = "dOTC_ModelRun_Downscaled_Predictions_Debug_50.nc"

    pred_ds.to_netcdf(output_filename)