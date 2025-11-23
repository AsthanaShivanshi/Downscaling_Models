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

device = torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--validation_1981_2010', action='store_true', help='Limit downscaling to 1981-2010')
args = parser.parse_args()

def scale_precip_log(x, mean, std, epsilon):
    x = np.log(x + epsilon)
    return (x - mean) / std

def scale_temp(x, mean, std):
    return (x - mean) / std

def descale_precip_log(x, mean, std, epsilon):
    x = x * std + mean
    return np.exp(x) - epsilon

def descale_temp(x, mean, std):
    return x * std + mean

sys.path.append(os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/UNet_Deterministic_Training_Dataset"))
model_path = os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/UNet_Deterministic_Training_Dataset/training_model_huber.pth")
training_checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model_instance = UNet(in_channels=5, out_channels=4)
model_instance.load_state_dict(training_checkpoint["model_state_dict"])
model_instance.to(device)
model_instance.eval()

# dOTC
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
    n_debug = min(10, n_time)  # Only process first 10 timesteps::: DEBUG

    precip_out = np.zeros((n_debug, target_shape[0], target_shape[1]), dtype=np.float32)
    temp_out   = np.zeros_like(precip_out)
    tmin_out   = np.zeros_like(precip_out)
    tmax_out   = np.zeros_like(precip_out)

    for t in trange(n_debug, desc="Downscaling frames (debug)", file=sys.stdout):
        rhiresd = scale_precip_log(np.where(dotc_input["precip"][t] < 0, 0, dotc_input["precip"][t].values), rhiresd_params["mean"], rhiresd_params["std"], rhiresd_params["epsilon"])
        tabsd   = scale_temp(dotc_input["temp"][t].values, tabsd_params["mean"], tabsd_params["std"])
        tmind   = scale_temp(dotc_input["tmin"][t].values, tmind_params["mean"], tmind_params["std"])
        tmaxd   = scale_temp(dotc_input["tmax"][t].values, tmaxd_params["mean"], tmaxd_params["std"])

        rhiresd = np.squeeze(rhiresd)
        tabsd   = np.squeeze(tabsd)
        tmind   = np.squeeze(tmind)
        tmaxd   = np.squeeze(tmaxd)
        elev    = elevation_2d

        frame_input = np.stack([rhiresd, tabsd, tmind, tmaxd, elev], axis=0)
        frame_input = torch.from_numpy(frame_input[np.newaxis, ...]).float().to(device)

        with torch.no_grad():
            output = model_instance(frame_input)
            output_np = output.cpu().numpy()[0]

        precip_out[t] = descale_precip_log(output_np[0], **scaling_params["RhiresD"])
        temp_out[t]   = descale_temp(output_np[1], scaling_params["TabsD"]["mean"], scaling_params["TabsD"]["std"])
        tmin_out[t]   = descale_temp(output_np[2], scaling_params["TminD"]["mean"], scaling_params["TminD"]["std"])
        tmax_out[t]   = descale_temp(output_np[3], scaling_params["TmaxD"]["mean"], scaling_params["TmaxD"]["std"])


print("precip_out shape:", precip_out.shape)
print("N shape:", dotc_input["N"].shape)
print("E shape:", dotc_input["E"].shape)
print("lat shape:", dotc_input["lat"].shape)
print("lon shape:", dotc_input["lon"].shape)

pred_ds = xr.Dataset(
    {
        "precip": (("time", "N", "E"), precip_out),
        "temp":   (("time", "N", "E"), temp_out),
        "tmin":   (("time", "N", "E"), tmin_out),
        "tmax":   (("time", "N", "E"), tmax_out),
    },
    coords={
        "time": dotc_input["time"].values[:n_debug],
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




print("precip_out shape:", precip_out.shape)
print("N shape:", dotc_input["N"].shape)
print("E shape:", dotc_input["E"].shape)
print("lat shape:", dotc_input["lat"].shape)
print("lon shape:", dotc_input["lon"].shape)
print("time shape:", dotc_input["time"].values[:n_debug].shape)
print("precip_out min/max:", np.nanmin(precip_out), np.nanmax(precip_out))



if args.validation_1981_2010:
    output_filename = "dOTC_ModelRun_Downscaled_Predictions_Validation_1981_2010.nc"
else:
    output_filename = "dOTC_ModelRun_Downscaled_Predictions_Debug_10.nc"

pred_ds.to_netcdf(output_filename)