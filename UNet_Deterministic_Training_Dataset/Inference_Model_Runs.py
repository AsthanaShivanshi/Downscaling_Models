import os
import yaml
import torch
import xarray as xr
import numpy as np
import json
import sys
from UNet import UNet
from directories import BASE_DIR, SCALING_DIR, ELEVATION_PATH

from skimage.transform import resize

device = torch.device("cpu")

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

dotc_input = xr.open_dataset(os.path.join(
    BASE_DIR,
    "sasthana/Downscaling/Downscaling_Models/BC_Model_Runs/dOTC/precip_temp_tmin_tmax_bicubic_r01.nc"
))


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

if elevation_2d.ndim > 2:
    print("WARNING: elevation_2d has shape", elevation_2d.shape, "- selecting first slice along last axis")
    elevation_2d = elevation_2d[..., 0]  # or elevation_2d[:, :, 0] if shape is (240, 370, 387)

target_shape = dotc_input["precip"].shape[-2:] 
print("target_shape for elevation:", target_shape)
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
lat_1d = dotc_input.lat.values if dotc_input.lat.ndim == 1 else dotc_input.lat.values[:, 0]
lon_1d = dotc_input.lon.values if dotc_input.lon.ndim == 1 else dotc_input.lon.values[0, :]
precip_out = np.zeros((n_time, target_shape[0], target_shape[1]), dtype=np.float32)
temp_out   = np.zeros_like(precip_out)
tmin_out   = np.zeros_like(precip_out)
tmax_out   = np.zeros_like(precip_out)


for t in range(n_time):
    rhiresd = scale_precip_log(np.where(dotc_input["precip"][t] < 0, 0, dotc_input["precip"][t].values), rhiresd_params["mean"], rhiresd_params["std"], rhiresd_params["epsilon"])
    tabsd   = scale_temp(dotc_input["temp"][t].values, tabsd_params["mean"], tabsd_params["std"])
    tmind   = scale_temp(dotc_input["tmin"][t].values, tmind_params["mean"], tmind_params["std"])
    tmaxd   = scale_temp(dotc_input["tmax"][t].values, tmaxd_params["mean"], tmaxd_params["std"])

    rhiresd = np.squeeze(rhiresd)
    tabsd   = np.squeeze(tabsd)
    tmind   = np.squeeze(tmind)
    tmaxd   = np.squeeze(tmaxd)

    elev = elevation_2d
    print(f"Frame {t} shapes: rhiresd={rhiresd.shape}, tabsd={tabsd.shape}, tmind={tmind.shape}, tmaxd={tmaxd.shape}, elev={elev.shape}")

    frame_input = np.stack([rhiresd, tabsd, tmind, tmaxd, elev], axis=0)  # (5, lat, lon)
    frame_input = torch.from_numpy(frame_input[np.newaxis, ...]).float().to(device)  # (1, 5, lat, lon)

    # Inference
    with torch.no_grad():
        output = model_instance(frame_input)  # (1, 4, lat, lon)
        output_np = output.cpu().numpy()[0]   # (4, lat, lon)


    # Destandardize and store
    precip_out[t] = descale_precip_log(output_np[0], **scaling_params["RhiresD"])
    temp_out[t]   = descale_temp(output_np[1], scaling_params["TabsD"]["mean"], scaling_params["TabsD"]["std"])
    tmin_out[t]   = descale_temp(output_np[2], scaling_params["TminD"]["mean"], scaling_params["TminD"]["std"])
    tmax_out[t]   = descale_temp(output_np[3], scaling_params["TmaxD"]["mean"], scaling_params["TmaxD"]["std"])
    
    
    
    # Optionally print progress
    if t % 10 == 0:
        print(f"Processed frame {t}/{n_time}")

dotc_input.close()

coords = {
    "time": dotc_input.time.values,
    "lat": lat_1d,
    "lon": lon_1d
}

ds_out = xr.Dataset(
    {
        "precip": (("time", "lat", "lon"), precip_out),
        "temp":   (("time", "lat", "lon"), temp_out),
        "tmin":   (("time", "lat", "lon"), tmin_out),
        "tmax":   (("time", "lat", "lon"), tmax_out),
    },
    coords=coords
)

if "units" in dotc_input.time.attrs:
    ds_out["time"].attrs["units"] = dotc_input.time.attrs["units"]
if "calendar" in dotc_input.time.attrs:
    ds_out["time"].attrs["calendar"] = dotc_input.time.attrs["calendar"]

ds_out.to_netcdf("dOTC_ModelRun_Downscaled_Predictions.nc")