import os
import yaml
import torch
import xarray as xr
import numpy as np
import json
import sys
import netCDF4
from UNet import UNet
from Downscaling_Dataset_Prep import DownscalingDataset
from torch.utils.data import DataLoader
from directories import BASE_DIR, SCALING_DIR, ELEVATION_PATH
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def scale_precip_log(x, mean, std, epsilon):
    x = np.log(x + epsilon)
    return (x - mean) / std

def scale_temp(x, mean, std):
    return (x - mean) / std

def descale_precip_log(x, mean, std, epsilon):
    x = x * std + mean
    x = np.clip(x, a_min=None, a_max=50)  # Prevent overflow in exp
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

# Load model run inputs (no targets needed)
#precip_input = xr.open_dataset(os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/BC_Model_Runs/EQM/precip_BC_bicubic_r01.nc"))
#temp_input   = xr.open_dataset(os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/BC_Model_Runs/EQM/temp_BC_bicubic_r01.nc"))
#tmin_input   = xr.open_dataset(os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/BC_Model_Runs/EQM/tmin_BC_bicubic_r01.nc"))
#tmax_input   = xr.open_dataset(os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/BC_Model_Runs/EQM/tmax_BC_bicubic_r01.nc"))

#inputs_merged = xr.merge([precip_input, temp_input, tmin_input, tmax_input])

#inputs_merged = inputs_merged.rename({
#    "precip": "RhiresD",
#    "temp": "TabsD",
#    "tmin": "TminD",
#    "tmax": "TmaxD"
#})


#For dOTC
dotc_input = xr.open_dataset(os.path.join(
    BASE_DIR,
    "sasthana/Downscaling/Downscaling_Models/BC_Model_Runs/dOTC/precip_temp_tmin_tmax_bicubic_r01.nc"
))

# Map model variables to UNet variables
model_var_names = ["precip", "temp", "tmin", "tmax"]
unet_var_names = ["RhiresD", "TabsD", "TminD", "TmaxD"]
var_map = dict(zip(model_var_names, unet_var_names))

config_path = os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/UNet_Deterministic_Training_Dataset/config.yaml")
with open(config_path, 'r') as f:
    directories = yaml.safe_load(f)

elevation_path = ELEVATION_PATH

# Load scaling parameters
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

# Scale inputs using your scaling functions, but keep original names
inputs_scaled = xr.Dataset()
inputs_scaled["precip"] = xr.DataArray(
    scale_precip_log(np.where(dotc_input["precip"] < 0, 0, dotc_input["precip"]), rhiresd_params["mean"], rhiresd_params["std"], rhiresd_params["epsilon"]),
    dims=dotc_input["precip"].dims,
    coords=dotc_input["precip"].coords
)
inputs_scaled["temp"] = xr.DataArray(
    scale_temp(dotc_input["temp"], tabsd_params["mean"], tabsd_params["std"]),
    dims=dotc_input["temp"].dims,
    coords=dotc_input["temp"].coords
)
inputs_scaled["tmin"] = xr.DataArray(
    scale_temp(dotc_input["tmin"], tmind_params["mean"], tmind_params["std"]),
    dims=dotc_input["tmin"].dims,
    coords=dotc_input["tmin"].coords
)
inputs_scaled["tmax"] = xr.DataArray(
    scale_temp(dotc_input["tmax"], tmaxd_params["mean"], tmaxd_params["std"]),
    dims=dotc_input["tmax"].dims,
    coords=dotc_input["tmax"].coords
)

ds = DownscalingDataset(inputs_scaled, None, directories, elevation_path)

# Dataloader (targets=None)
test_loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

# Get 1D lat/lon
if dotc_input.lat.ndim == 2:
    lat_1d = dotc_input.lat.values[:, 0]
    lon_1d = dotc_input.lon.values[0, :]
else:
    lat_1d = dotc_input.lat.values
    lon_1d = dotc_input.lon.values

n_time = len(ds)
n_lat = len(lat_1d)
n_lon = len(lon_1d)

# Chunked writing using netCDF4
with netCDF4.Dataset("dOTC_ModelRun_Downscaled_Predictions.nc", "w") as nc_out:
    nc_out.createDimension("time", n_time)
    nc_out.createDimension("lat", n_lat)
    nc_out.createDimension("lon", n_lon)
    for var in model_var_names:
        nc_out.createVariable(var, "f4", ("time", "lat", "lon"), chunksizes=(32, n_lat, n_lon))

    nc_out.createVariable("lat", "f4", ("lat",))
    nc_out.createVariable("lon", "f4", ("lon",))
    nc_out.createVariable("time", "f8", ("time",))
    nc_out.variables["lat"][:] = lat_1d
    nc_out.variables["lon"][:] = lon_1d
    nc_out.variables["time"][:] = dotc_input.time.values

    time_idx = 0
    with torch.no_grad():
        for input_batch in tqdm(test_loader, desc="Inference on model runs"):
            if isinstance(input_batch, (list, tuple)):
                input_batch = input_batch[0]
            input_batch = input_batch.to(device)
            output_batch = model_instance(input_batch)
            batch_np = output_batch.cpu().numpy()  # shape: (batch_size, 4, H, W)

            # Destandardize batch and map to model variable names
            nc_out.variables["precip"][time_idx:time_idx+batch_np.shape[0], :, :] = descale_precip_log(batch_np[:, 0, :, :], **scaling_params["RhiresD"])
            nc_out.variables["temp"][time_idx:time_idx+batch_np.shape[0], :, :]   = descale_temp(batch_np[:, 1, :, :], scaling_params["TabsD"]["mean"], scaling_params["TabsD"]["std"])
            nc_out.variables["tmin"][time_idx:time_idx+batch_np.shape[0], :, :]   = descale_temp(batch_np[:, 2, :, :], scaling_params["TminD"]["mean"], scaling_params["TminD"]["std"])
            nc_out.variables["tmax"][time_idx:time_idx+batch_np.shape[0], :, :]   = descale_temp(batch_np[:, 3, :, :], scaling_params["TmaxD"]["mean"], scaling_params["TmaxD"]["std"])
            time_idx += batch_np.shape[0]

print("Batch-wise inference and saving complete.")