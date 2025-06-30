import os
import yaml
import torch
import xarray as xr
import numpy as np
import json
import sys
from UNet import UNet
from Downscaling_Dataset_Prep import DownscalingDataset
from torch.utils.data import DataLoader
import torch.nn as nn
os.environ["BASE_DIR"] = "/work/FAC/FGSE/IDYST/tbeucler/downscaling"
BASE_DIR = os.environ["BASE_DIR"]
sys.path.append(os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/models_UNet/UNet_Deterministic_Pretraining_Dataset"))

# Scaling functions : now using parameters from the loaded JSON files

def norm_precip(x, params):
    return (x - params["min"]) / (params["max"] - params["min"])

def norm_temp(x, params):
    return (x - params["mean"]) / params["std"]

def norm_tmin(x, params):
    return (x - params["mean"]) / params["std"]

def norm_tmax(x, params):
    return (x - params["mean"]) / params["std"]


#Descaling fucntion
def descale_precip(x,min_val, max_val):
    return x* (max_val - min_val) + min_val
def descale_temp(x, mean, std):
    return x * std + mean


model_path = os.path.join("best_model_Huber_pretraining_chronological_split_FULL.pth")
training_checkpoint =torch.load(model_path,map_location=torch.device('cpu')) #Moving model to CPU 

#model instance, weights 
model_instance= UNet(in_channels=5, out_channels=4)
model_instance.load_state_dict(training_checkpoint["model_state_dict"])
model_instance.eval()

# Scaling params loading from .json files of the pretraining dataset
scaling_dir = os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/Pretraining_Chronological_Dataset")
rhiresd_params = json.load(open(os.path.join(scaling_dir, "precip_scaling_params_chronological.json")))
tabsd_params   = json.load(open(os.path.join(scaling_dir, "temp_scaling_params_chronological.json")))
tmind_params   = json.load(open(os.path.join(scaling_dir, "tmin_scaling_params_chronological.json")))
tmaxd_params   = json.load(open(os.path.join(scaling_dir, "tmax_scaling_params_chronological.json")))


#prepping inputs 

precip_input = xr.open_dataset(
    os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/Training_Chronological_Dataset/RhiresD_step3_interp.nc")
).sel(time=slice("2011-01-01", "2020-12-31"))["RhiresD"].chunk({"time": 100}).rename("precip")
temp_input = xr.open_dataset(
    os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/Training_Chronological_Dataset/TabsD_step3_interp.nc")
).sel(time=slice("2011-01-01", "2020-12-31"))["TabsD"].chunk({"time": 100}).rename("temp")
tmin_input = xr.open_dataset(
    os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/Training_Chronological_Dataset/TminD_step3_interp.nc")
).sel(time=slice("2011-01-01", "2020-12-31"))["TminD"].chunk({"time": 100}).rename("tmin")
tmax_input = xr.open_dataset(
    os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/Training_Chronological_Dataset/TmaxD_step3_interp.nc")
).sel(time=slice("2011-01-01", "2020-12-31"))["TmaxD"].chunk({"time": 100}).rename("tmax")

# Normalize using loaded JSON parameters
precip_input = norm_precip(precip_input, rhiresd_params).rename("precip")
temp_input   = norm_temp(temp_input, tabsd_params).rename("temp")
tmin_input   = norm_tmin(tmin_input, tmind_params).rename("tmin")
tmax_input   = norm_tmax(tmax_input, tmaxd_params).rename("tmax")


#Prepping targets 
precip=xr.open_dataset(os.path.join(BASE_DIR, "sasthana/Downscaling/Processing_and_Analysis_Scripts/data_1971_2023/HR_files_full/RhiresD_1971_2023.nc"))
temp=xr.open_dataset(os.path.join(BASE_DIR, "sasthana/Downscaling/Processing_and_Analysis_Scripts/data_1971_2023/HR_files_full/TabsD_1971_2023.nc"))
tmin=xr.open_dataset(os.path.join(BASE_DIR, "sasthana/Downscaling/Processing_and_Analysis_Scripts/data_1971_2023/HR_files_full/TminD_1971_2023.nc"))
tmax=xr.open_dataset(os.path.join(BASE_DIR, "sasthana/Downscaling/Processing_and_Analysis_Scripts/data_1971_2023/HR_files_full/TmaxD_1971_2023.nc"))

precip_target = precip_target = norm_precip(
    precip.sel(time=slice("2011-01-01", "2020-12-31"))["RhiresD"].chunk({"time": 100}),
    rhiresd_params).rename("precip")
temp_target = norm_temp(
    temp.sel(time=slice("2011-01-01", "2020-12-31"))["TabsD"].chunk({"time": 100}),
    tabsd_params).rename("temp")
tmin_target = norm_tmin(
    tmin.sel(time=slice("2011-01-01", "2020-12-31"))["TminD"].chunk({"time": 100}),
    tmind_params).rename("tmin")
tmax_target = norm_tmax(
    tmax.sel(time=slice("2011-01-01", "2020-12-31"))["TmaxD"].chunk({"time": 100}),
    tmaxd_params).rename("tmax")

# config used for training 
config_path = os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/models_UNet/UNet_Deterministic_Pretraining_Dataset/config.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

elevation_path = os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/elevation.tif")

# Merging ds for DataLoader
inputs_merged = xr.merge([precip_input, temp_input, tmin_input, tmax_input])
targets_merged = xr.merge([precip_target, temp_target, tmin_target, tmax_target])
print(inputs_merged.lat.shape)
print(inputs_merged.lon.shape)

ds = DownscalingDataset(inputs_merged, targets_merged, config, elevation_path)
paired_ds = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)


#Inference loop
loss_fn= nn.HuberLoss(delta=0.05)  # Identical as in the config provided at training time
all_preds = []
all_targets = []
losses = []
with torch.no_grad():
    for input_batch, target_batch in paired_ds:
        output_batch = model_instance(input_batch)
        all_preds.append(output_batch.squeeze(0).cpu().numpy())
        all_targets.append(target_batch.squeeze(0).cpu().numpy())
        # Computing test average loss
        loss = loss_fn(output_batch, target_batch)
        losses.append(loss.item())

all_preds = np.stack(all_preds)
all_targets = np.stack(all_targets) 
# Printing average test loss
print(f"Average test loss: {np.mean(losses)}")

#Saving predictions concatenated as a single time series 

all_preds_denorm = np.empty_like(all_preds)
all_preds_denorm[:, 0, :, :] = descale_precip(all_preds[:, 0, :, :], rhiresd_params["min"], rhiresd_params["max"])
all_preds_denorm[:, 1, :, :] = descale_temp(all_preds[:, 1, :, :], tabsd_params["mean"], tabsd_params["std"])
all_preds_denorm[:, 2, :, :] = descale_temp(all_preds[:, 2, :, :], tmind_params["mean"], tmind_params["std"])
all_preds_denorm[:, 3, :, :] = descale_temp(all_preds[:, 3, :, :], tmaxd_params["mean"], tmaxd_params["std"])


if inputs_merged.lat.ndim==2:
    lat_1d=inputs_merged.lat.values[:, 0]
    lon_1d=inputs_merged.lon.values[0,:]
else:
    lat_1d=inputs_merged.lat.values
    lon_1d=inputs_merged.lon.values


var_names = ["precip", "temp", "tmin", "tmax"]
pred_vars = {}
for i, var in enumerate(var_names):
    pred_vars[var] = xr.DataArray(
        all_preds_denorm[:, i, :, :],
        dims=("time", "lat", "lon"),
        coords={
            "time": inputs_merged.time.values,
            "lat": lat_1d,
            "lon": lon_1d,
        },
        name=var
    )


pred_ds = xr.Dataset(pred_vars)
pred_ds.to_netcdf("downscaled_predictions_Unet_1771_2020_ds.nc")

