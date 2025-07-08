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

def descale_precip(x, min_val, max_val):
    return x * (max_val - min_val) + min_val

def descale_temp(x, mean, std):
    return x * std + mean

os.environ["BASE_DIR"] = "/work/FAC/FGSE/IDYST/tbeucler/downscaling"
BASE_DIR = os.environ["BASE_DIR"]

sys.path.append(os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/models_UNet/UNet_Deterministic_Pretraining_Dataset"))

# Load scaling params
scaling_dir = os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/Pretraining_Chronological_Dataset")
rhiresd_params = json.load(open(os.path.join(scaling_dir, "precip_scaling_params_chronological.json")))
tabsd_params   = json.load(open(os.path.join(scaling_dir, "temp_scaling_params_chronological.json")))
tmind_params   = json.load(open(os.path.join(scaling_dir, "tmin_scaling_params_chronological.json")))
tmaxd_params   = json.load(open(os.path.join(scaling_dir, "tmax_scaling_params_chronological.json")))

#test set : 2011-2020 from the HR files (not reconstructuon)
precip = xr.open_dataset(os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/Training_Chronological_Dataset/RhiresD_step3_interp.nc"))
temp   = xr.open_dataset(os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/Training_Chronological_Dataset/TabsD_step3_interp.nc"))
tmin   = xr.open_dataset(os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/Training_Chronological_Dataset/TminD_step3_interp.nc"))
tmax   = xr.open_dataset(os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/Training_Chronological_Dataset/TmaxD_step3_interp.nc"))

precip_input = precip.sel(time=slice("2011-01-01", "2020-12-31"))
precip_input = precip_input.copy()
precip_input['RhiresD'] = (precip_input['RhiresD'] - rhiresd_params["min"]) / (rhiresd_params["max"] - rhiresd_params["min"])

temp_input = temp.sel(time=slice("2011-01-01", "2020-12-31"))
temp_input = temp_input.copy()
temp_input['TabsD'] = (temp_input['TabsD'] - tabsd_params["mean"]) / tabsd_params["std"]

tmin_input = tmin.sel(time=slice("2011-01-01", "2020-12-31"))
tmin_input = tmin_input.copy()
tmin_input['TminD'] = (tmin_input['TminD'] - tmind_params["mean"]) / tmind_params["std"]

tmax_input = tmax.sel(time=slice("2011-01-01", "2020-12-31"))
tmax_input = tmax_input.copy()
tmax_input['TmaxD'] = (tmax_input['TmaxD'] - tmaxd_params["mean"]) / tmaxd_params["std"]

# Rename variables to match config of the UNet trained on the reconstruction ds
precip_input = precip_input.rename({'RhiresD': 'precip'})
temp_input   = temp_input.rename({'TabsD': 'temp'})
tmin_input   = tmin_input.rename({'TminD': 'tmin'})
tmax_input   = tmax_input.rename({'TmaxD': 'tmax'})

# Merging: for dataloader
inputs_merged = xr.merge([precip_input, temp_input, tmin_input, tmax_input])

config_path = os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/models_UNet/UNet_Deterministic_Pretraining_Dataset/config.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

elevation_path = os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/elevation.tif")

# Dummy targets (not used for inference, but required by DownscalingDataset functionality)
targets_merged = inputs_merged.copy()

ds = DownscalingDataset(inputs_merged, targets_merged, config, elevation_path)
paired_ds = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)

model_path = os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/models_UNet/UNet_Deterministic_Pretraining_Dataset/full_best_model_huber_pretraining_FULL_RLOP.pth")
training_checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model_instance = UNet(in_channels=5, out_channels=4)
model_instance.load_state_dict(training_checkpoint["model_state_dict"])
model_instance.eval()

var_names = ["precip", "temp", "tmin", "tmax"]
all_preds = []

with torch.no_grad():
    for input_batch, _ in paired_ds:
        output_batch = model_instance(input_batch)
        all_preds.append(output_batch.squeeze(0).cpu().numpy())

all_preds = np.stack(all_preds)

# Denormalization
all_preds_denorm = np.empty_like(all_preds)
all_preds_denorm[:, 0, :, :] = descale_precip(all_preds[:, 0, :, :], rhiresd_params["min"], rhiresd_params["max"])
all_preds_denorm[:, 1, :, :] = descale_temp(all_preds[:, 1, :, :], tabsd_params["mean"], tabsd_params["std"])
all_preds_denorm[:, 2, :, :] = descale_temp(all_preds[:, 2, :, :], tmind_params["mean"], tmind_params["std"])
all_preds_denorm[:, 3, :, :] = descale_temp(all_preds[:, 3, :, :], tmaxd_params["mean"], tmaxd_params["std"])

if inputs_merged.lat.ndim == 2:
    lat_1d = inputs_merged.lat.values[:, 0]
    lon_1d = inputs_merged.lon.values[0, :]
else:
    lat_1d = inputs_merged.lat.values
    lon_1d = inputs_merged.lon.values

# Save predictions as xarray Dataset
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
pred_ds.to_netcdf("Pretraining_Dataset_Downscaled_Predictions_2011_2020.nc")