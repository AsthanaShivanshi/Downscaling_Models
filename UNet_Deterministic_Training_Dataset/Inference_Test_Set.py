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
from directories import BASE_DIR, SCALING_DIR, ELEVATION_PATH
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def descale_precip_log(x, mean, std, epsilon):
    x = x * std + mean
    return np.exp(x) - epsilon

def descale_temp(x, mean, std):
    return x * std + mean

sys.path.append(os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/UNet_Deterministic_Training_Dataset"))



model_path = os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/UNet_Deterministic_Training_Dataset/trial_no_trial_best_model.pth")
training_checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

model_instance = UNet(in_channels=5, out_channels=4)
model_instance.load_state_dict(training_checkpoint["model_state_dict"])
model_instance.to(device)
model_instance.eval()

precip_input = xr.open_dataset(os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/Training_Chronological_Dataset/RhiresD_input_test_chronological_scaled.nc"))
temp_input   = xr.open_dataset(os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/Training_Chronological_Dataset/TabsD_input_test_chronological_scaled.nc"))
tmin_input   = xr.open_dataset(os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/Training_Chronological_Dataset/TminD_input_test_chronological_scaled.nc"))
tmax_input   = xr.open_dataset(os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/Training_Chronological_Dataset/TmaxD_input_test_chronological_scaled.nc"))

config_path = os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/UNet_Deterministic_Training_Dataset/config.yaml")
with open(config_path, 'r') as f:
    directories = yaml.safe_load(f)

elevation_path = ELEVATION_PATH

# Dataloader (no targets)
inputs_merged = xr.merge([precip_input, temp_input, tmin_input, tmax_input])
ds = DownscalingDataset(inputs_merged, None, directories, elevation_path)
test_loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)

var_names = ["RhiresD", "TabsD", "TminD", "TmaxD"]
all_preds = []

with torch.no_grad():
    for input_batch in tqdm(test_loader, desc="Inference on test set"):
        if isinstance(input_batch, (list, tuple)):
            input_batch = input_batch[0]
        input_batch = input_batch.to(device)
        output_batch = model_instance(input_batch)
        all_preds.append(output_batch.squeeze(0).cpu().numpy())

all_preds = np.stack(all_preds)

# Load scaling params
rhiresd_params = json.load(open(os.path.join(SCALING_DIR, "RhiresD_scaling_params_chronological.json")))
tabsd_params   = json.load(open(os.path.join(SCALING_DIR, "TabsD_scaling_params_chronological.json")))
tmind_params   = json.load(open(os.path.join(SCALING_DIR, "TminD_scaling_params_chronological.json")))
tmaxd_params   = json.load(open(os.path.join(SCALING_DIR, "TmaxD_scaling_params_chronological.json")))

# Destandardising preds
all_preds_denorm = np.empty_like(all_preds)
all_preds_denorm[:, 0, :, :] = descale_precip_log(all_preds[:, 0, :, :], rhiresd_params["mean"], rhiresd_params["std"], rhiresd_params["epsilon"])
all_preds_denorm[:, 1, :, :] = descale_temp(all_preds[:, 1, :, :], tabsd_params["mean"], tabsd_params["std"])
all_preds_denorm[:, 2, :, :] = descale_temp(all_preds[:, 2, :, :], tmind_params["mean"], tmind_params["std"])
all_preds_denorm[:, 3, :, :] = descale_temp(all_preds[:, 3, :, :], tmaxd_params["mean"], tmaxd_params["std"])



if inputs_merged.lat.ndim == 2:
    lat_1d = inputs_merged.lat.values[:, 0]
    lon_1d = inputs_merged.lon.values[0, :]
else:
    lat_1d = inputs_merged.lat.values
    lon_1d = inputs_merged.lon.values

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
pred_ds.to_netcdf("Training_Dataset_Downscaled_Predictions_2011_2020.nc")