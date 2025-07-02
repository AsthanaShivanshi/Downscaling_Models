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
import torch.nn.functional as F
from losses import WeightedHuberLoss,WeightedMSELoss

#For later descaling of predicted outputs

def descale_precip(x, min_val, max_val):
    return x * (max_val - min_val) + min_val

def descale_temp(x, mean, std):
    return x * std + mean


os.environ["BASE_DIR"] = "/work/FAC/FGSE/IDYST/tbeucler/downscaling"
BASE_DIR = os.environ["BASE_DIR"]

sys.path.append(os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/models_UNet/UNet_Deterministic_Training_Dataset"))


model_path = os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/models_UNet/UNet_Deterministic_Training_Dataset/best_model_Huber_FULL_RLOP.pth")
training_checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

# Model instanc, weights
model_instance = UNet(in_channels=5, out_channels=4)
model_instance.load_state_dict(training_checkpoint["model_state_dict"])
model_instance.eval()

precip_input = xr.open_dataset(os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/Training_Chronological_Dataset/RhiresD_input_test_chronological_scaled.nc"))
temp_input   = xr.open_dataset(os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/Training_Chronological_Dataset/TabsD_input_test_chronological_scaled.nc"))
tmin_input   = xr.open_dataset(os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/Training_Chronological_Dataset/TminD_input_test_chronological_scaled.nc"))
tmax_input   = xr.open_dataset(os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/Training_Chronological_Dataset/TmaxD_input_test_chronological_scaled.nc"))

precip_target = xr.open_dataset(os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/Training_Chronological_Dataset/RhiresD_target_test_chronological_scaled.nc"))
temp_target   = xr.open_dataset(os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/Training_Chronological_Dataset/TabsD_target_test_chronological_scaled.nc"))
tmin_target   = xr.open_dataset(os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/Training_Chronological_Dataset/TminD_target_test_chronological_scaled.nc"))
tmax_target   = xr.open_dataset(os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/Training_Chronological_Dataset/TmaxD_target_test_chronological_scaled.nc"))

config_path = os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/models_UNet/UNet_Deterministic_Training_Dataset/config.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

elevation_path = os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/elevation.tif")

#merging datasets for dataloader
inputs_merged = xr.merge([precip_input, temp_input, tmin_input, tmax_input])
targets_merged = xr.merge([precip_target, temp_target, tmin_target, tmax_target])
print(inputs_merged.lat.shape)
print(inputs_merged.lon.shape)


ds = DownscalingDataset(inputs_merged, targets_merged, config, elevation_path)

paired_ds = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)

# Loss function
loss_fn_name = config["train"].get("loss_fn", "huber").lower()
if loss_fn_name == "huber":
    weights = [0.4, 0.2, 0.2, 0.2]
    delta = config["train"].get("huber_delta", 0.05)
    criterion = WeightedHuberLoss(weights=weights, delta=delta)
elif loss_fn_name == "mse":
    weights = [0.4, 0.2, 0.2, 0.2]
    criterion = WeightedMSELoss(weights=weights)
else:
    raise ValueError(f"Unknown loss function: {loss_fn_name}")


var_names = ["RhiresD", "TabsD", "TminD", "TmaxD"]

all_preds = []
all_targets = []
losses=[]
channel_losses_individual= []

with torch.no_grad():
    for input_batch, target_batch in paired_ds:
        output_batch = model_instance(input_batch)
        all_preds.append(output_batch.squeeze(0).cpu().numpy())
        all_targets.append(target_batch.squeeze(0).cpu().numpy())
        #Weighted total across four channels : using criteria defined in Experiments
        total_loss = criterion(output_batch, target_batch).item()
        losses.append(total_loss)
        # For per-channel loss, you can use the underlying F.huber_loss or F.mse_loss as before:
        if loss_fn_name == "huber":
            individual = [
                F.huber_loss(output_batch[:, c], target_batch[:, c], delta=delta, reduction='mean').item()
                for c in range(output_batch.shape[1])
            ]
        else:
            individual = [
                F.mse_loss(output_batch[:, c], target_batch[:, c], reduction='mean').item()
                for c in range(output_batch.shape[1])
            ]
        channel_losses_individual.append(individual)

all_preds = np.stack(all_preds)
all_targets = np.stack(all_targets)

print(f"Average test loss: {np.mean(losses)}")

channel_losses_individual = np.array(channel_losses_individual)
avg_channel_losses = np.mean(channel_losses_individual, axis=0)
for var, loss in zip(var_names, avg_channel_losses):
    print(f"Average loss for channel {var}: {loss}")


# Scaling params loading from the .json files
scaling_dir = os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/Training_Chronological_Dataset")
rhiresd_params = json.load(open(os.path.join(scaling_dir, "RhiresD_scaling_params_chronological.json")))
tabsd_params   = json.load(open(os.path.join(scaling_dir, "TabsD_scaling_params_chronological.json")))
tmind_params   = json.load(open(os.path.join(scaling_dir, "TminD_scaling_params_chronological.json")))
tmaxd_params   = json.load(open(os.path.join(scaling_dir, "TmaxD_scaling_params_chronological.json")))


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
pred_ds.to_netcdf("downscaled_predictions_2011_2020_ds.nc")