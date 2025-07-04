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
import matplotlib.pyplot as plt

#functions for scaling and descaling
def descale_precip(x, min_val, max_val):
    return x * (max_val - min_val) + min_val

def descale_temp(x, mean, std):
    return x * std + mean

def scale_precip(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)

def scale_temp(x, mean, std):
    return (x - mean) / std

os.environ["BASE_DIR"] = "/work/FAC/FGSE/IDYST/tbeucler/downscaling"
BASE_DIR = os.environ["BASE_DIR"]

sys.path.append(os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/models_UNet/UNet_Deterministic_Pretraining_Dataset"))

# Scaling params loading from the .json files
scaling_dir = os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/Pretraining_Chronological_Dataset")
rhiresd_params = json.load(open(os.path.join(scaling_dir, "precip_scaling_params_chronological.json")))
tabsd_params   = json.load(open(os.path.join(scaling_dir, "temp_scaling_params_chronological.json")))
tmind_params   = json.load(open(os.path.join(scaling_dir, "tmin_scaling_params_chronological.json")))
tmaxd_params   = json.load(open(os.path.join(scaling_dir, "tmax_scaling_params_chronological.json")))

#Test dataset remains the same across configurations trained 
precip= xr.open_dataset(os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/Training_Chronological_Dataset/RhiresD_step3_interp.nc"))
temp=   xr.open_dataset(os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/Training_Chronological_Dataset/TabsD_step3_interp.nc"))
tmin=   xr.open_dataset(os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/Training_Chronological_Dataset/TminD_step3_interp.nc"))
tmax=   xr.open_dataset(os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/Training_Chronological_Dataset/TmaxD_step3_interp.nc"))


# Selecting 2011-2020 and scaling using pretraining params from json files
precip_input = precip.sel(time=slice("2011-01-01", "2020-12-31"))
precip_input = precip_input.copy()
precip_input['RhiresD'] = scale_precip(
    precip_input['RhiresD'],
    rhiresd_params["min"],
    rhiresd_params["max"]
)

temp_input = temp.sel(time=slice("2011-01-01", "2020-12-31"))
temp_input = temp_input.copy()
temp_input['TabsD'] = scale_temp(
    temp_input['TabsD'],
    tabsd_params["mean"],
    tabsd_params["std"]
)

tmin_input = tmin.sel(time=slice("2011-01-01", "2020-12-31"))
tmin_input = tmin_input.copy()
tmin_input['TminD'] = scale_temp(
    tmin_input['TminD'],
    tmind_params["mean"],
    tmind_params["std"]
)

tmax_input = tmax.sel(time=slice("2011-01-01", "2020-12-31"))
tmax_input = tmax_input.copy()
tmax_input['TmaxD'] = scale_temp(
    tmax_input['TmaxD'],
    tmaxd_params["mean"],
    tmaxd_params["std"]
)


model_path = os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/models_UNet/UNet_Deterministic_Pretraining_Dataset/full_best_model_huber_pretraining_FULL_RLOP.pth")
training_checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

# Model instanc, weights
model_instance = UNet(in_channels=5, out_channels=4)
model_instance.load_state_dict(training_checkpoint["model_state_dict"])
model_instance.eval()

# HR target file paths
hr_dir = os.path.join(BASE_DIR, "sasthana/Downscaling/Processing_and_Analysis_Scripts/data_1971_2023/HR_files_full")
precip_hr = xr.open_dataset(os.path.join(hr_dir, "RhiresD_1971_2023.nc"))
temp_hr   = xr.open_dataset(os.path.join(hr_dir, "TabsD_1971_2023.nc"))
tmin_hr   = xr.open_dataset(os.path.join(hr_dir, "TminD_1971_2023.nc"))
tmax_hr   = xr.open_dataset(os.path.join(hr_dir, "TmaxD_1971_2023.nc"))

# Select 2011-2020 and scale
precip_target = precip_hr.sel(time=slice("2011-01-01", "2020-12-31"))
precip_target = precip_target.copy()
precip_target['RhiresD'] = scale_precip(
    precip_target['RhiresD'],
    rhiresd_params["min"],
    rhiresd_params["max"]
)

temp_target = temp_hr.sel(time=slice("2011-01-01", "2020-12-31"))
temp_target = temp_target.copy()
temp_target['TabsD'] = scale_temp(
    temp_target['TabsD'],
    tabsd_params["mean"],
    tabsd_params["std"]
)

tmin_target = tmin_hr.sel(time=slice("2011-01-01", "2020-12-31"))
tmin_target = tmin_target.copy()
tmin_target['TminD'] = scale_temp(
    tmin_target['TminD'],
    tmind_params["mean"],
    tmind_params["std"]
)

tmax_target = tmax_hr.sel(time=slice("2011-01-01", "2020-12-31"))
tmax_target = tmax_target.copy()
tmax_target['TmaxD'] = scale_temp(
    tmax_target['TmaxD'],
    tmaxd_params["mean"],
    tmaxd_params["std"]
)

config_path = os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/models_UNet/UNet_Deterministic_Pretraining_Dataset/config.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

elevation_path = os.path.join(BASE_DIR, "sasthana/Downscaling/Downscaling_Models/elevation.tif")


# Renaming vars to match expected config in DownscalingDataset/config
precip_input = precip_input.rename({'RhiresD': 'precip'})
temp_input   = temp_input.rename({'TabsD': 'temp'})
tmin_input   = tmin_input.rename({'TminD': 'tmin'})
tmax_input   = tmax_input.rename({'TmaxD': 'tmax'})

precip_target = precip_target.rename({'RhiresD': 'precip'})
temp_target   = temp_target.rename({'TabsD': 'temp'})
tmin_target   = tmin_target.rename({'TminD': 'tmin'})
tmax_target   = tmax_target.rename({'TmaxD': 'tmax'})

#for dataloader
inputs_merged = xr.merge([precip_input, temp_input, tmin_input, tmax_input])
targets_merged = xr.merge([precip_target, temp_target, tmin_target, tmax_target])
print(inputs_merged.lat.shape)
print(inputs_merged.lon.shape)


ds = DownscalingDataset(inputs_merged, targets_merged, config, elevation_path)

paired_ds = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)

# Loss function
loss_fn_name = config["train"].get("loss_fn", "huber").lower()
if loss_fn_name == "huber":
    weights = [0.25, 0.25, 0.25, 0.25]
    delta = config["train"].get("huber_delta", 0.05)
    criterion = WeightedHuberLoss(weights=weights, delta=delta)
elif loss_fn_name == "mse":
    weights = [0.25, 0.25, 0.25, 0.25]
    criterion = WeightedMSELoss(weights=weights)
else:
    raise ValueError(f"Unknown loss function: {loss_fn_name}")


var_names = ["precip", "temp", "tmin", "tmax"]

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
        # For per-channel loss
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


#First naive look at how model did on the tails on the 2011-2020 test set
#Computing extreme quantiles for checking  tails of the distribution performance
quantiles= list(range(5,100,5)) #All quantiles
thresholds={}
for i, var in enumerate(var_names):
    targets_flattened = all_targets[:, i, :, :].flatten()
    preds_flattened = all_preds[:, i, :, :].flatten()
    thresholds[var] = [np.quantile(targets_flattened, q/100) for q in quantiles]
    mses = []
    for q, thresh in zip(quantiles, thresholds[var]):
        mask = targets_flattened >= thresh
        if np.sum(mask) == 0:
            mse = np.nan #Skipping it??
            continue
        else:
            mse = np.mean((targets_flattened[mask] - preds_flattened[mask])**2)
        mses.append(mse)
        if q == quantiles[-1] and np.sum(mask) > 0:
            plt.scatter(targets_flattened[mask], preds_flattened[mask], alpha=0.5, label=f"{q}th quantile scatter")
    plt.figure(figsize=(10, 5))
    plt.plot(quantiles, mses, marker='o', label="Thresholded MSE across different quantiles")
    plt.xlabel("Quantile (%)")
    plt.ylabel("MSE")
    plt.title(f"{var} - Thresholded MSE by Quantile")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Pretraining_{var}_thresholded_mse.png", dpi=1000)
    plt.close()



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
pred_ds.to_netcdf("Pretraining_Dataset_Downscaled_Predictions_2011_2020.nc")