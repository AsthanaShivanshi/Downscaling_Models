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

ds = DownscalingDataset(inputs_merged, targets_merged, config, elevation_path)

paired_ds = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)

# looping for inference, also printing the test loss in the end of the inference loop
loss_fn= nn.HuberLoss(delta=0.05) #Identical as in the config provided at traning time
all_preds = []
all_targets = []
losses=[]
with torch.no_grad():
    for input_batch, target_batch in paired_ds:
        output_batch = model_instance(input_batch)
        all_preds.append(output_batch.squeeze(0).cpu().numpy())
        all_targets.append(target_batch.squeeze(0).cpu().numpy())
        #Computing loss
        loss=loss_fn(output_batch, target_batch)
        losses.append(loss.item())

all_preds = np.stack(all_preds)
all_targets = np.stack(all_targets)

#printing average test loss
print(f"Average test loss: {np.mean(losses)}")



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


pred_da = xr.DataArray(
    all_preds_denorm,
    dims=("time", "channel", "lat", "lon"),
    coords={
        "time": inputs_merged.time.values,
        "channel": ["RhiresD", "TabsD", "TminD", "TmaxD"],
        "lat": inputs_merged.lat.values,
        "lon": inputs_merged.lon.values,
    },
    name="downscaled"
)
pred_da.to_netcdf("downscaled_predictions_2011_2020_ds.nc")