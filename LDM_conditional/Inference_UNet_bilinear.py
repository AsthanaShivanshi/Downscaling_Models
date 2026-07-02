import os
import json
import numpy as np
import torch
import xarray as xr
from tqdm import tqdm
import sys

sys.path.append("..")
sys.path.append("../..")

from models.unet_module import DownscalingUnetLightning
from DownscalingDataModule import DownscalingDataModule



Swiss_mask = xr.open_dataset("Dataset_Setup_I_Chronological_12km/Swiss_HR_mask.nc")["TabsD"]



def denorm_pr(x, pr_params):
    return np.exp(x * pr_params["std"] + pr_params["mean"]) - pr_params["epsilon"]


def denorm_temp(x, params):
    return x * params["std"] + params["mean"]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("Dataset_Setup_I_Chronological_12km/RhiresD_bilinear_scaling_params.json", "r") as f:
    pr_params = json.load(f)
with open("Dataset_Setup_I_Chronological_12km/TabsD_bilinear_scaling_params.json", "r") as f:
    temp_params = json.load(f)

train_input_paths = {
    "precip": "Dataset_Setup_I_Chronological_12km/RhiresD_bilinear_input_train_scaled.nc",
    "temp": "Dataset_Setup_I_Chronological_12km/TabsD_bilinear_input_train_scaled.nc",
}
train_target_paths = {
    "precip": "Dataset_Setup_I_Chronological_12km/RhiresD_bilinear_target_train_scaled.nc",
    "temp": "Dataset_Setup_I_Chronological_12km/TabsD_bilinear_target_train_scaled.nc",
}
val_input_paths = {
    "precip": "Dataset_Setup_I_Chronological_12km/RhiresD_bilinear_input_val_scaled.nc",
    "temp": "Dataset_Setup_I_Chronological_12km/TabsD_bilinear_input_val_scaled.nc",
}
val_target_paths = {
    "precip": "Dataset_Setup_I_Chronological_12km/RhiresD_bilinear_target_val_scaled.nc",
    "temp": "Dataset_Setup_I_Chronological_12km/TabsD_bilinear_target_val_scaled.nc",
}
test_input_paths = {
    "precip": "Dataset_Setup_I_Chronological_12km/RhiresD_bilinear_input_test_scaled.nc",
    "temp": "Dataset_Setup_I_Chronological_12km/TabsD_bilinear_input_test_scaled.nc",
}
test_target_paths = {
    "precip": "Dataset_Setup_I_Chronological_12km/RhiresD_bilinear_target_test_scaled.nc",
    "temp": "Dataset_Setup_I_Chronological_12km/TabsD_bilinear_target_test_scaled.nc",
}
elevation_path = "elevation.tif"



dm = DownscalingDataModule(

    train_input=train_input_paths,
    train_target=train_target_paths,
    val_input=val_input_paths,
    val_target=val_target_paths,
    test_input=test_input_paths,
    test_target=test_target_paths,
    elevation=elevation_path,
    batch_size=32,
    num_workers=4,
    preprocessing={
        "variables": {
            "input": {"precip": "RhiresD", "temp": "TabsD"},
            "target": {"precip": "RhiresD", "temp": "TabsD"},
        },
        "preprocessing": {"nan_to_num": True, "nan_value": 0.0},
    },
)



dm.setup()


val_loader = dm.val_dataloader()

all_val_inputs = []

for batch_inputs, _ in val_loader:
    all_val_inputs.append(batch_inputs)
val_inputs = torch.cat(all_val_inputs, dim=0)
N = val_inputs.shape[0]
spatial_shape = val_inputs.shape[2:]  # (H, W)

with xr.open_dataset(val_target_paths["precip"]) as ds:
    times = ds["time"].values
with xr.open_dataset(val_input_paths["precip"]) as ds:
    lat2d = ds["lat"].values if "lat" in ds else None
    lon2d = ds["lon"].values if "lon" in ds else None


unet_all = np.empty((N, 2, *spatial_shape), dtype=np.float32)



unet_regr = DownscalingUnetLightning(
    in_ch=3,
    out_ch=2,
    features=[64, 128, 256, 512],
    channel_names=["precip", "temp"],
    precip_scaling_json="Dataset_Setup_I_Chronological_12km/RhiresD_bilinear_scaling_params.json",
)


unet_regr_ckpt = torch.load(
    "LDM_conditional/trained_ckpts/12km/BILINEAR_LDM_conditional.models.unet_module.DownscalingUnetLightning_bs32_lr0.001_delta1.0_factor0.5_pat3.ckpt",
    map_location="cpu",
    weights_only=False,

)["state_dict"]



unet_regr.load_state_dict(unet_regr_ckpt, strict=False)
unet_regr = unet_regr.to(device)
unet_regr.eval()

params_list = [pr_params, temp_params]

for idx in tqdm(range(N), desc="UNet inference (val)"):
    with torch.no_grad():
        input_sample = val_inputs[idx].unsqueeze(0).to(device)  # (1, C_in, H, W)
        unet_pred = unet_regr(input_sample)[0].cpu().numpy()    # (2, H, W)

        unet_pred_denorm = np.empty_like(unet_pred)
        for i, params in enumerate(params_list):
            if i == 0:
                unet_pred_denorm[i] = denorm_pr(unet_pred[i], pr_params)
            else:
                unet_pred_denorm[i] = denorm_temp(unet_pred[i], params)

        unet_all[idx] = unet_pred_denorm



unet_preds_np = np.transpose(unet_all, (0, 2, 3, 1))
var_names = ["precip", "temp"]




ds_unet = xr.Dataset(
    {var: (("time", "N", "E"), unet_preds_np[:, :, :, i])   # shape (time, H=240, W=370) — no inner transpose
     for i, var in enumerate(var_names)},
    coords={
        "time": times,
        "N": Swiss_mask.coords["N"].values,   # length 240
        "E": Swiss_mask.coords["E"].values,   # length 370
        "lat": (("N", "E"), lat2d) if lat2d is not None else None,
        "lon": (("N", "E"), lon2d) if lon2d is not None else None,
    },
)

output_dir = "DDIM_conditional_derived/output_inference"
os.makedirs(output_dir, exist_ok=True)
out_path = os.path.join(output_dir, "BILINEAR_unet_downscaled_val_set.nc")

# Swiss_mask is (N, E) — broadcasts automatically over time;;; Asthan aSh



ds_unet["precip"] = ds_unet["precip"].where(Swiss_mask)
ds_unet["temp"]   = ds_unet["temp"].where(Swiss_mask)

ds_unet.to_netcdf(out_path)
print(f"Saved val UNet inference: {out_path} | shape={unet_preds_np.shape}")