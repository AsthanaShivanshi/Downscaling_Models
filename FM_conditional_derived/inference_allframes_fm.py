import json
import sys
import numpy as np
import torch
import xarray as xr
from tqdm import tqdm

sys.path.append("..")
sys.path.append("../..")

from models.unet_module import DownscalingUnetLightning
from DownscalingDataModule import DownscalingDataModule
from models.components.diff.denoiser.unet import UNetModel
from models.components.diff.conditioner import AFNOConditionerNetCascade
from models.diff_module import FMContextual

base_seed = 124
num_steps = 30
num_samples = 11

def denorm_pr(x, pr_params):
    return np.exp(x * pr_params["std"] + pr_params["mean"]) - pr_params["epsilon"]

def denorm_temp(x, params):
    return x * params["std"] + params["mean"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("Dataset_Setup_I_Chronological_12km/RhiresD_scaling_params.json", "r") as f:
    pr_params = json.load(f)
with open("Dataset_Setup_I_Chronological_12km/TabsD_scaling_params.json", "r") as f:
    temp_params = json.load(f)

train_input_paths = {
    "precip": "Dataset_Setup_I_Chronological_12km/RhiresD_input_train_scaled.nc",
    "temp": "Dataset_Setup_I_Chronological_12km/TabsD_input_train_scaled.nc",
}
train_target_paths = {
    "precip": "Dataset_Setup_I_Chronological_12km/RhiresD_target_train_scaled.nc",
    "temp": "Dataset_Setup_I_Chronological_12km/TabsD_target_train_scaled.nc",
}
val_input_paths = {
    "precip": "Dataset_Setup_I_Chronological_12km/RhiresD_input_val_scaled.nc",
    "temp": "Dataset_Setup_I_Chronological_12km/TabsD_input_val_scaled.nc",
}
val_target_paths = {
    "precip": "Dataset_Setup_I_Chronological_12km/RhiresD_target_val_scaled.nc",
    "temp": "Dataset_Setup_I_Chronological_12km/TabsD_target_val_scaled.nc",
}
test_input_paths = {
    "precip": "Dataset_Setup_I_Chronological_12km/RhiresD_input_test_scaled.nc",
    "temp": "Dataset_Setup_I_Chronological_12km/TabsD_input_test_scaled.nc",
}
test_target_paths = {
    "precip": "Dataset_Setup_I_Chronological_12km/RhiresD_target_test_scaled.nc",
    "temp": "Dataset_Setup_I_Chronological_12km/TabsD_target_test_scaled.nc",
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
test_loader = dm.test_dataloader()

# Collect all test inputs
all_test_inputs = []
for batch_inputs, _ in test_loader:
    all_test_inputs.append(batch_inputs)
test_inputs = torch.cat(all_test_inputs, dim=0)
N = test_inputs.shape[0]
spatial_shape = test_inputs.shape[2:]

params_list = [pr_params, temp_params]

with xr.open_dataset(test_target_paths['precip']) as ds:
    times = ds['time'].values

# Load model
unet_regr = DownscalingUnetLightning(
    in_ch=3,
    out_ch=2,
    features=[64, 128, 256, 512],
    channel_names=["precip", "temp"],
    precip_scaling_json="Dataset_Setup_I_Chronological_12km/RhiresD_scaling_params.json",
)
unet_regr_ckpt = torch.load(
    "LDM_conditional/trained_ckpts_optimised/12km/UNet_ckpts/LDM_conditional.models.unet_module.DownscalingUnetLightning_12km_logtransform_lr0.001_precip_loss_weight1.0_1.0_crps[]_factor0.5_pat3.ckpt.ckpt",
    map_location="cpu", weights_only=False
)["state_dict"]
unet_regr.load_state_dict(unet_regr_ckpt, strict=False)
unet_regr = unet_regr.to(device)
unet_regr.eval()

denoiser = UNetModel(
    model_channels=32,
    in_channels=2,
    out_channels=2,
    num_res_blocks=2,
    attention_resolutions=[1, 2, 4],
    context_ch=[32, 64, 128],
    channel_mult=[1, 2, 4],
    conv_resample=True,
    dims=2,
    use_fp16=False,
    num_heads=2,
)
conditioner = AFNOConditionerNetCascade(
    autoencoder=None,
    input_channels=[2],
    embed_dim=[32, 64, 128],
    analysis_depth=3,
    cascade_depth=3,
    context_ch=[32, 64, 128],
)
fm_model = FMContextual(
    denoiser=denoiser,
    context_encoder=conditioner,
    loss_type="l2",
    use_ema=True,
    ema_decay=0.9999,
    lr=1e-4,
    source_init="noise",
)
fm_ckpt = torch.load(
    "FM_conditional_derived/trained_ckpts/12km/VPFM_L2.ckpt",
    map_location=device,
)


fm_model.load_state_dict(fm_ckpt["state_dict"], strict=False)
fm_model = fm_model.to(device)
fm_model.eval()
if fm_model.context_encoder is not None:
    fm_model.context_encoder.eval()

# Allocate output array: (N, num_samples, 2, H, W)
fm_all = np.empty((N, num_samples, 2, *spatial_shape), dtype=np.float32)

for idx in tqdm(range(N), desc="Downscaling frames"):
    input_sample = test_inputs[idx:idx+1].to(device)
    with torch.no_grad():
        unet_pred = unet_regr(input_sample)
    for j in range(num_samples):
        torch.manual_seed(base_seed + j)
        np.random.seed(base_seed + j)
        with torch.no_grad():
            fm_pred = fm_model.sample(
                x=input_sample[:, :2],
                num_steps=num_steps,
                use_ema=True,
                solver="heun2",
                init_noise_std=1.0,
                coarse_pred=unet_pred
            )
        final_pred_np = fm_pred[0].detach().cpu().numpy()
        fm_pred_denorm = np.empty_like(final_pred_np)
        for i, params in enumerate(params_list):
            if i == 0:
                fm_pred_denorm[i] = denorm_pr(final_pred_np[i], pr_params)
            else:
                fm_pred_denorm[i] = denorm_temp(final_pred_np[i], params)
        fm_all[idx, j] = fm_pred_denorm

# Save as xarray Dataset
with xr.open_dataset(test_input_paths['precip']) as ds:
    lat2d = ds["lat"].values if "lat" in ds else None
    lon2d = ds["lon"].values if "lon" in ds else None

fm_preds_np = np.transpose(fm_all, (0, 1, 3, 4, 2))  # (time, sample, y, x, channel)
var_names = ["precip", "temp"]

ds_fm = xr.Dataset(
    {
        var: (("time", "sample", "y", "x"), fm_preds_np[:, :, :, :, i])
        for i, var in enumerate(var_names)
    },
    coords={
        "time": times,
        "sample": np.arange(num_samples),
        "y": np.arange(spatial_shape[0]),
        "x": np.arange(spatial_shape[1]),
        "lat": (("y", "x"), lat2d) if lat2d is not None else None,
        "lon": (("y", "x"), lon2d) if lon2d is not None else None,
    }
)
encoding = {var: {"_FillValue": np.nan} for var in var_names}
ds_fm.to_netcdf("FM_conditional_derived/output_inference/fm_downscaled_30steps_test_set_11samples_2011_2023.nc", encoding=encoding)
print(f"FM downscaled test set saved with shape: {fm_preds_np.shape}")