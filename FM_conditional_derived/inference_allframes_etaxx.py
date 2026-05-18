#DDPM experiment
from pyexpat import model

import numpy as np
import torch
import json
import sys
from tqdm import tqdm
import xarray as xr
sys.path.append("..")
sys.path.append("../..")

import warnings
warnings.filterwarnings("ignore")

from models.unet_module import DownscalingUnetLightning
from DownscalingDataModule import DownscalingDataModule
from models.components.diff.denoiser.unet import UNetModel
from models.components.diff.denoiser.sample import flow_matching_sample


from models.components.diff.conditioner import AFNOConditionerNetCascade
from models.diff_module import DDIMResidualContextual


from concurrent.futures import ThreadPoolExecutor

num_samples = 6 #Deterministic sample : single run,,,,,
eta = 0.0  #For DDIM sampling : determinisitc. 
base_seed = 124

S=2 #Selected basis the sensitivity analysis : AsthanaSh


def denorm_pr(x, pr_params):
    return np.exp(x * pr_params['std'] + pr_params['mean']) - pr_params['epsilon']

def denorm_temp(x, params):
    return x * params['std'] + params['mean']



with xr.open_dataset("Dataset_Setup_I_Chronological_12km/RhiresD_target_train_scaled.nc") as ds:
    precip_mask_full = ~np.isnan(ds["RhiresD"].values[0])  # shape [H_full, W_full]



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("Dataset_Setup_I_Chronological_12km/RhiresD_scaling_params.json", 'r') as f:
    pr_params = json.load(f)
with open("Dataset_Setup_I_Chronological_12km/TabsD_scaling_params.json", 'r') as f:
    temp_params = json.load(f)

train_input_paths = {
    'precip': "Dataset_Setup_I_Chronological_12km/RhiresD_input_train_scaled.nc",
    'temp': "Dataset_Setup_I_Chronological_12km/TabsD_input_train_scaled.nc",
}
train_target_paths = {
    'precip': "Dataset_Setup_I_Chronological_12km/RhiresD_target_train_scaled.nc",
    'temp':  "Dataset_Setup_I_Chronological_12km/TabsD_target_train_scaled.nc",
}
val_input_paths = {
    'precip': "Dataset_Setup_I_Chronological_12km/RhiresD_input_val_scaled.nc",
    'temp': "Dataset_Setup_I_Chronological_12km/TabsD_input_val_scaled.nc",
}
val_target_paths = {
    'precip': "Dataset_Setup_I_Chronological_12km/RhiresD_target_val_scaled.nc",
    'temp': "Dataset_Setup_I_Chronological_12km/TabsD_target_val_scaled.nc",
}
test_input_paths = {
    'precip': "Dataset_Setup_I_Chronological_12km/RhiresD_input_test_scaled.nc",
    'temp': "Dataset_Setup_I_Chronological_12km/TabsD_input_test_scaled.nc",
}
test_target_paths = {
    'precip': "Dataset_Setup_I_Chronological_12km/RhiresD_target_test_scaled.nc",
    'temp': "Dataset_Setup_I_Chronological_12km/TabsD_target_test_scaled.nc",
}
elevation_path = 'elevation.tif'

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
        'variables': {
            'input': {'precip': 'RhiresD', 'temp': 'TabsD'},
            'target': {'precip': 'RhiresD', 'temp': 'TabsD'}
        },
        'preprocessing': {'nan_to_num': True, 'nan_value': 0.0}
    }
)

dm.setup()
test_loader = dm.test_dataloader()

all_test_inputs = []
all_test_targets = []
for batch_inputs, batch_targets in test_loader:
    all_test_inputs.append(batch_inputs)
    all_test_targets.append(batch_targets)


    
test_inputs = torch.cat(all_test_inputs, dim=0)
test_targets = torch.cat(all_test_targets, dim=0)
N = test_inputs.shape[0]
spatial_shape = test_inputs.shape[2:]  # (H, W)


params_list = [pr_params, temp_params]
with xr.open_dataset(test_target_paths['precip']) as ds:
    times = ds['time'].values


unet_all = np.empty((N, 2, *spatial_shape), dtype=np.float32)
target_all = np.empty((N, 2, *spatial_shape), dtype=np.float32)

# UNet
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
    num_heads=2
 )
conditioner = AFNOConditionerNetCascade(
     autoencoder=None,
     input_channels=[2],
     embed_dim=[32, 64, 128],
     analysis_depth=3,
     cascade_depth=3,
     context_ch=[32, 64, 128]
 )


fm_model = DDIMResidualContextual(
    denoiser=denoiser,
    context_encoder=conditioner,
    loss_type="l1",
    use_ema=True,
    ema_decay=0.9999,
    lr=1e-4
)
fm_ckpt = torch.load(
    "FM_conditional_derived/trained_ckpts/12km/FM_checkpoint.ckpt",  # <-- use your FM checkpoint
    map_location=device
)
fm_model.load_state_dict(fm_ckpt["state_dict"], strict=False)
fm_model = fm_model.to(device)
fm_model.eval()


fm_all = np.empty((N, num_samples, 2, *spatial_shape), dtype=np.float32)



def fm_sample_worker(j, base_seed, sample_shape, context, model, unet_pred, pr_params, temp_params, params_list, device, steps=50):
    torch.manual_seed(base_seed + j)
    np.random.seed(base_seed + j)
    z = torch.randn((1, *sample_shape), device=device)
    residual = flow_matching_sample(
        model,
        context,
        shape=sample_shape,
        steps=steps,
        device=device,
        x_T=z,
        verbose=False
    )
    final_pred = unet_pred + residual
    final_pred_np = final_pred[0].cpu().numpy()
    fm_pred_denorm = np.empty_like(final_pred_np)
    for i, params in enumerate(params_list):
        fm_pred_denorm[i] = denorm_pr(final_pred_np[i], pr_params) if i == 0 else denorm_temp(final_pred_np[i], params)
    return fm_pred_denorm


for idx in tqdm(range(N), desc="Downscaling frames"):
    with torch.no_grad():
        input_sample = test_inputs[idx].unsqueeze(0).to(device)
        unet_pred = unet_regr(input_sample)
        context = [(unet_pred, None)]
        sample_shape = unet_pred.shape[1:]
        target_np = test_targets[idx][:unet_pred.shape[1]].cpu().numpy()

        unet_pred_np = unet_pred[0].cpu().numpy()
        unet_pred_denorm = np.empty_like(unet_pred_np)
        target_denorm = np.empty_like(target_np)
        for i, params in enumerate(params_list):
                unet_pred_denorm[i] = denorm_pr(unet_pred_np[i], pr_params) if i == 0 else denorm_temp(unet_pred_np[i], params)
                target_denorm[i] = denorm_pr(target_np[i], pr_params) if i == 0 else denorm_temp(target_np[i], params)
        unet_all[idx] = unet_pred_denorm
        target_all[idx] = target_denorm

            # Parallelising over samples
    with ThreadPoolExecutor(max_workers=num_samples) as executor:
        futures = [
            executor.submit(
                fm_sample_worker, j, base_seed, sample_shape, context, fm_model,
                unet_pred, pr_params, temp_params, params_list, device, 2
            )
            for j in range(num_samples)
        ]
        for j, future in enumerate(futures):
            fm_all[idx, j] = future.result()



with xr.open_dataset(test_input_paths['precip']) as ds:
    lat2d = ds["lat"].values if "lat" in ds else None
    lon2d = ds["lon"].values if "lon" in ds else None

unet_preds_np = np.transpose(unet_all, (0, 2, 3, 1))  # (time, y, x, channel)
target_np = np.transpose(target_all, (0, 2, 3, 1))    # (time, y, x, channel)

var_names = ["precip", "temp"]

# UNet pred
ds_unet = xr.Dataset(
    {
        var: (("time", "y", "x"), unet_preds_np[:, :, :, i])
        for i, var in enumerate(var_names)
    },
    coords={
        "time": times,
        "y": np.arange(spatial_shape[0]),
        "x": np.arange(spatial_shape[1]),
        "lat": (("y", "x"), lat2d) if lat2d is not None else None,
        "lon": (("y", "x"), lon2d) if lon2d is not None else None,
    }
)





 #FM sample

fm_preds_np = np.transpose(fm_all, (0, 1, 2, 3, 4))
fm_preds_np = np.transpose(fm_preds_np, (0, 1, 3, 4, 2))

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

ds_fm.to_netcdf("FM_conditional_derived/output_inference/fm_downscaled_Euler_steps_test_set_6samples_2011_2023.nc", encoding=encoding)
print(f"FM downscaled test set saved with shape: {fm_preds_np.shape}")