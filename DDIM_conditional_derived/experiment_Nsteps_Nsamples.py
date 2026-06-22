import numpy as np
import torch
import json
import sys
from tqdm import tqdm
import xarray as xr
import os
import time
import pandas as pd
import gc
import argparse

sys.path.append("..")
sys.path.append("../..")

import warnings
warnings.filterwarnings("ignore")

from models.unet_module import DownscalingUnetLightning
from DownscalingDataModule import DownscalingDataModule
from models.components.diff.denoiser.unet import UNetModel
from models.components.diff.denoiser.ddim import DDIMSampler
from models.components.diff.conditioner import AFNOConditionerNetCascade
from models.diff_module import DDIMResidualContextual

from concurrent.futures import ThreadPoolExecutor

import properscoring as ps
from skimage.metrics import structural_similarity as ssim

#--------------------------------------------------------------------------------#

eta = 0.0  # For DDIM sampling : deterministic.
base_seed = 124
Y = 2010

# Exp parameters:

parser = argparse.ArgumentParser()
parser.add_argument("--S", type=int, default=None)
parser.add_argument("--num_samples", type=int, default=None)
args = parser.parse_args()

if args.S is not None and args.num_samples is not None:
    denoising_steps_list = [args.S]
    num_samples_list = [args.num_samples]
else:
    denoising_steps_list = [10, 20, 30, 50, 100, 250]
    num_samples_list = [2, 4, 6, 10, 15]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#--------------------------------------------------------------------------------#

def denorm_pr(x, pr_params):
    return np.exp(x * pr_params['std'] + pr_params['mean']) - pr_params['epsilon']

def denorm_temp(x, params):
    return x * params['std'] + params['mean']

#--------------------------------------------------------------------------------#

def pooled_ssim(gt, pred, mask):
    if pred.ndim == 4:  # (S, T, H, W)
        pred = pred.mean(axis=0)  # mean over samples
    ssim_vals = []
    for t in range(gt.shape[0]):
        m = mask if mask.ndim == 2 else mask[t]
        if np.any(m):
            ssim_val = ssim(gt[t], pred[t], data_range=np.nanmax(gt[t]) - np.nanmin(gt[t]), win_size=11, gaussian_weights=True, use_sample_covariance=False, multichannel=False, mask=m)
            ssim_vals.append(ssim_val)
    return np.mean(ssim_vals)

def crps_score(gt, pred, mask):
    if pred.ndim == 4:  # (S, T, H, W)
        pred = np.moveaxis(pred, 0, 1)  # (T, S, H, W)
        pred = pred.reshape(pred.shape[0], -1, pred.shape[-2]*pred.shape[-1])  # (T, S, H*W)
        gt = gt.reshape(gt.shape[0], -1)  # (T, H*W)
        mask = mask.reshape(gt.shape)
        crps_vals = []
        for t in range(gt.shape[0]):
            valid = mask[t].flatten()
            if np.any(valid):
                crps = ps.crps_ensemble(gt[t][valid], pred[t][:, valid].T).mean()
                crps_vals.append(crps)
        return np.mean(crps_vals)
    else:
        gt = gt.reshape(gt.shape[0], -1)
        pred = pred.reshape(pred.shape[0], -1)
        mask = mask.reshape(gt.shape)
        crps_vals = []
        for t in range(gt.shape[0]):
            valid = mask[t].flatten()
            if np.any(valid):
                crps = ps.crps_ensemble(gt[t][valid], pred[t][valid][:, None]).mean()
                crps_vals.append(crps)
        return np.mean(crps_vals)

#--------------------------------------------------------------------------------#

with xr.open_dataset("Dataset_Setup_I_Chronological_12km/RhiresD_target_train_scaled.nc") as ds:
    precip_mask_full = ~np.isnan(ds["RhiresD"].isel(time=0).load().values)  # shape [H_full, W_full]

with open("Dataset_Setup_I_Chronological_12km/RhiresD_scaling_params.json", 'r') as f:
    pr_params = json.load(f)
with open("Dataset_Setup_I_Chronological_12km/TabsD_scaling_params.json", 'r') as f:
    temp_params = json.load(f)

print("params loaded")

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

#--------------------------------------------------------------------------------#

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
val_loader = dm.val_dataloader()

print("DataLoader ready")




with xr.open_dataset(val_target_paths['precip']) as ds:
    times = ds['time'].load().values
    year_mask = (times >= np.datetime64(f'{Y}-01-01')) & (times < np.datetime64(f'{Y+1}-01-01'))
    val_indices = np.where(year_mask)[0]

# ref masks: for Y
with xr.open_dataset("Dataset_Setup_I_Chronological_12km/TabsD_step1_latlon.nc") as ds_temp:
    ref_temp = ds_temp["TabsD"].sel(time=slice(f"{Y}-01-01", f"{Y}-12-31")).load()
    mask_temp = ~np.isnan(ref_temp.values)
with xr.open_dataset("Dataset_Setup_I_Chronological_12km/RhiresD_step1_latlon.nc") as ds_precip:
    ref_precip = ds_precip["RhiresD"].sel(time=slice(f"{Y}-01-01", f"{Y}-12-31")).load()
    mask_precip = ~np.isnan(ref_precip.values)

gc.collect()

print("Reference masks ready")

val_inputs = []
val_targets = []
for i, (batch_input, batch_targets) in enumerate(dm.val_dataloader()):
    for j in range(batch_input.shape[0]):
        idx = i * dm.batch_size + j
        if idx in val_indices:
            val_inputs.append(batch_input[j].cpu())
            val_targets.append(batch_targets[j].cpu())
del batch_input, batch_targets
gc.collect()

val_inputs = torch.stack(val_inputs)
val_targets = torch.stack(val_targets)
N = val_inputs.shape[0]
spatial_shape = val_inputs.shape[2:]

params_list = [pr_params, temp_params]

#--------------------------------------------------------------------------------#
# UNet inference caching
unet_save_path = f"DDIM_conditional_derived/output_inference/unet_downscaled_val_set_year_{Y}.npy"

if os.path.exists(unet_save_path):
    print("Loading UNet predictions from file...")
    unet_all = np.load(unet_save_path)
else:
    print("Running UNet inference...")
    unet_all = np.empty((N, 2, *spatial_shape), dtype=np.float32)
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
        map_location=device, weights_only=False
    )["state_dict"]

    unet_regr.load_state_dict(unet_regr_ckpt, strict=False)
    unet_regr = unet_regr.to(device)
    unet_regr.eval()
    del unet_regr_ckpt
    gc.collect()

    print("UNet ready")

    for idx in tqdm(range(N), desc="UNet inference"):
        with torch.no_grad():
            inp = val_inputs[idx].unsqueeze(0).to(device)
            pred = unet_regr(inp).cpu().numpy()[0]
            unet_all[idx] = pred
            del inp, pred
            gc.collect()
    np.save(unet_save_path, unet_all)
    print(f"UNet predictions saved to {unet_save_path}")

target_all = val_targets.numpy()

#--------------------------------------------------------------------------------#
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

ddim = DDIMResidualContextual(
    denoiser=denoiser,
    context_encoder=conditioner,
    timesteps=1000,
    parameterization="v",
    loss_type="l1",
    beta_schedule="cosine",
    linear_start=1e-4,
    linear_end=2e-2,
    cosine_s=8e-3,
    use_ema=True,
    ema_decay=0.9999,
    lr=1e-4
)

ddim_ckpt = torch.load(
    "DDIM_conditional_derived/trained_ckpts/12km/DDIM_checkpoint_L1_cosine_schedule_loss_parameterisation_v.ckpt",
    map_location=device
)

ddim.load_state_dict(ddim_ckpt["state_dict"], strict=False)
ddim = ddim.to(device)
ddim.eval()
sampler = DDIMSampler(ddim, device=device)
del ddim_ckpt
gc.collect()
print("DDIM ready")

#--------------------------------------------------------------------------------#

def ddim_sample_worker(j,
    base_seed,
    sample_shape, context, sampler, eta,
    unet_pred, pr_params, temp_params,
    params_list, device, S):

    torch.manual_seed(base_seed + j)
    np.random.seed(base_seed + j)

    z = torch.randn((1, *sample_shape), device=device)
    with torch.no_grad():
        residual, _ = sampler.sample(
            S=S,
            batch_size=1,
            shape=sample_shape,
            conditioning=context,
            eta=eta,
            verbose=False,
            x_T=z,
            schedule="cosine",
        )

        final_pred = unet_pred + residual
        final_pred_np = final_pred[0].cpu().numpy()
        ddim_pred_denorm = np.empty_like(final_pred_np)
        for i, params in enumerate(params_list):
            ddim_pred_denorm[i] = denorm_pr(final_pred_np[i], pr_params) if i == 0 else denorm_temp(final_pred_np[i], params)
        del residual, final_pred, final_pred_np
        gc.collect()
    return ddim_pred_denorm

#--------------------------------------------------------------------------------#

results = []

print("Running DDIM inference...")

for S in tqdm(denoising_steps_list, desc="Denoising steps"):
    for num_samples in tqdm(num_samples_list, desc="Num samples", leave=False):
        out_path = f"DDIM_conditional_derived/output_inference/ddim_samples_year_{Y}_S{S}_samples{num_samples}.nc"
        metrics_path = f"DDIM_conditional_derived/Metrics_Test_Set/outputs/ddim_inference_experiment_results_S{S}_samples{num_samples}.csv"
        # Skip if already done
        if os.path.exists(out_path) and os.path.exists(metrics_path):
            print(f"Skipping S={S}, num_samples={num_samples} (already done)")
            continue

        print(f"Starting inference for S={S}, num_samples={num_samples}...")

        ddim_all = np.empty((N, num_samples, 2, *spatial_shape), dtype=np.float32)
        start_time = time.time()
        for idx in tqdm(range(N), desc=f"Inference S={S}, samples={num_samples}", leave=False):
            with torch.no_grad():
                # Use already computed UNet prediction
                unet_pred = torch.from_numpy(unet_all[idx]).unsqueeze(0).to(device)
                context = [(unet_pred, None)]
                sample_shape = unet_pred.shape[1:]

                with ThreadPoolExecutor(max_workers=num_samples) as executor:
                    futures = [
                        executor.submit(
                            ddim_sample_worker,
                            j,
                            base_seed,
                            sample_shape,
                            context,
                            sampler,
                            eta,
                            unet_pred,
                            pr_params,
                            temp_params,
                            params_list,
                            device,
                            S
                        )
                        for j in range(num_samples)
                    ]
                    for j, future in enumerate(futures):
                        ddim_all[idx, j] = future.result()
                del unet_pred, context, sample_shape, futures
                gc.collect()
        elapsed_time = time.time() - start_time

        # Save DDIM samples
        print(f"Saving results to {out_path}...")

        ddim_preds_np = np.transpose(ddim_all, (0, 1, 2, 3, 4))  # (time, sample, channel, y, x)
        ddim_preds_np = np.transpose(ddim_preds_np, (0, 1, 3, 4, 2))  # (time, sample, y, x, channel)
        ds_ddim = xr.Dataset(
            {
                var: (("time", "sample", "y", "x"), ddim_preds_np[:, :, :, :, i])
                for i, var in enumerate(["precip", "temp"])
            },
            coords={
                "time": np.arange(N),
                "sample": np.arange(num_samples),
                "y": np.arange(spatial_shape[0]),
                "x": np.arange(spatial_shape[1]),
            }
        )

        encoding = {var: {"_FillValue": np.nan} for var in ["precip", "temp"]}
        ds_ddim.to_netcdf(out_path, encoding=encoding)
        print(f"Saved {out_path}")
        del ds_ddim
        gc.collect()

        # (n_samples, time, lat, lon)
        pred_temp = np.moveaxis(ddim_preds_np[..., 1], 1, 0)  # (samples, time, y, x)
        pred_precip = np.moveaxis(ddim_preds_np[..., 0], 1, 0)
        pred_precip = np.where(pred_precip < 0, 0, pred_precip)

        # Compute metrics
        ssim_temp_val = pooled_ssim(ref_temp.values, pred_temp, mask_temp)
        ssim_precip_val = pooled_ssim(ref_precip.values, pred_precip, mask_precip)
        crps_temp_val = crps_score(ref_temp.values, pred_temp, mask_temp)
        crps_precip_val = crps_score(ref_precip.values, pred_precip, mask_precip)

        # Also compute UNet metrics (single deterministic prediction)
        unet_temp = unet_all[:, 1]
        unet_precip = np.where(unet_all[:, 0] < 0, 0, unet_all[:, 0])
        unet_temp = unet_temp.astype(np.float32)
        unet_precip = unet_precip.astype(np.float32)
        ssim_temp_unet = pooled_ssim(ref_temp.values, unet_temp, mask_temp)
        ssim_precip_unet = pooled_ssim(ref_precip.values, unet_precip, mask_precip)
        crps_temp_unet = crps_score(ref_temp.values, unet_temp, mask_temp)
        crps_precip_unet = crps_score(ref_precip.values, unet_precip, mask_precip)

        result_row = {
            "denoising_steps": S,
            "num_samples": num_samples,
            "inference_time_mins": elapsed_time / 60,
            "nc_file": out_path,
            "SSIM_temp_DDIM": ssim_temp_val,
            "SSIM_precip_DDIM": ssim_precip_val,
            "CRPS_temp_DDIM": crps_temp_val,
            "CRPS_precip_DDIM": crps_precip_val,
            "SSIM_temp_UNet": ssim_temp_unet,
            "SSIM_precip_UNet": ssim_precip_unet,
            "CRPS_temp_UNet": crps_temp_unet,
            "CRPS_precip_UNet": crps_precip_unet,
        }
        results.append(result_row)

        # Save metrics for this run
        pd.DataFrame([result_row]).to_csv(metrics_path, index=False)
        print(f"Metrics saved to {metrics_path}")

        print(f"Denoising steps: {S}, Num samples: {num_samples}, Inference time (in mins): {elapsed_time / 60:.2f}")

        del ddim_all, ddim_preds_np, pred_temp, pred_precip
        gc.collect()