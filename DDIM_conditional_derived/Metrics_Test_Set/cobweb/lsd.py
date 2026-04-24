import os

import pandas as pd

import xarray as xr
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def lsd_idx(idx, obs_flat, pred_flat, n_fft, eps):
    obs_valid = obs_flat[:, idx]
    pred_valid = pred_flat[:, idx]
    if np.sum(~np.isnan(obs_valid)) < n_fft or np.sum(~np.isnan(pred_valid)) < n_fft:
        return idx, np.nan
    obs_fft = np.fft.rfft(obs_valid, n=n_fft)
    pred_fft = np.fft.rfft(pred_valid, n=n_fft)
    obs_log = np.log(np.abs(obs_fft) + eps)
    pred_log = np.log(np.abs(pred_fft) + eps)
    return idx, np.sqrt(np.mean((obs_log - pred_log) ** 2))

def gridwise_temporal_lsd(obs, pred, n_fft=256, eps=1e-8, n_workers=os.cpu_count()):
    obs_arr = obs.values
    pred_arr = pred.values
    mask = ~np.isnan(obs_arr) & ~np.isnan(pred_arr)
    obs_arr = np.where(mask, obs_arr, np.nan)
    pred_arr = np.where(mask, pred_arr, np.nan)
    T, N, E = obs_arr.shape
    obs_flat = obs_arr.reshape(T, -1)
    pred_flat = pred_arr.reshape(T, -1)
    valid_mask = ~np.isnan(obs_flat).any(axis=0) & ~np.isnan(pred_flat).any(axis=0)
    indices = np.where(valid_mask)[0]
    lsd = np.full(obs_flat.shape[1], np.nan)
    count = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(lsd_idx, idx, obs_flat, pred_flat, n_fft, eps): idx
            for idx in indices
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="LSD grid cells", miniters=1000):
            idx, value = future.result()
            lsd[idx] = value
            count += 1
            if count % 1000 == 0:
                print(f"Processed {count} LSD grid cells")
    return np.nanmean(lsd)

#--------------------------------------------------------------------#

obs_precip = xr.open_dataset('Dataset_Setup_I_Chronological_12km/RhiresD_step1_latlon.nc')["RhiresD"].sel(time=slice("2011-01-01","2023-12-31"))
unet_precip = xr.open_dataset("DDIM_conditional_derived/output_inference/UNet_downscaled_test_set_2011_2023.nc")["precip"].sel(time=slice("2011-01-01","2023-12-31"))
coarse_precip = xr.open_dataset("Dataset_Setup_I_Chronological_12km/RhiresD_step2_coarse.nc")["RhiresD"].sel(time=slice("2011-01-01","2023-12-31"))

coarse_precip_interp = coarse_precip.interp(
    N=obs_precip.N, E=obs_precip.E, method="nearest"
)

bicubic_precip = xr.open_dataset("Dataset_Setup_I_Chronological_12km/RhiresD_step3_interp.nc")["RhiresD"].sel(time=slice("2011-01-01","2023-12-31"))
ddim_precip = xr.open_dataset("DDIM_conditional_derived/output_inference/ddim_downscaled_100steps_test_set_5samples_eta_0.0.nc")["precip"].sel(time=slice("2011-01-01", "2023-12-31"))
#--------------------------------------------------------------------#




# Only for precip 
obs_precip = obs_precip.where(obs_precip >= 0)
unet_precip = unet_precip.where(unet_precip >= 0)
coarse_precip = coarse_precip.where(coarse_precip >= 0)
coarse_precip_interp = coarse_precip_interp.where(coarse_precip_interp >= 0)
bicubic_precip = bicubic_precip.where(bicubic_precip >= 0)
ddim_precip = ddim_precip.where(ddim_precip >= 0)

if "sample" in ddim_precip.dims:
    ddim_ens_precip = ddim_precip.rename({"sample": "ensemble"})
    ddim_ens_precip = ddim_ens_precip.where(ddim_ens_precip >= 0)

models = {
    "Coarse": coarse_precip_interp,
    "Bicubic": bicubic_precip,
    "UNet": unet_precip,
    "DDIM": ddim_ens_precip.mean(dim="ensemble")
}

metrics = {}

for name, pred in models.items():
    lsd = gridwise_temporal_lsd(obs_precip, pred)
    metrics[name] = lsd


#--------------------------------------------------------------------#
metric_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["LSD"])
metric_df.to_csv("DDIM_conditional_derived/Metrics_Test_Set/outputs/lsd_allmodels_precip.csv")

#--------------------------------------------------------------------#