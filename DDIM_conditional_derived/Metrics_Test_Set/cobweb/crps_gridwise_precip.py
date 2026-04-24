import os

import pandas as pd
import xarray as xr
import numpy as np
from properscoring import crps_ensemble
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def crps_cell(i, j, obs_arr, ens_arr):
    obs_series = obs_arr[:, i, j]
    ens_series = ens_arr[:, :, i, j]
    mask = ~np.isnan(obs_series)
    if np.sum(mask) < 2:
        return (i, j, np.nan)
    obs_valid = obs_series[mask]
    ens_valid = ens_series[:, mask]
    if obs_valid.shape[0] == 0:
        return (i, j, np.nan)
    crps_vals = crps_ensemble(obs_valid, ens_valid.T, fair=True)
    return (i, j, np.mean(crps_vals))

def gridwise_temporal_crps(obs, ens_pred, n_workers=os.cpu_count()):
    obs_arr = obs.values
    ens_arr = ens_pred
    T, N, E = obs_arr.shape
    indices = [(i, j) for i in range(N) for j in range(E)]
    crps_grid = np.full((N, E), np.nan)
    count = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(crps_cell, i, j, obs_arr, ens_arr): (i, j) for i, j in indices}
        for future in tqdm(as_completed(futures), total=len(futures), desc="CRPS grid cells", miniters=1000):
            i, j, value = future.result()
            crps_grid[i, j] = value
            count += 1
            if count % 1000 == 0:
                print(f"Processed {count} CRPS grid cells")
    return np.nanmean(crps_grid)



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
    # Align to only overlapping coordinates (inner join)
    obs_aligned, pred_aligned = xr.align(obs_precip, pred, join="inner")
    crps = gridwise_temporal_crps(obs_aligned, pred_aligned)
    metrics[name] = crps


#--------------------------------------------------------------------#
metric_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["CRPS"])
metric_df.to_csv("DDIM_conditional_derived/Metrics_Test_Set/outputs/crps_allmodels_precip.csv")

#--------------------------------------------------------------------#


