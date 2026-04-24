import pandas as pd

import os
import sys
import xarray as xr
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def pitd_cell(i, j, obs, ens_pred, bin_edges):
    obs_series = obs[:, i, j].values
    ens_series = ens_pred[:, :, i, j]
    mask = ~np.isnan(obs_series)
    if np.sum(mask) < 2:
        return (i, j, np.nan)
    obs_valid = obs_series[mask]
    ens_valid = ens_series[:, mask]
    pit = []
    for t in range(ens_valid.shape[1]):
        ecdf = ECDF(obs_valid)
        for k in range(ens_valid.shape[0]):
            pit.append(ecdf(ens_valid[k, t]))
    pit = np.array(pit)
    if pit.size == 0:
        return (i, j, np.nan)
    pit_hist, _ = np.histogram(pit, bins=bin_edges, density=True)
    pit_hist = pit_hist / pit_hist.sum()
    uniform = np.ones_like(pit_hist) / len(pit_hist)
    return (i, j, np.sqrt(np.mean((pit_hist - uniform) ** 2)))


    

def gridwise_pitd_rmse(obs, ens_pred, bins=20, n_workers=os.cpu_count()):
    N, E = obs.shape[1], obs.shape[2]
    bin_edges = np.linspace(0, 1, bins + 1)
    indices = [(i, j) for i in range(N) for j in range(E)]
    pitd_grid = np.full((N, E), np.nan)
    count = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(pitd_cell, i, j, obs, ens_pred, bin_edges): (i, j) for i, j in indices}
        for future in tqdm(as_completed(futures), total=len(futures), desc="PITD grid cells", miniters=1000):
            i, j, value = future.result()
            pitd_grid[i, j] = value
            count += 1
            if count % 1000 == 0:
                print(f"Processed {count} PITD RMSE for grid cell")
    return np.nanmean(pitd_grid)




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
    pitd = gridwise_pitd_rmse(obs_precip, pred)
    metrics[name] = pitd


#--------------------------------------------------------------------#
metric_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["PITD"])
metric_df.to_csv("DDIM_conditional_derived/Metrics_Test_Set/outputs/pitd_allmodels_precip.csv")

#--------------------------------------------------------------------#