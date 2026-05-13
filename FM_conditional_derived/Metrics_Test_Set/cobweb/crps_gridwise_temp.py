import os
import pandas as pd
import xarray as xr
import numpy as np
from properscoring import crps_ensemble

def ensure_NE(da, ref):
    if "y" in da.dims and "x" in da.dims:
        da = da.rename({"y": "N", "x": "E"})
    if "N" in da.dims and "E" in da.dims:
        da = da.assign_coords(N=ref.N, E=ref.E)
    return da

def gridwise_temporal_crps(obs, ens_pred):
    obs_arr = obs.values  # (T, N, E)
    ens_arr = ens_pred    # (ensemble, T, N, E)
    T, N, E = obs_arr.shape
    crps_grid = np.full((N, E), np.nan)
    for i in range(N):
        for j in range(E):
            obs_series = obs_arr[:, i, j]
            ens_series = ens_arr[:, :, i, j]
            mask = ~np.isnan(obs_series)
            if np.sum(mask) < 2:
                continue
            obs_valid = obs_series[mask]
            ens_valid = ens_series[:, mask]
            if obs_valid.shape[0] == 0:
                continue
            crps_vals = crps_ensemble(obs_valid, ens_valid.T)
            crps_grid[i, j] = np.nanmean(crps_vals)
    return crps_grid

#--------------------------------------------------------------------#

obs_temp = xr.open_dataset('Dataset_Setup_I_Chronological_12km/TabsD_step1_latlon.nc')["TabsD"].sel(time=slice("2011-01-01","2023-12-31"))
unet_temp = xr.open_dataset("DDIM_conditional_derived/output_inference/UNet_downscaled_test_set_2011_2023.nc")["temp"].sel(time=slice("2011-01-01","2023-12-31"))
coarse_temp = xr.open_dataset("Dataset_Setup_I_Chronological_12km/TabsD_step2_coarse.nc")["TabsD"].sel(time=slice("2011-01-01","2023-12-31"))
bicubic_temp = xr.open_dataset("Dataset_Setup_I_Chronological_12km/TabsD_step3_interp.nc")["TabsD"].sel(time=slice("2011-01-01","2023-12-31"))
ddim_temp = xr.open_dataset("DDIM_conditional_derived/output_inference/ddim_downscaled_30steps_test_set_11samples_2011_2023.nc")["temp"].sel(time=slice("2011-01-01", "2023-12-31"))

coarse_temp_interp = coarse_temp.interp_like(obs_temp, method="nearest")

obs_temp = ensure_NE(obs_temp, obs_temp)
unet_temp = ensure_NE(unet_temp, obs_temp)
coarse_temp_interp = ensure_NE(coarse_temp_interp, obs_temp)
bicubic_temp = ensure_NE(bicubic_temp, obs_temp)

if "sample" in ddim_temp.dims:
    ddim_ens_temp = ddim_temp.rename({"sample": "ensemble"})
else:
    raise ValueError("Expected 'sample' dimension in ddim_temp")
ddim_ens_temp = ensure_NE(ddim_ens_temp, obs_temp)

models = {
    "Coarse": coarse_temp_interp.expand_dims(ensemble=[0]).transpose("ensemble", "time", "N", "E"),
    "Bicubic": bicubic_temp.expand_dims(ensemble=[0]).transpose("ensemble", "time", "N", "E"),
    "UNet": unet_temp.expand_dims(ensemble=[0]).transpose("ensemble", "time", "N", "E"),
    "DDIM": ddim_ens_temp.transpose("ensemble", "time", "N", "E")
}

metrics = {}

for name, pred in models.items():
    crps_grid = gridwise_temporal_crps(obs_temp, pred.values)
    metrics[name] = crps_grid

mean_crps = {name: np.nanmean(grid) for name, grid in metrics.items()}

metric_df = pd.DataFrame.from_dict(mean_crps, orient="index", columns=["CRPS"])
metric_df.to_csv("DDIM_conditional_derived/Metrics_Test_Set/cobweb/outputs/crps_allmodels_temp.csv")
#--------------------------------------------------------------------#