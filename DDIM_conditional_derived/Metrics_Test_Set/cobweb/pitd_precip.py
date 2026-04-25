import os
import pandas as pd
import xarray as xr
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF

from tqdm.auto import tqdm as tqdm_auto
from joblib import parallel

from joblib import Parallel, delayed


def ensure_NE(da, ref):
    # Rename if needed
    if "y" in da.dims and "x" in da.dims:
        da = da.rename({"y": "N", "x": "E"})
    # Assign coords if needed
    if "N" in da.dims and "E" in da.dims:
        da = da.assign_coords(N=ref.N, E=ref.E)
    return da


def pitd_for_grid(i, j, obs_arr, ens_arr, bins):
    obs_series = obs_arr[:, i, j]
    ens_series = ens_arr[:, :, i, j]
    mask = ~np.isnan(obs_series)
    if np.sum(mask) < 2:
        return np.nan
    obs_valid = obs_series[mask]
    ens_valid = ens_series[:, mask]
    ecdf = ECDF(obs_valid)
    # Vectorized PIT calculation
    pit = ecdf(ens_valid.ravel())
    if pit.size == 0:
        return np.nan
    bin_edges = np.linspace(0, 1, bins + 1)
    pit_hist, _ = np.histogram(pit, bins=bin_edges, density=True)
    pit_hist = pit_hist / pit_hist.sum()
    uniform = np.ones_like(pit_hist) / len(pit_hist)
    return np.sqrt(np.mean((pit_hist - uniform) ** 2))


def gridwise_temporal_pitd(obs, ens_pred, bins=20, n_jobs=-1, mask=None):
    obs_arr = obs.values  # (T, N, E)
    ens_arr = ens_pred    # (ensemble, T, N, E)
    T, N, E = obs_arr.shape
    tasks = []
    for i in range(N):
        for j in range(E):
            if mask is not None and not mask.values[i, j]:
                continue  # skip masked-out grid cells
            tasks.append((i, j))
    pitd_grid = np.full((N, E), np.nan)

    class TqdmBatchCompletionCallback(parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

    with tqdm_auto(total=len(tasks), desc="Grid cells processed", unit="cell") as tqdm_object:
        results = Parallel(n_jobs=n_jobs)(
            delayed(pitd_for_grid)(i, j, obs_arr, ens_arr, bins)
            for i, j in tasks
        )

    for idx, (i, j) in enumerate(tasks):
        pitd_grid[i, j] = results[idx]
    return pitd_grid

#--------------------------------------------------------------------#

obs_precip = xr.open_dataset(
    'Dataset_Setup_I_Chronological_12km/RhiresD_step1_latlon.nc'
)["RhiresD"].sel(time=slice("2011-01-01", "2023-12-31"))

unet_precip = xr.open_dataset(
    "DDIM_conditional_derived/output_inference/UNet_downscaled_test_set_2011_2023.nc"
)["precip"].sel(time=slice("2011-01-01", "2023-12-31"))

coarse_precip = xr.open_dataset(
    "Dataset_Setup_I_Chronological_12km/RhiresD_step2_coarse.nc"
)["RhiresD"].sel(time=slice("2011-01-01", "2023-12-31"))

bicubic_precip = xr.open_dataset(
    "Dataset_Setup_I_Chronological_12km/RhiresD_step3_interp.nc"
)["RhiresD"].sel(time=slice("2011-01-01", "2023-12-31"))

ddim_precip = xr.open_dataset(
    "DDIM_conditional_derived/output_inference/ddim_downscaled_30steps_test_set_11samples_2011_2023.nc"
)["precip"].sel(time=slice("2011-01-01", "2023-12-31"))




coarse_precip_interp = coarse_precip.interp_like(obs_precip, method="nearest")




obs_precip = ensure_NE(obs_precip, obs_precip).clip(min=0)
unet_precip = ensure_NE(unet_precip, obs_precip).clip(min=0)
coarse_precip_interp = ensure_NE(coarse_precip_interp, obs_precip).clip(min=0)
bicubic_precip = ensure_NE(bicubic_precip, obs_precip).clip(min=0)



if "sample" in ddim_precip.dims:
    ddim_ens_precip = ddim_precip.rename({"sample": "ensemble"})
else:
    raise ValueError("Expected 'sample' dimension in ddim_precip")
ddim_ens_precip = ensure_NE(ddim_ens_precip, obs_precip).clip(min=0)

models = {
    "Coarse": coarse_precip_interp.expand_dims(ensemble=[0]).transpose("ensemble", "time", "N", "E"),
    "Bicubic": bicubic_precip.expand_dims(ensemble=[0]).transpose("ensemble", "time", "N", "E"),
    "UNet": unet_precip.expand_dims(ensemble=[0]).transpose("ensemble", "time", "N", "E"),
    "DDIM": ddim_ens_precip.transpose("ensemble", "time", "N", "E")
}

#--------------------------------------------------------------------#


obs_temp = xr.open_dataset("Dataset_Setup_I_Chronological_12km/TabsD_step1_latlon.nc")["TabsD"].sel(time=slice("2011-01-01", "2011-01-02"))
obs_mask_precip = ~np.isnan(obs_temp.isel(time=0))

#--------------------------------------------------------------------#
metrics = {}

for name, pred in models.items():
    pitd_grid = gridwise_temporal_pitd(obs_precip, pred.values, mask=obs_mask_precip)
    metrics[name] = pitd_grid

mean_pitd = {name: np.nanmean(grid) for name, grid in metrics.items()}

metric_df = pd.DataFrame.from_dict(mean_pitd, orient="index", columns=["PITD"])
metric_df.to_csv("DDIM_conditional_derived/Metrics_Test_Set/cobweb/outputs/pitd_allmodels_precip.csv")



#--------------------------------------------------------------------#