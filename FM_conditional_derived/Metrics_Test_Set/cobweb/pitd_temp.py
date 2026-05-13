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


def gridwise_temporal_pitd(obs, ens_pred, bins=20, n_jobs=-1):
    obs_arr = obs.values  # (T, N, E)
    ens_arr = ens_pred    # (ensemble, T, N, E)
    T, N, E = obs_arr.shape
    tasks = [(i, j) for i in range(N) for j in range(E)]
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

    for idx, val in enumerate(results):
        i = idx // E
        j = idx % E
        pitd_grid[i, j] = val
    return pitd_grid

#--------------------------------------------------------------------#

obs_temp = xr.open_dataset(
    'Dataset_Setup_I_Chronological_12km/TabsD_step1_latlon.nc'
)["TabsD"].sel(time=slice("2011-01-01", "2023-12-31"))

unet_temp = xr.open_dataset(
    "DDIM_conditional_derived/output_inference/UNet_downscaled_test_set_2011_2023.nc"
)["temp"].sel(time=slice("2011-01-01", "2023-12-31"))

coarse_temp = xr.open_dataset(
    "Dataset_Setup_I_Chronological_12km/TabsD_step2_coarse.nc"
)["TabsD"].sel(time=slice("2011-01-01", "2023-12-31"))

bicubic_temp = xr.open_dataset(
    "Dataset_Setup_I_Chronological_12km/TabsD_step3_interp.nc"
)["TabsD"].sel(time=slice("2011-01-01", "2023-12-31"))

ddim_temp = xr.open_dataset(
    "DDIM_conditional_derived/output_inference/ddim_downscaled_30steps_test_set_11samples_2011_2023.nc"
)["temp"].sel(time=slice("2011-01-01", "2023-12-31"))




coarse_temp_interp = coarse_temp.interp_like(obs_temp, method="nearest")






if "sample" in ddim_temp.dims:
    ddim_ens_temp = ddim_temp.rename({"sample": "ensemble"})
else:
    raise ValueError("Expected 'sample' dimension in ddim_temp")
ddim_ens_temp = ensure_NE(ddim_ens_temp, obs_temp).clip(min=0)

models = {
    "Coarse": ensure_NE(coarse_temp_interp, obs_temp).expand_dims(ensemble=[0]).transpose("ensemble", "time", "N", "E"),
    "Bicubic": ensure_NE(bicubic_temp, obs_temp).expand_dims(ensemble=[0]).transpose("ensemble", "time", "N", "E"),
    "UNet": ensure_NE(unet_temp, obs_temp).expand_dims(ensemble=[0]).transpose("ensemble", "time", "N", "E"),
    "DDIM": ensure_NE(ddim_ens_temp, obs_temp).transpose("ensemble", "time", "N", "E")
}

#--------------------------------------------------------------------#

metrics = {}

for name, pred in models.items():
    pitd_grid = gridwise_temporal_pitd(obs_temp, pred.values)
    metrics[name] = pitd_grid

mean_pitd = {name: np.nanmean(grid) for name, grid in metrics.items()}

metric_df = pd.DataFrame.from_dict(mean_pitd, orient="index", columns=["PITD"])
metric_df.to_csv("DDIM_conditional_derived/Metrics_Test_Set/cobweb/outputs/pitd_allmodels_temp.csv")



#--------------------------------------------------------------------#