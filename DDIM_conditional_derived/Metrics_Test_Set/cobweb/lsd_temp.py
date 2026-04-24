import os
import pandas as pd
import xarray as xr
import numpy as np
from tqdm.auto import tqdm as tqdm_auto
from joblib import Parallel, delayed, parallel

def lsd_for_grid(i, j, obs_arr, pred_arr, n_fft=256, eps=1e-8):
    obs_valid = obs_arr[:, i, j]
    pred_valid = pred_arr[:, i, j]
    if np.sum(~np.isnan(obs_valid)) < n_fft or np.sum(~np.isnan(pred_valid)) < n_fft:
        return np.nan
    obs_fft = np.fft.rfft(obs_valid, n=n_fft)
    pred_fft = np.fft.rfft(pred_valid, n=n_fft)
    obs_log = np.log(np.abs(obs_fft) + eps)
    pred_log = np.log(np.abs(pred_fft) + eps)
    return np.sqrt(np.mean((obs_log - pred_log) ** 2))




def gridwise_temporal_lsd(obs, pred, n_fft=256, eps=1e-8, n_jobs=-1):
    obs_arr = obs.values  # (T, N, E)
    pred_arr = pred.values  # (T, N, E)
    T, N, E = obs_arr.shape
    tasks = [(i, j) for i in range(N) for j in range(E)]
    lsd_grid = np.full((N, E), np.nan)




    class TqdmBatchCompletionCallback(parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

    with tqdm_auto(total=len(tasks), desc="Grid cells processed", unit="cell") as tqdm_object:
        results = Parallel(n_jobs=n_jobs)(
            delayed(lsd_for_grid)(i, j, obs_arr, pred_arr, n_fft, eps)
            for i, j in tasks
        )

    for idx, val in enumerate(results):
        i = idx // E
        j = idx % E
        lsd_grid[i, j] = val
    return lsd_grid

#--------------------------------------------------------------------#

obs_temp = xr.open_dataset('Dataset_Setup_I_Chronological_12km/TabsD_step1_latlon.nc')["TabsD"].sel(time=slice("2011-01-01","2023-12-31"))
unet_temp = xr.open_dataset("DDIM_conditional_derived/output_inference/UNet_downscaled_test_set_2011_2023.nc")["temp"].sel(time=slice("2011-01-01","2023-12-31"))
coarse_temp = xr.open_dataset("Dataset_Setup_I_Chronological_12km/TabsD_step2_coarse.nc")["TabsD"].sel(time=slice("2011-01-01","2023-12-31"))
coarse_temp_interp = coarse_temp.interp_like(obs_temp, method="nearest")
bicubic_temp = xr.open_dataset("Dataset_Setup_I_Chronological_12km/TabsD_step3_interp.nc")["TabsD"].sel(time=slice("2011-01-01","2023-12-31"))
ddim_temp = xr.open_dataset("DDIM_conditional_derived/output_inference/ddim_downscaled_30steps_test_set_11samples_2011_2023.nc")["temp"].sel(time=slice("2011-01-01", "2023-12-31"))




if "sample" in ddim_temp.dims:
    ddim_ens_temp = ddim_temp.rename({"sample": "ensemble"})
else:
    raise ValueError("Expected 'sample' dimension in ddim_temp")

models = {
    "Coarse": coarse_temp_interp,
    "Bicubic": bicubic_temp,
    "UNet": unet_temp,
    "DDIM": ddim_ens_temp.mean(dim="ensemble")
}

metrics = {}

for name, pred in models.items():
    lsd_grid = gridwise_temporal_lsd(obs_temp, pred)
    metrics[name] = np.nanmean(lsd_grid)

metric_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["LSD"])
metric_df.to_csv("DDIM_conditional_derived/Metrics_Test_Set/cobweb/outputs/lsd_allmodels_temp.csv")
#--------------------------------------------------------------------#