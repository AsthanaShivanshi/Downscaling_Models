import xarray as xr
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg') #Non int backend
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from properscoring import crps_ensemble
from statsmodels.distributions.empirical_distribution import ECDF
from tqdm import tqdm
import dask.array as da

#for the cobweb plot . 


#SR metrics for poster K. Plot.. 



""" gridwise/framewise  : 

RMSE :: sqrt(mean((i-j)^2))
SSIM:: lib from skimage, averaged over frames
CRPS :: using properscoring crps_ensemble, averaged over time and grid cells
PITD:: (RMSE of PIT from observational distribution uniformity)
LSD:: log spectral distance, computed on gridwise temporal FFTs, spatially averaged"""



def compute_gridwise_temporal_metrics(obs, pred):
    diff_sq= (obs-pred)**2
    rmse_grid= (diff_sq.mean(dim="time", skipna=True))**0.5
    return float(rmse_grid.mean().values)




def compute_gridwise_temporal_crps(obs, ens_pred):
    # obs: [time, N, E], ens_pred: [ensemble, time, N, E]
    obs_arr = obs.values
    ens_arr = ens_pred  # already np array
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
            crps_grid[i, j] = np.mean(crps_vals)
    return np.nanmean(crps_grid)






def compute_framewise_ssim(obs, pred):
    ssim_frames = []
    for t in range(obs.shape[0]):
        obs_frame = obs.isel(time=t).values
        pred_frame = pred.isel(time=t).values
        mask = ~np.isnan(obs_frame) & ~np.isnan(pred_frame)
        if not np.any(mask):
            ssim_frames.append(np.nan)
            continue


        obs_filled = np.where(mask, obs_frame, np.nanmean(obs_frame[mask]))

        pred_filled = np.where(mask, pred_frame, np.nanmean(pred_frame[mask]))

        data_range = obs_filled[mask].max() - obs_filled[mask].min()


        if data_range == 0:
            ssim_frames.append(np.nan)
            continue
        try:


            ssim = structural_similarity(obs_filled, pred_filled, data_range=data_range)
        except Exception:
            ssim = np.nan
        ssim_frames.append(ssim)
    return np.nanmean(ssim_frames)


def compute_gridwise_temporal_lsd(obs, pred, n_fft=256, eps=1e-8):
    obs_arr = obs.values
    pred_arr = pred.values
    mask = ~np.isnan(obs_arr) & ~np.isnan(pred_arr)
    obs_arr = np.where(mask, obs_arr, np.nan)
    pred_arr = np.where(mask, pred_arr, np.nan)
    # Reshape to [time, -1] for vectorised FFT
    T, N, E = obs_arr.shape
    obs_flat = obs_arr.reshape(T, -1)
    pred_flat = pred_arr.reshape(T, -1)
    valid_mask = ~np.isnan(obs_flat).any(axis=0) & ~np.isnan(pred_flat).any(axis=0)
    lsd = np.full(obs_flat.shape[1], np.nan)
    for idx in np.where(valid_mask)[0]:
        obs_valid = obs_flat[:, idx]
        pred_valid = pred_flat[:, idx]
        if np.sum(~np.isnan(obs_valid)) < n_fft or np.sum(~np.isnan(pred_valid)) < n_fft:
            continue
        obs_fft = np.fft.rfft(obs_valid, n=n_fft)
        pred_fft = np.fft.rfft(pred_valid, n=n_fft)
        obs_log = np.log(np.abs(obs_fft) + eps)
        pred_log = np.log(np.abs(pred_fft) + eps)
        lsd[idx] = np.sqrt(np.mean((obs_log - pred_log) ** 2))
    return np.nanmean(lsd)


def compute_gridcell_pitd_rmse(obs, ens_pred, bins=20):



    N, E = obs.shape[1], obs.shape[2]
    pitd_grid = np.full((N, E), np.nan)
    bin_edges = np.linspace(0, 1, bins + 1)
    for i in range(N):
        for j in range(E):
            obs_series = obs[:, i, j].values
            ens_series = ens_pred[:, :, i, j]  # [ensemble, time]
            mask = ~np.isnan(obs_series)


            if np.sum(mask) < 2:
                continue


            obs_valid = obs_series[mask]
            ens_valid = ens_series[:, mask]  # [ensemble, valid_time]
            # Pool all ensemble predictions for PIT
            pit = []



            for t in range(ens_valid.shape[1]):
                ecdf = ECDF(obs_valid)
                for k in range(ens_valid.shape[0]):
                    pit.append(ecdf(ens_valid[k, t]))
            pit = np.array(pit)
            if pit.size == 0:
                continue

            pit_hist, _ = np.histogram(pit, bins=bin_edges, density=True)
            pit_hist = pit_hist / pit_hist.sum()
            uniform = np.ones_like(pit_hist) / len(pit_hist)
            pitd_grid[i, j] = np.sqrt(np.mean((pit_hist - uniform) ** 2))
    return pitd_grid



#--------------------------------------------------------------------------------------------#



obs_temp = xr.open_dataset('Dataset_Setup_I_Chronological_12km/TabsD_step1_latlon.nc', chunks={"time": 100})["TabsD"].sel(time=slice("2011-01-01","2023-12-31"))


unet_temp= xr.open_dataset("DDIM_conditional_derived/output_inference/UNet_downscaled_test_set_2011_2023.nc", chunks={"time": 100})["temp"].sel(time=slice("2011-01-01","2023-12-31"))

coarse_temp= xr.open_dataset("Dataset_Setup_I_Chronological_12km/TabsD_step2_coarse.nc", chunks={"time": 100})["TabsD"].sel(time=slice("2011-01-01","2023-12-31"))



#only interpolating for gridwise metric calculation. For the framewise SSIM, we can use the original coarse grid to avoid interpolation artifacts.



coarse_temp_interp = coarse_temp.interp(
    N=obs_temp.N, E=obs_temp.E, method="nearest"
)

bicubic_temp= xr.open_dataset("Dataset_Setup_I_Chronological_12km/TabsD_step3_interp.nc", chunks={"time": 100})["TabsD"].sel(time=slice("2011-01-01","2023-12-31"))


ddim_ds = xr.open_dataset("DDIM_conditional_derived/output_inference/ddim_downscaled_100steps_test_set_5samples_eta_0.0.nc", chunks={"time": 100})
ddim_ens_temp = ddim_ds["temp"].sel(time=slice("2011-01-01", "2023-12-31"))

if "sample" in ddim_ens_temp.dims:
    ddim_ens_temp = ddim_ens_temp.rename({"sample": "ensemble"})

#---------------------------------------------------------------------------------------------#


models = {
    "Coarse": coarse_temp_interp,  # interpolated for gridwise metrics,, not used for SSIM
    "Bicubic": bicubic_temp,
    "UNet": unet_temp,
    "DDIM": ddim_ens_temp.mean(dim="ensemble")
}

metrics = {}

metric_names = ["SSIM", "RMSE", "CRPS", "PITD", "LSD"]


for name, pred in tqdm(models.items(), desc="Models"):
    ssim = compute_framewise_ssim(obs_temp, pred)
    rmse = compute_gridwise_temporal_metrics(obs_temp, pred)
    lsd = compute_gridwise_temporal_lsd(obs_temp, pred)
    if name == "DDIM":
        crps = compute_gridwise_temporal_crps(obs_temp, ddim_ens_temp.values)
        pitd_grid = compute_gridcell_pitd_rmse(obs_temp, ddim_ens_temp.values, bins=20)
        pitd = np.nanmean(pitd_grid)
    else:
        crps = float(np.nanmean(np.abs(obs_temp.values - pred.values)))
        pitd_grid = compute_gridcell_pitd_rmse(obs_temp, np.expand_dims(pred.values, axis=0), bins=20)
        pitd = np.nanmean(pitd_grid)

    metrics[name] = [ssim, rmse, crps, pitd, lsd]


#---------------------------------------------------------------------------------------------#

for idx, metric in enumerate(metric_names):
    metric_dict = {k: v[idx] for k, v in metrics.items()}
    metric_df = pd.DataFrame.from_dict(metric_dict, orient="index", columns=[metric])
    metric_df.to_csv(f"DDIM_conditional_derived/Metrics_Test_Set/outputs/{metric.lower()}_allmodels.csv")



#---------------------------------------------------------------------------------------------#
