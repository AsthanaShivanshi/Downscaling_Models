import xarray as xr
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg') # Non-interactive backend
from skimage.metrics import structural_similarity
from properscoring import crps_ensemble
from statsmodels.distributions.empirical_distribution import ECDF
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed



#helpers

#--------------------------------------------------------------------#



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

def pitd_cell(i, j, obs, ens_pred, bin_edges):
    obs_series = obs[:, i, j].values
    ens_series = ens_pred[:, :, i, j]  # [ensemble, time]
    mask = ~np.isnan(obs_series)
    if np.sum(mask) < 2:
        return (i, j, np.nan)
    obs_valid = obs_series[mask]
    ens_valid = ens_series[:, mask]  # [ensemble, valid_time]
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

#--------------------------------------------------------------------#



def gridwise_temporal_rmse(obs, pred):
    diff_sq = (obs - pred) ** 2
    rmse_grid = (diff_sq.mean(dim="time", skipna=True)) ** 0.5
    return float(rmse_grid.mean().values)




def gridwise_temporal_crps(obs, ens_pred, n_workers=None):
    obs_arr = obs.values
    ens_arr = ens_pred
    T, N, E = obs_arr.shape
    indices = [(i, j) for i in range(N) for j in range(E)]
    crps_grid = np.full((N, E), np.nan)
    count=0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(crps_cell, i, j, obs_arr, ens_arr): (i, j) for i, j in indices}
        for future in tqdm(as_completed(futures), total=len(futures), desc="CRPS grid cells",miniters=1000):
            i, j, value = future.result()
            crps_grid[i, j] = value


            count=count+1
            if count%1000==0:
                print(f"Processed {count} CRPS grid cells")
    return np.nanmean(crps_grid)

def framewise_ssim(obs, pred):
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

def gridwise_temporal_lsd(obs, pred, n_fft=256, eps=1e-8, n_workers=None):
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
    count=0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(lsd_idx, idx, obs_flat, pred_flat, n_fft, eps): idx
            for idx in indices
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="LSD grid cells", miniters=1000):
            idx, value = future.result()
            lsd[idx] = value

            count=count+1
            if count%1000==0:
                print(f"Processed {count} LSD grid cells")
    return np.nanmean(lsd)

def gridwise_pitd_rmse(obs, ens_pred, bins=20, n_workers=None):
    N, E = obs.shape[1], obs.shape[2]
    bin_edges = np.linspace(0, 1, bins + 1)
    indices = [(i, j) for i in range(N) for j in range(E)]
    pitd_grid = np.full((N, E), np.nan)
    count=0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(pitd_cell, i, j, obs, ens_pred, bin_edges): (i, j) for i, j in indices}
        for future in tqdm(as_completed(futures), total=len(futures), desc="PITD grid cells", miniters=1000):
            i, j, value = future.result()
            pitd_grid[i, j] = value


            count=count+1
            if count%1000==0:
                print(f"Processed {count} PITD RMSE for grid cell")
    return np.nanmean(pitd_grid)

#--------------------------------------------------------------------------------------------#

obs_precip = xr.open_dataset('Dataset_Setup_I_Chronological_12km/RhiresD_step1_latlon.nc')["RhiresD"].sel(time=slice("2011-01-01","2023-12-31"))
unet_precip = xr.open_dataset("DDIM_conditional_derived/output_inference/UNet_downscaled_test_set_2011_2023.nc")["precip"].sel(time=slice("2011-01-01","2023-12-31"))
coarse_precip = xr.open_dataset("Dataset_Setup_I_Chronological_12km/RhiresD_step2_coarse.nc")["RhiresD"].sel(time=slice("2011-01-01","2023-12-31"))

# Only interpolating for gridwise metric calculation. For the framewise SSIM,  original coarse grid 
coarse_precip_interp = coarse_precip.interp(
    N=obs_precip.N, E=obs_precip.E, method="nearest"
)
bicubic_precip = xr.open_dataset("Dataset_Setup_I_Chronological_12km/RhiresD_step3_interp.nc")["RhiresD"].sel(time=slice("2011-01-01","2023-12-31"))
ddim_precip = xr.open_dataset("DDIM_conditional_derived/output_inference/ddim_downscaled_100steps_test_set_5samples_eta_0.0.nc")["precip"].sel(time=slice("2011-01-01", "2023-12-31"))


#--------------------------------------------------------------------#
#Only for precip 
obs_precip = obs_precip.where(obs_precip >= 0)
unet_precip = unet_precip.where(unet_precip >= 0)
coarse_precip = coarse_precip.where(coarse_precip >= 0)
coarse_precip_interp = coarse_precip_interp.where(coarse_precip_interp >= 0)
bicubic_precip = bicubic_precip.where(bicubic_precip >= 0)
ddim_precip = ddim_precip.where(ddim_precip >= 0)

if "sample" in ddim_precip.dims:
    ddim_ens_precip = ddim_precip.rename({"sample": "ensemble"})
    ddim_ens_precip = ddim_ens_precip.where(ddim_ens_precip >= 0)

#---------------------------------------------------------------------------------------------#

models = {
    "Coarse": coarse_precip_interp, #not used for SSIM
    "Bicubic": bicubic_precip,
    "UNet": unet_precip,
    "DDIM": ddim_ens_precip.mean(dim="ensemble")
}
#--------------------------------------------------------------------#



metrics = {}
metric_names = ["CRPS", "RMSE", "SSIM", "PITD", "LSD"]



#--------------------------------------------------------------------#
for name, pred in tqdm(models.items(), desc="Models"):
    ssim = framewise_ssim(obs_precip, pred)
    rmse = gridwise_temporal_rmse(obs_precip, pred)
    lsd = gridwise_temporal_lsd(obs_precip, pred, n_workers=4)



    if name == "DDIM":
        crps = gridwise_temporal_crps(obs_precip, ddim_ens_precip.values, n_workers=4)
        pitd = gridwise_pitd_rmse(obs_precip, ddim_ens_precip.values, bins=20, n_workers=4)
    else:


        crps = float(np.nanmean(np.abs(obs_precip.values - pred.values)))
        pitd = gridwise_pitd_rmse(obs_precip, np.expand_dims(pred.values, axis=0), bins=20, n_workers=4)






        
    metrics[name] = [crps, rmse, ssim, pitd, lsd]

#---------------------------------------------------------------------------------------------#

for idx, metric in enumerate(metric_names):
    metric_dict = {k: v[idx] for k, v in metrics.items()}
    metric_df = pd.DataFrame.from_dict(metric_dict, orient="index", columns=[metric])
    metric_df.to_csv(f"DDIM_conditional_derived/Metrics_Test_Set/outputs/{metric.lower()}_allmodels_precip.csv")

#---------------------------------------------------------------------------------------------#