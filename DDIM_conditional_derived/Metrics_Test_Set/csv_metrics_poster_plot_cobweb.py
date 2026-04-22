import xarray as xr
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg') #Non int backend
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from properscoring import crps_ensemble
from statsmodels.distributions.empirical_distribution import ECDF


#for the cobweb plot . 


#SR metrics for poster K. Plot.. 



""" gridwise/framewise  : 

PSNR::: 20 * log10(data_range / sqrt(MSE))
RMSE :: sqrt(mean((i-j)^2))
SSIM:: lib from skimage, averaged over frames
CRPS :: using properscoring crps_ensemble, averaged over time and grid cells
PITD:: (RMSE of PIT from observational distribution uniformity)
LSD:: log spectral distance, computed on gridwise temporal FFTs, spatially averaged"""




def compute_gridwise_temporal_metrics(obs, pred):
    # obs, pred: [time, N, E]
    N, E = obs.shape[1], obs.shape[2]
    psnr_grid = np.full((N, E), np.nan)
    rmse_grid = np.full((N, E), np.nan)

    for i in range(N):
        for j in range(E):
            obs_series = obs[:, i, j].values
            pred_series = pred[:, i, j].values
            mask = ~np.isnan(obs_series) & ~np.isnan(pred_series)

            obs_valid = obs_series[mask]
            pred_valid = pred_series[mask]

            # Skip if no valid data
            if obs_valid.size == 0 or pred_valid.size == 0:
                continue

            data_range = obs_valid.max() - obs_valid.min()
            if data_range == 0:
                psnr_grid[i, j] = np.nan
            else:
                psnr_grid[i, j] = peak_signal_noise_ratio(obs_valid, pred_valid, data_range=data_range)
            rmse_grid[i, j] = np.sqrt(mean_squared_error(obs_valid, pred_valid))

    psnr = np.nanmean(psnr_grid)
    rmse = np.nanmean(rmse_grid)
    return psnr, rmse



def compute_gridwise_temporal_crps(obs, ens_pred):
    # obs: [time, N, E], ens_pred: [ensemble, time, N, E]


    N, E = obs.shape[1], obs.shape[2]


    crps_grid = np.full((N, E), np.nan)
    for i in range(N):
        for j in range(E):
            obs_series = obs[:, i, j].values
            ens_series = ens_pred[:, :, i, j]  # [ensemble, time]
            mask = ~np.isnan(obs_series)



            obs_valid = obs_series[mask]
            ens_valid = ens_series[:, mask]

            crps_vals = []

            for t in range(obs_valid.shape[0]):
                if np.any(np.isnan(ens_valid[:, t])):
                    continue


                crps_vals.append(crps_ensemble([obs_valid[t]], ens_valid[:, t][None, :])[0])
            if crps_vals:
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
    # obs, pred: [time, N, E]
    N, E = obs.shape[1], obs.shape[2]
    lsd_grid = np.full((N, E), np.nan)
    for i in range(N):


        for j in range(E):
            obs_series = obs[:, i, j].values
            pred_series = pred[:, i, j].values
            mask = ~np.isnan(obs_series) & ~np.isnan(pred_series)

            obs_valid = obs_series[mask]
            pred_valid = pred_series[mask]


            if len(obs_valid) < n_fft:
                continue


            # Compute FFT
            obs_fft = np.fft.rfft(obs_valid, n=n_fft)
            pred_fft = np.fft.rfft(pred_valid, n=n_fft)
            obs_log = np.log(np.abs(obs_fft) + eps)
            pred_log = np.log(np.abs(pred_fft) + eps)
            lsd = np.sqrt(np.mean((obs_log - pred_log) ** 2))
            lsd_grid[i, j] = lsd
    return np.nanmean(lsd_grid)



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



obs_temp = xr.open_dataset('Dataset_Setup_I_Chronological_12km/TabsD_step1_latlon.nc')["TabsD"].sel(time=slice("2011-01-01","2023-12-31"))


unet_temp= xr.open_dataset("DDIM_conditional_derived/output_inference/UNet_downscaled_test_set_2011_2023.nc")["temp"].sel(time=slice("2011-01-01","2023-12-31"))

coarse_temp= xr.open_dataset("Dataset_Setup_I_Chronological_12km/TabsD_step2_coarse.nc")["TabsD"].sel(time=slice("2011-01-01","2023-12-31"))



#only interpolating for gridwise metric calculation. For the framewise SSIM, we can use the original coarse grid to avoid interpolation artifacts.



coarse_temp_interp = coarse_temp.interp(
    N=obs_temp.N, E=obs_temp.E, method="nearest"
)

bicubic_temp= xr.open_dataset("Dataset_Setup_I_Chronological_12km/TabsD_step3_interp.nc")["TabsD"].sel(time=slice("2011-01-01","2023-12-31"))


ddim_ds = xr.open_dataset("DDIM_conditional_derived/output_inference/ddim_downscaled_100steps_test_set_5samples_eta_0.0.nc")
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

for name, pred in models.items():

    ssim = compute_framewise_ssim(obs_temp, pred)
    psnr, rmse = compute_gridwise_temporal_metrics(obs_temp, pred)
    lsd = compute_gridwise_temporal_lsd(obs_temp, pred)
    if name == "DDIM":
        crps = compute_gridwise_temporal_crps(obs_temp, ddim_ens_temp.values)
        pitd_grid = compute_gridcell_pitd_rmse(obs_temp, ddim_ens_temp.values, bins=20)
        pitd = np.nanmean(pitd_grid)
    else:
        crps = float(np.nanmean(np.abs(obs_temp.values - pred.values)))
        pitd_grid = compute_gridcell_pitd_rmse(obs_temp, np.expand_dims(pred.values, axis=0), bins=20)
        pitd = np.nanmean(pitd_grid)

    metrics[name] = [psnr, ssim, rmse, crps, pitd, lsd]

#---------------------------------------------------------------------------------------------#

metrics_df = pd.DataFrame.from_dict(
    metrics, orient="index",
    columns=["PSNR", "SSIM", "RMSE", "CRPS", "PITD", "LSD"]
)
metrics_df.to_csv("DDIM_conditional_derived/Metrics_Test_Set/outputs/cobweb_temp_metrics_allmodels.csv")



#---------------------------------------------------------------------------------------------#
