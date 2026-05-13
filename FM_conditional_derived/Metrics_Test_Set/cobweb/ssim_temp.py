import pandas as pd
import xarray as xr
import numpy as np
from skimage.metrics import structural_similarity

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




#--------------------------------------------------------------------#


obs_temp = xr.open_dataset('Dataset_Setup_I_Chronological_12km/TabsD_step1_latlon.nc')["TabsD"].sel(time=slice("2011-01-01","2023-12-31"))
unet_temp = xr.open_dataset("DDIM_conditional_derived/output_inference/UNet_downscaled_test_set_2011_2023.nc")["temp"].sel(time=slice("2011-01-01","2023-12-31"))
coarse_temp = xr.open_dataset("Dataset_Setup_I_Chronological_12km/TabsD_step2_coarse.nc")["TabsD"].sel(time=slice("2011-01-01","2023-12-31"))

coarse_temp_interp = coarse_temp.interp(
    N=obs_temp.N, E=obs_temp.E, method="nearest"
)

bicubic_temp = xr.open_dataset("Dataset_Setup_I_Chronological_12km/TabsD_step3_interp.nc")["TabsD"].sel(time=slice("2011-01-01","2023-12-31"))
ddim_temp = xr.open_dataset("DDIM_conditional_derived/output_inference/ddim_downscaled_30steps_test_set_11samples_2011_2023.nc")["temp"].sel(time=slice("2011-01-01", "2023-12-31"))
#--------------------------------------------------------------------#



unet_temp = unet_temp.rename({"y": "N", "x": "E"})
ddim_temp = ddim_temp.rename({"y": "N", "x": "E"})





mask = ~np.isnan(obs_temp).any(dim="time")

mask3d = xr.DataArray(mask, dims=("N", "E")).expand_dims(time=obs_temp.time)


unet_temp = unet_temp.assign_coords(N=obs_temp.N, E=obs_temp.E)


ddim_temp = ddim_temp.assign_coords(N=obs_temp.N, E=obs_temp.E)


if "sample" in ddim_temp.dims:
    ddim_ens_temp = ddim_temp.rename({"sample": "ensemble"})

def ensemble_ssim(obs, ens_pred):
    # ens_pred: (ensemble, time, N, E)
    ssim_list = []
    for i in range(ens_pred.shape[0]):
        ssim = framewise_ssim(obs, ens_pred.isel(ensemble=i))
        ssim_list.append(ssim)
    return np.nanmean(ssim_list)


def best_ensemble_ssim(obs, ens_pred):
    ssim_list = []
    for i in range(ens_pred.shape[0]):
        ssim = framewise_ssim(obs, ens_pred.isel(ensemble=i))
        ssim_list.append(ssim)
    ssim_array = np.array(ssim_list)
    best_idx = np.nanargmax(ssim_array)
    return ssim_array[best_idx], best_idx


models = {
    "Coarse": coarse_temp_interp,
    "Bicubic": bicubic_temp,
    "UNet": unet_temp,
    "DDIM": ddim_ens_temp
}

metrics = {}

for name, pred in models.items():
    if name == "DDIM":
        pred_ens_first = pred.transpose("ensemble", "time", "N", "E")
        ssim = ensemble_ssim(obs_temp, pred_ens_first)
        best_ssim, best_idx = best_ensemble_ssim(obs_temp, pred_ens_first)
        metrics[name] = {"SSIM": ssim, "Best SSIM": best_ssim, "Best Ensemble Index": best_idx}
    else:
        ssim = framewise_ssim(obs_temp, pred)
        metrics[name] = {"SSIM": ssim}


#--------------------------------------------------------------------#
metric_df = pd.DataFrame.from_dict(metrics, orient="index")
metric_df.to_csv("DDIM_conditional_derived/Metrics_Test_Set/cobweb/outputs/ssim_allmodels_temp.csv")

#--------------------------------------------------------------------#