import pandas as pd
import xarray as xr
import numpy as np
from skimage.metrics import structural_similarity

def framewise_ssim(obs, pred, mask2d=None):
    ssim_frames = []
    for t in range(obs.shape[0]):
        obs_frame = obs.isel(time=t).values
        pred_frame = pred.isel(time=t).values
        if mask2d is not None:
            valid_mask = mask2d.values
        else:
            valid_mask = ~np.isnan(obs_frame) & ~np.isnan(pred_frame)
        if not np.any(valid_mask):
            ssim_frames.append(np.nan)
            continue
        obs_filled = np.where(valid_mask, obs_frame, np.nanmean(obs_frame[valid_mask]))
        pred_filled = np.where(valid_mask, pred_frame, np.nanmean(pred_frame[valid_mask]))
        data_range = obs_filled[valid_mask].max() - obs_filled[valid_mask].min()
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


obs_precip = xr.open_dataset('Dataset_Setup_I_Chronological_12km/RhiresD_step1_latlon.nc')["RhiresD"].sel(time=slice("2011-01-01","2023-12-31"))
unet_precip = xr.open_dataset("DDIM_conditional_derived/output_inference/UNet_downscaled_test_set_2011_2023.nc")["precip"].sel(time=slice("2011-01-01","2023-12-31"))
coarse_precip = xr.open_dataset("Dataset_Setup_I_Chronological_12km/RhiresD_step2_coarse.nc")["RhiresD"].sel(time=slice("2011-01-01","2023-12-31"))

coarse_precip_interp = coarse_precip.interp(
    N=obs_precip.N, E=obs_precip.E, method="nearest"
)

bicubic_precip = xr.open_dataset("Dataset_Setup_I_Chronological_12km/RhiresD_step3_interp.nc")["RhiresD"].sel(time=slice("2011-01-01","2023-12-31"))
ddim_precip = xr.open_dataset("DDIM_conditional_derived/output_inference/ddim_downscaled_30steps_test_set_11samples_2011_2023.nc")["precip"].sel(time=slice("2011-01-01", "2023-12-31"))
#--------------------------------------------------------------------#



unet_precip = unet_precip.rename({"y": "N", "x": "E"})
ddim_precip = ddim_precip.rename({"y": "N", "x": "E"})





#--------------------------------------------------------------------#
#mask = ~np.isnan(obs_precip.isel(time=10))
#mask3d = xr.DataArray(mask, dims=("N", "E")).expand_dims(time=obs_precip.time)


#--------------------------------------------------------------------#


obs_precip = obs_precip.clip(min=0)

unet_precip = unet_precip.clip(min=0)

coarse_precip = coarse_precip.clip(min=0)
coarse_precip_interp = coarse_precip_interp.clip(min=0)

bicubic_precip = bicubic_precip.clip(min=0)
ddim_precip = ddim_precip.clip(min=0)

#--------------------------------------------------------------------#


ddim_precip = ddim_precip.assign_coords(N=obs_precip.N, E=obs_precip.E)


if "sample" in ddim_precip.dims:
    ddim_ens_precip = ddim_precip.rename({"sample": "ensemble"})

def ensemble_ssim(obs, ens_pred, mask2d=None):
    # ens_pred: (ensemble, time, N, E)
    ssim_list = []
    for i in range(ens_pred.shape[0]):
        ssim = framewise_ssim(obs, ens_pred.isel(ensemble=i), mask2d)
        ssim_list.append(ssim)
    return np.nanmean(ssim_list)


def best_ensemble_ssim(obs, ens_pred, mask2d=None):
    ssim_list = []
    for i in range(ens_pred.shape[0]):
        ssim = framewise_ssim(obs, ens_pred.isel(ensemble=i), mask2d)
        ssim_list.append(ssim)
    ssim_array = np.array(ssim_list)
    if np.all(np.isnan(ssim_array)):
        return np.nan, None  
    best_idx = np.nanargmax(ssim_array)
    return ssim_array[best_idx], best_idx


models = {
    "Coarse": coarse_precip_interp,
    "Bicubic": bicubic_precip,
    "UNet": unet_precip,
    "DDIM(11 samples)": ddim_ens_precip
}



metrics = {}


for name, pred in models.items():
    if name == "DDIM(11 samples)":
        pred_ens_first = pred.transpose("ensemble", "time", "N", "E")
        ssim = ensemble_ssim(obs_precip, pred_ens_first, None)
        best_ssim, best_idx = best_ensemble_ssim(obs_precip, pred_ens_first, None)
        metrics[name] = {"SSIM": ssim, "Best SSIM": best_ssim, "Best Ensemble Index": best_idx}
    else:
        ssim = framewise_ssim(obs_precip, pred, None)
        metrics[name] = {"SSIM": ssim}



#--------------------------------------------------------------------#
metric_df = pd.DataFrame.from_dict(metrics, orient="index")
metric_df.to_csv("DDIM_conditional_derived/Metrics_Test_Set/cobweb/outputs/ssim_allmodels_precip.csv")

#--------------------------------------------------------------------#