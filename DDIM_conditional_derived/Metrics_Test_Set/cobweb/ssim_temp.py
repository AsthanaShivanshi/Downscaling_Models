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
ddim_temp = xr.open_dataset("DDIM_conditional_derived/output_inference/ddim_downscaled_100steps_test_set_5samples_eta_0.0.nc")["temp"].sel(time=slice("2011-01-01", "2023-12-31"))
#--------------------------------------------------------------------#


if "sample" in ddim_temp.dims:
    ddim_ens_temp = ddim_temp.rename({"sample": "ensemble"})



models = {
    "Coarse": coarse_temp_interp,
    "Bicubic": bicubic_temp,
    "UNet": unet_temp,
    "DDIM": ddim_ens_temp.mean(dim="ensemble")
}

metrics = {}

for name, pred in models.items():
    ssim = framewise_ssim(obs_temp, pred)
    metrics[name] = ssim


#--------------------------------------------------------------------#
metric_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["SSIM"])
metric_df.to_csv("DDIM_conditional_derived/Metrics_Test_Set/cobweb/outputs/ssim_allmodels_temp.csv")

#--------------------------------------------------------------------#