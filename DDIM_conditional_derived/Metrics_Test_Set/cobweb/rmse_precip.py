import xarray as xr
import numpy as np
import pandas as pd

# Load datasets
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
    "DDIM_conditional_derived/output_inference/ddim_downscaled_100steps_test_set_5samples_eta_0.0.nc"
)["precip"].sel(time=slice("2011-01-01", "2023-12-31"))




coarse_precip_interp = coarse_precip.interp_like(obs_precip, method="nearest")



obs_precip=obs_precip.clip(min=0)

unet_precip = unet_precip.clip(min=0)

bicubic_precip = bicubic_precip.clip(min=0)

ddim_precip = ddim_precip.clip(min=0)

coarse_precip_interp = coarse_precip_interp.clip(min=0)



if "sample" in ddim_precip.dims:
    ddim_ens_precip = ddim_precip.rename({"sample": "ensemble"})
else:
    ddim_ens_precip = ddim_precip
ddim_ens_precip_mean = ddim_ens_precip.mean(dim="ensemble") if "ensemble" in ddim_ens_precip.dims else ddim_ens_precip




print("obs_precip shape:", obs_precip.shape)
print("unet_precip shape:", unet_precip.shape)
print("coarse_precip_interp shape:", coarse_precip_interp.shape)
print("bicubic_precip shape:", bicubic_precip.shape)
print("ddim_ens_precip_mean shape:", ddim_ens_precip_mean.shape)




unet_precip = unet_precip.rename({"y": "N", "x": "E"})
ddim_ens_precip_mean = ddim_ens_precip_mean.rename({"y": "N", "x": "E"})




unet_precip = unet_precip.assign_coords(N=obs_precip.N, E=obs_precip.E)
ddim_ens_precip_mean = ddim_ens_precip_mean.assign_coords(N=obs_precip.N, E=obs_precip.E)

# Create mask for valid data
mask = ~np.isnan(obs_precip.isel(time=0))
mask3d = xr.DataArray(mask, dims=("N", "E")).expand_dims(time=obs_precip.time)



def rmse(a, b, mask3d):
    diff = a - b
    diff = diff.where(mask3d)
    rmse_grid = np.sqrt((diff ** 2).mean(dim="time"))
    spatial_mean_rmse = rmse_grid.mean(dim=["N", "E"]).item()
    return spatial_mean_rmse






metrics = {
    "Coarse": rmse(obs_precip, coarse_precip_interp, mask3d),
    "Bicubic": rmse(obs_precip, bicubic_precip, mask3d),
    "UNet": rmse(obs_precip, unet_precip, mask3d),
    "DDIM": rmse(obs_precip, ddim_ens_precip_mean, mask3d),
}

for name, value in metrics.items():
    print(f"{name}: Mean RMSE = {value}")



metric_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["RMSE"])
metric_df.to_csv("DDIM_conditional_derived/Metrics_Test_Set/cobweb/outputs/rmse_allmodels_precip.csv")