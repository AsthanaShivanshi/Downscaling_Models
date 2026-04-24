import xarray as xr
import numpy as np
import pandas as pd

# Load datasets
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
    "DDIM_conditional_derived/output_inference/ddim_downscaled_100steps_test_set_5samples_eta_0.0.nc"
)["temp"].sel(time=slice("2011-01-01", "2023-12-31"))




coarse_temp_interp = coarse_temp.interp_like(obs_temp, method="nearest")




if "sample" in ddim_temp.dims:
    ddim_ens_temp = ddim_temp.rename({"sample": "ensemble"})
else:
    ddim_ens_temp = ddim_temp
ddim_ens_temp_mean = ddim_ens_temp.mean(dim="ensemble") if "ensemble" in ddim_ens_temp.dims else ddim_ens_temp




print("obs_temp shape:", obs_temp.shape)
print("unet_temp shape:", unet_temp.shape)
print("coarse_temp_interp shape:", coarse_temp_interp.shape)
print("bicubic_temp shape:", bicubic_temp.shape)
print("ddim_ens_temp_mean shape:", ddim_ens_temp_mean.shape)




unet_temp = unet_temp.rename({"y": "N", "x": "E"})
ddim_ens_temp_mean = ddim_ens_temp_mean.rename({"y": "N", "x": "E"})




unet_temp = unet_temp.assign_coords(N=obs_temp.N, E=obs_temp.E)
ddim_ens_temp_mean = ddim_ens_temp_mean.assign_coords(N=obs_temp.N, E=obs_temp.E)

# Create mask for valid data
mask = ~np.isnan(obs_temp.isel(time=0))
mask3d = xr.DataArray(mask, dims=("N", "E")).expand_dims(time=obs_temp.time)



def rmse(a, b, mask3d):
    diff = a - b
    diff = diff.where(mask3d)
    rmse_grid = np.sqrt((diff ** 2).mean(dim="time"))
    spatial_mean_rmse = rmse_grid.mean(dim=["N", "E"]).item()
    return spatial_mean_rmse






metrics = {
    "Coarse": rmse(obs_temp, coarse_temp_interp, mask3d),
    "Bicubic": rmse(obs_temp, bicubic_temp, mask3d),
    "UNet": rmse(obs_temp, unet_temp, mask3d),
    "DDIM": rmse(obs_temp, ddim_ens_temp_mean, mask3d),
}

for name, value in metrics.items():
    print(f"{name}: Mean RMSE = {value}")



metric_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["RMSE"])
metric_df.to_csv("DDIM_conditional_derived/Metrics_Test_Set/cobweb/outputs/rmse_allmodels_temp.csv")