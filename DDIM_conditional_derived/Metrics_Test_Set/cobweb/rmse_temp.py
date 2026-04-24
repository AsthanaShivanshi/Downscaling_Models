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
    "DDIM_conditional_derived/output_inference/ddim_downscaled_50steps_test_set_5samples_eta_0.0.nc"
)["temp"].sel(time=slice("2011-01-01", "2023-12-31"))

# Interpolate coarse to obs grid
coarse_temp_interp = coarse_temp.interp_like(obs_temp, method="nearest")



# Ensure all arrays have N/E dims and coords
def ensure_NE(da, ref):
    # Rename if needed
    if "y" in da.dims and "x" in da.dims:
        da = da.rename({"y": "N", "x": "E"})
    # Assign coords if needed
    if "N" in da.dims and "E" in da.dims:
        da = da.assign_coords(N=ref.N, E=ref.E)
    return da




obs_temp = ensure_NE(obs_temp, obs_temp)
unet_temp = ensure_NE(unet_temp, obs_temp)
coarse_temp_interp = ensure_NE(coarse_temp_interp, obs_temp)
bicubic_temp = ensure_NE(bicubic_temp, obs_temp)



# Handle DDIM ensemble
if "sample" in ddim_temp.dims:
    ddim_ens_temp = ddim_temp.rename({"sample": "ensemble"})
else:
    ddim_ens_temp = ddim_temp
ddim_ens_temp = ensure_NE(ddim_ens_temp, obs_temp)
ddim_ens_temp_mean = ddim_ens_temp.mean(dim="ensemble") if "ensemble" in ddim_ens_temp.dims else ddim_ens_temp



ddim_ens_temp_mean = ensure_NE(ddim_ens_temp_mean, obs_temp)

print("obs_temp shape:", obs_temp.shape)
print("unet_temp shape:", unet_temp.shape)
print("coarse_temp_interp shape:", coarse_temp_interp.shape)
print("bicubic_temp shape:", bicubic_temp.shape)
print("ddim_ens_temp_mean shape:", ddim_ens_temp_mean.shape)




mask = ~np.isnan(obs_temp.isel(time=0))
mask3d = xr.DataArray(mask, dims=("N", "E")).expand_dims(time=obs_temp.time)

def rmse(a, b, mask3d):
    diff = a - b
    diff = diff.where(mask3d)
    rmse_grid = np.sqrt((diff ** 2).mean(dim="time"))
    spatial_mean_rmse = rmse_grid.mean(dim=["N", "E"]).item()
    return spatial_mean_rmse


def ensemble_rmse(obs, ens_pred, mask3d):
    rmse_list = []
    for i in range(ens_pred.sizes["ensemble"]):
        rmse_val = rmse(obs, ens_pred.isel(ensemble=i), mask3d)
        rmse_list.append(rmse_val)
    return np.mean(rmse_list)

def best_ensemble_rmse(obs, ens_pred, mask3d):
    rmse_list = []
    for i in range(ens_pred.sizes["ensemble"]):
        rmse_val = rmse(obs, ens_pred.isel(ensemble=i), mask3d)
        rmse_list.append(rmse_val)
    rmse_array = np.array(rmse_list)
    best_idx = np.argmin(rmse_array)
    return rmse_array[best_idx], best_idx


metrics = {
    "Coarse": {"RMSE": rmse(obs_temp, coarse_temp_interp, mask3d)},
    "Bicubic": {"RMSE": rmse(obs_temp, bicubic_temp, mask3d)},
    "UNet": {"RMSE": rmse(obs_temp, unet_temp, mask3d)},
}

if "ensemble" in ddim_ens_temp.dims:
    mean_rmse = ensemble_rmse(obs_temp, ddim_ens_temp, mask3d)
    best_rmse, best_idx = best_ensemble_rmse(obs_temp, ddim_ens_temp, mask3d)
    metrics["DDIM"] = {
        "RMSE": mean_rmse,
        "Best RMSE": best_rmse,
        "Best Ensemble Index": best_idx
    }
else:
    metrics["DDIM"] = {"RMSE": rmse(obs_temp, ddim_ens_temp, mask3d)}

metric_df = pd.DataFrame.from_dict(metrics, orient="index")
metric_df.to_csv("DDIM_conditional_derived/Metrics_Test_Set/cobweb/outputs/rmse_allmodels_temp.csv")