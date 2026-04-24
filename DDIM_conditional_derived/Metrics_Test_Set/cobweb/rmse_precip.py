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
    "DDIM_conditional_derived/output_inference/ddim_downscaled_30steps_test_set_11samples_2011_2023.nc"
)["precip"].sel(time=slice("2011-01-01", "2023-12-31"))

# Interpolate coarse to obs grid
coarse_precip_interp = coarse_precip.interp_like(obs_precip, method="nearest")

# Clip negatives
obs_precip = obs_precip.clip(min=0)
unet_precip = unet_precip.clip(min=0)
bicubic_precip = bicubic_precip.clip(min=0)
ddim_precip = ddim_precip.clip(min=0)
coarse_precip_interp = coarse_precip_interp.clip(min=0)

# Ensure all arrays have N/E dims and coords
def ensure_NE(da, ref):
    # Rename if needed
    if "y" in da.dims and "x" in da.dims:
        da = da.rename({"y": "N", "x": "E"})
    # Assign coords if needed
    if "N" in da.dims and "E" in da.dims:
        da = da.assign_coords(N=ref.N, E=ref.E)
    return da

obs_precip = ensure_NE(obs_precip, obs_precip)
unet_precip = ensure_NE(unet_precip, obs_precip)
coarse_precip_interp = ensure_NE(coarse_precip_interp, obs_precip)
bicubic_precip = ensure_NE(bicubic_precip, obs_precip)

# Handle DDIM ensemble
if "sample" in ddim_precip.dims:
    ddim_ens_precip = ddim_precip.rename({"sample": "ensemble"})
else:
    ddim_ens_precip = ddim_precip
ddim_ens_precip = ensure_NE(ddim_ens_precip, obs_precip)
ddim_ens_precip_mean = ddim_ens_precip.mean(dim="ensemble") if "ensemble" in ddim_ens_precip.dims else ddim_ens_precip

# Also ensure mean has correct dims/coords
ddim_ens_precip_mean = ensure_NE(ddim_ens_precip_mean, obs_precip)

# Print shapes for sanity
print("obs_precip shape:", obs_precip.shape)
print("unet_precip shape:", unet_precip.shape)
print("coarse_precip_interp shape:", coarse_precip_interp.shape)
print("bicubic_precip shape:", bicubic_precip.shape)
print("ddim_ens_precip_mean shape:", ddim_ens_precip_mean.shape)

# Create mask for valid data
mask = ~np.isnan(obs_precip.isel(time=0))
mask3d = xr.DataArray(mask, dims=("N", "E")).expand_dims(time=obs_precip.time)

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
    "Coarse": {"RMSE": rmse(obs_precip, coarse_precip_interp, mask3d)},
    "Bicubic": {"RMSE": rmse(obs_precip, bicubic_precip, mask3d)},
    "UNet": {"RMSE": rmse(obs_precip, unet_precip, mask3d)},
}

if "ensemble" in ddim_ens_precip.dims:
    mean_rmse = ensemble_rmse(obs_precip, ddim_ens_precip, mask3d)
    best_rmse, best_idx = best_ensemble_rmse(obs_precip, ddim_ens_precip, mask3d)
    metrics["DDIM"] = {
        "RMSE": mean_rmse,
        "Best RMSE": best_rmse,
        "Best Ensemble Index": best_idx
    }
else:
    metrics["DDIM"] = {"RMSE": rmse(obs_precip, ddim_ens_precip, mask3d)}

metric_df = pd.DataFrame.from_dict(metrics, orient="index")
metric_df.to_csv("DDIM_conditional_derived/Metrics_Test_Set/cobweb/outputs/rmse_allmodels_precip.csv")