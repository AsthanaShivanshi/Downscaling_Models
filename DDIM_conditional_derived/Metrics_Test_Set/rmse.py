import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error

# File paths and denoising steps
files = [
    ("DDIM_conditional_derived/output_inference/ddim_downscaled_50steps_test_set_5samples_eta_0.0.nc", 50),
    ("DDIM_conditional_derived/output_inference/ddim_downscaled_100steps_test_set_5samples_eta_0.0.nc", 100),
    ("DDIM_conditional_derived/output_inference/ddim_downscaled_500steps_test_set_2samples_eta_0.0.nc", 500),
]

ref_temp = xr.open_dataset("Dataset_Setup_I_Chronological_12km/TabsD_step1_latlon.nc")["TabsD"].sel(time=slice("2011-01-01", "2023-12-31"))


ref_precip = xr.open_dataset("Dataset_Setup_I_Chronological_12km/RhiresD_step1_latlon.nc")["RhiresD"].sel(time=slice("2011-01-01", "2023-12-31"))




mask_temp = ~np.isnan(ref_temp.values)
mask_precip = ~np.isnan(ref_precip.values)

def pooled_rmse(ref, pred, mask):


    if pred.shape[1] < pred.shape[0]:
        pred = np.moveaxis(pred, 1, 0)  # (sample, time, N, E)
    # Broadcast mask to pred shape
    mask_broadcast = np.broadcast_to(mask, pred.shape)
    ref_broadcast = np.broadcast_to(ref.values, pred.shape)
    # Masked flatten
    ref_flat = ref_broadcast[mask_broadcast]
    pred_flat = pred[mask_broadcast]

    valid = ~np.isnan(pred_flat)
    ref_flat = ref_flat[valid]
    pred_flat = pred_flat[valid]
    return np.sqrt(mean_squared_error(ref_flat, pred_flat))

rmse_temp = []
rmse_precip = []
steps = []

for f, step in files:
    ds = xr.open_dataset(f)
    temp = ds['temp'].sel(time=slice("2011-01-01", "2023-12-31")).values  # (time, sample, N, E)
    rmse_temp.append(pooled_rmse(ref_temp, temp, mask_temp))


    precip = ds['precip'].sel(time=slice("2011-01-01", "2023-12-31")).values
    precip = np.where(precip < 0, 0, precip)
    rmse_precip.append(pooled_rmse(ref_precip, precip, mask_precip))
    steps.append(step)




plt.figure(figsize=(12,8))
plt.plot(steps, rmse_temp, marker='o', label='Temperature')
plt.plot(steps, rmse_precip, marker='s', label='Precipitation')
plt.xlabel("Denoising Steps")
plt.ylabel("RMSE (spatiotemporally pooled)")
plt.title("RMSE vs Denoising Steps")
plt.legend()
plt.tight_layout()
plt.savefig("DDIM_conditional_derived/Metrics_Test_Set/rmse_vs_denoising_steps.pdf")
plt.show()