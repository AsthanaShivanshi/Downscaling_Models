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

def rmse(ref, pred, mask):


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


    temp = np.moveaxis(temp, 1, 0)  # (sample, time, N, E)
    rmse_temp.append(rmse(ref_temp, temp, mask_temp))

    precip = ds['precip'].sel(time=slice("2011-01-01", "2023-12-31")).values
    
    precip = np.where(precip < 0, 0, precip)
    precip = np.moveaxis(precip, 1, 0)  # (sample, time, N, E)


    rmse_precip.append(rmse(ref_precip, precip, mask_precip))
    steps.append(step)




plt.figure(figsize=(10,10))
#2 rows, 1 column, 1st
plt.subplot(2,1,1) 
plt.plot(steps, rmse_temp, marker='o', color='red', label='Temperature')
plt.ylabel("RMSE(pooled) $\downarrow$") #downarrow
plt.title("RMSE vs Denoising Steps")
plt.legend()



#2 rows, 1 column, 2nd

plt.subplot(2,1,2) 
plt.plot(steps, rmse_precip, marker='s', color='blue', label='Precipitation')
plt.ylabel("RMSE(pooled) $\downarrow$") #downarrow
plt.title("RMSE vs Denoising Steps")
plt.legend()






plt.tight_layout()
plt.savefig("DDIM_conditional_derived/Metrics_Test_Set/rmse_vs_steps.pdf")
plt.show()