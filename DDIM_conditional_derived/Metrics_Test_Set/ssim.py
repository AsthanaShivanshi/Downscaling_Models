import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim



files = [
    ("DDIM_conditional_derived/output_inference/ddim_downscaled_50steps_test_set_5samples_eta_0.0.nc", 50),
    ("DDIM_conditional_derived/output_inference/ddim_downscaled_100steps_test_set_5samples_eta_0.0.nc", 100),
    ("DDIM_conditional_derived/output_inference/ddim_downscaled_500steps_test_set_2samples_eta_0.0.nc", 500),
]



ref_temp = xr.open_dataset("Dataset_Setup_I_Chronological_12km/TabsD_step1_latlon.nc")["TabsD"].sel(time=slice("2011-01-01", "2023-12-31"))
ref_precip = xr.open_dataset("Dataset_Setup_I_Chronological_12km/RhiresD_step1_latlon.nc")["RhiresD"].sel(time=slice("2011-01-01", "2023-12-31"))

mask_temp = ~np.isnan(ref_temp.values)
mask_precip = ~np.isnan(ref_precip.values)

def pooled_ssim(ref, pred, mask):


    if pred.shape[1] < pred.shape[0]:
        pred = np.moveaxis(pred, 1, 0)  
    mask_broadcast = np.broadcast_to(mask, pred.shape)
    ref_broadcast = np.broadcast_to(ref.values, pred.shape)
    ref_flat = ref_broadcast[mask_broadcast]
    pred_flat = pred[mask_broadcast]
    valid = ~np.isnan(pred_flat)
    ref_flat = ref_flat[valid]
    pred_flat = pred_flat[valid]

    ref_flat = ref_flat.reshape(1, -1)
    pred_flat = pred_flat.reshape(1, -1)
    return ssim(ref_flat, pred_flat, data_range=ref_flat.max() - ref_flat.min())



ssim_temp = []
ssim_precip = []
steps = []

for f, step in files:
    ds = xr.open_dataset(f)
    temp = ds['temp'].sel(time=slice("2011-01-01", "2023-12-31")).values
    ssim_temp.append(pooled_ssim(ref_temp, temp, mask_temp))
    precip = ds['precip'].sel(time=slice("2011-01-01", "2023-12-31")).values
    precip = np.where(precip < 0, 0, precip)
    ssim_precip.append(pooled_ssim(ref_precip, precip, mask_precip))
    steps.append(step)


plt.figure(figsize=(12,8))
plt.plot(steps, ssim_temp, marker='o', label='Temperature')
plt.plot(steps, ssim_precip, marker='s', label='Precipitation')
plt.xlabel("Denoising Steps")
plt.ylabel("SSIM (pooled)")
plt.title("SSIM vs Denoising Steps")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("DDIM_conditional_derived/Metrics_Test_Set/ssim_vs_steps.pdf")
plt.show()