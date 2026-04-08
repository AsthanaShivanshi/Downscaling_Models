import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity

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
    imagewise_scores = []
    n_samples, n_time, N, E = pred.shape




    for s in range(n_samples):
        for t in range(n_time):
            ref_img = ref.values[t]
            pred_img = pred[s, t]
            mask_img = mask[t]




            if np.sum(mask_img) > 0:
                ref_valid = ref_img[mask_img]
                pred_valid = pred_img[mask_img]
                if ref_valid.size > 0 and pred_valid.size > 0:
                    try:
                        ssim = structural_similarity(
                            ref_valid, pred_valid, 
                            data_range=ref_valid.max() - ref_valid.min()
                        )
                    except Exception:
                        ssim = np.nan
                else:
                    ssim = np.nan
            else:
                ssim = np.nan

            imagewise_scores.append(ssim)

    return np.nanmean(imagewise_scores)







ssim_temp = []
ssim_precip = []
steps = []




for f, step in files:
    ds = xr.open_dataset(f)
    temp = ds['temp'].sel(time=slice("2011-01-01", "2023-12-31")).values
    temp = np.moveaxis(temp, 1, 0)
    ssim_temp.append(pooled_ssim(ref_temp, temp, mask_temp))

    precip = ds['precip'].sel(time=slice("2011-01-01", "2023-12-31")).values
    precip = np.where(precip < 0, 0, precip)
    precip = np.moveaxis(precip, 1, 0)
    ssim_precip.append(pooled_ssim(ref_precip, precip, mask_precip))
    steps.append(step)





print("Steps:", steps)
print("SSIM Temperature:", ssim_temp)
print("SSIM Precipitation:", ssim_precip)





plt.figure(figsize=(10,10))
plt.subplot(2,1,1) 
plt.plot(steps, ssim_temp, marker='o', color='red', label='Temperature')
plt.ylabel(r"SSIM(pooled) $\uparrow$") 
plt.title("SSIM vs Denoising Steps")
plt.legend()




plt.subplot(2,1,2) 
plt.plot(steps, ssim_precip, marker='s', color='blue', label='Precipitation')
plt.ylabel(r"SSIM(pooled) $\uparrow$") 
plt.title("SSIM vs Denoising Steps")
plt.legend()




plt.tight_layout()
plt.savefig("DDIM_conditional_derived/Metrics_Test_Set/ssim_vs_steps.pdf")
plt.show()