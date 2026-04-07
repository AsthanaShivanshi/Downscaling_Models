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

def ssim(ref, pred, mask):

#ref:xarray datarray with dims (time, lat, lon)

    imagewise_scores = []
    n_samples, n_time, N, E = pred.shape

    for s in range(n_samples):
        for t in range(n_time):
            ref_img = ref.values[t]  # (N, E) #gives underlying numpy arry. 
            pred_img = pred[s, t]  # (N, E)
            mask_img= mask[t]



            #calcualting only for valid pixs


            if np.sum(mask_img) > 0:




                ssim_score = ssim(ref_img, pred_img, data_range=ref_img.max() - ref_img.min(), mask=mask_img)
                imagewise_scores.append(ssim_score)

    return np.mean(imagewise_scores)



ssim_temp = []
ssim_precip = []
steps = []

for f, step in files:

    ds = xr.open_dataset(f)
    temp = ds['temp'].sel(time=slice("2011-01-01", "2023-12-31")).values  # (time, sample, N, E)


    temp = np.moveaxis(temp, 1, 0)  # (sample, time, N, E)
    ssim_temp.append(ssim(ref_temp, temp, mask_temp))

    precip = ds['precip'].sel(time=slice("2011-01-01", "2023-12-31")).values
    
    precip = np.where(precip < 0, 0, precip)
    precip = np.moveaxis(precip, 1, 0)  # (sample, time, N, E)


    ssim_precip.append(ssim(ref_precip, precip, mask_precip))
    steps.append(step)



plt.figure(figsize=(10,10))


#2 rows, 1 column, 1st


plt.subplot(2,1,1) 
plt.plot(steps, ssim_temp, marker='o', color='red', label='Temperature')
plt.ylabel(r"SSIM(pooled) $\uparrow$") #uparrow
plt.title("SSIM vs Denoising Steps")
plt.legend()

#2 rows, 1 column, 2nd

plt.subplot(2,1,2) 
plt.plot(steps, ssim_precip, marker='s', color='blue', label='Precipitation')
plt.ylabel(r"SSIM(pooled) $\uparrow$") #uparrow
plt.title("SSIM vs Denoising Steps")
plt.legend()






plt.tight_layout()
plt.savefig("DDIM_conditional_derived/Metrics_Test_Set/ssim_vs_steps.pdf")
plt.show()