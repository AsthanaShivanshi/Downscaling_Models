import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity

files = [
    ("DDIM_conditional_derived/output_inference/ddim_downscaled_50steps_test_set_5samples_eta_0.0.nc", 50),
    ("DDIM_conditional_derived/output_inference/ddim_downscaled_100steps_test_set_5samples_eta_0.0.nc", 100),
    ("DDIM_conditional_derived/output_inference/ddim_downscaled_250steps_test_set_5samples_eta_0.0.nc", 250),
    ("DDIM_conditional_derived/output_inference/UNet_downscaled_test_set_2011_2023.nc",None)
]



ref_temp = xr.open_dataset("Dataset_Setup_I_Chronological_12km/TabsD_step1_latlon.nc")["TabsD"].sel(time=slice("2011-01-01", "2023-12-31"))
ref_precip = xr.open_dataset("Dataset_Setup_I_Chronological_12km/RhiresD_step1_latlon.nc")["RhiresD"].sel(time=slice("2011-01-01", "2023-12-31"))

mask_temp = ~np.isnan(ref_temp.values)
mask_precip = ~np.isnan(ref_precip.values)



print("temp shape bfor ssim : ", ref_temp.values.shape)
print("precip shape bfor ssim : ", ref_precip.values.shape)

def pooled_ssim(ref, pred, mask):
    imagewise_scores = []

    print("pooled_ssim shapes:")
    print("  ref:", ref.values.shape)
    print("  pred:", pred.shape)
    print("  mask:", mask.shape)

    if pred.ndim == 3:
        pred = np.expand_dims(pred, axis=0)  # (1, time, N, E)
    if pred.shape[1:] != ref.values.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape}, ref {ref.values.shape}")

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


    if step is None:  # UNet



        precip = ds['precip'].sel(time=slice("2011-01-01", "2023-12-31")).values  # (4748, 240, 370)
        temp = ds['temp'].sel(time=slice("2011-01-01", "2023-12-31")).values      # (4748, 240, 370)
        precip = precip[None, ...]  # (1, 4748, 240, 370)
        temp = temp[None, ...]      # (1, 4748, 240, 370)




        
    else:
        temp = ds['temp'].sel(time=slice("2011-01-01", "2023-12-31")).values
        temp = np.moveaxis(temp, 1, 0)  # (n_samples, time, lat, lon)
        precip = ds['precip'].sel(time=slice("2011-01-01", "2023-12-31")).values
        precip = np.where(precip < 0, 0, precip)
        precip = np.moveaxis(precip, 1, 0)

    ssim_temp.append(pooled_ssim(ref_temp, temp, mask_temp))
    ssim_precip.append(pooled_ssim(ref_precip, precip, mask_precip))

    if step is None:
        steps.append("UNet")
    else:
        steps.append(step)







print("Steps:", steps)
print("SSIM Temperature:", ssim_temp)
print("SSIM Precipitation:", ssim_precip)




step_labels = [str(s) for s in steps]
x_pos = list(range(len(steps)))  




plt.figure(figsize=(10,10))
plt.subplot(2,1,1) 
plt.plot(x_pos, ssim_temp, marker='o', color='red', label='Temperature')
plt.ylabel(r"SSIM(pooled) $\uparrow$") 
plt.title("SSIM vs Denoising Steps")
plt.legend()
plt.xticks(x_pos, step_labels)





plt.subplot(2,1,2) 
plt.plot(x_pos, ssim_precip, marker='s', color='blue', label='Precipitation')
plt.ylabel(r"SSIM(pooled) $\uparrow$") 
plt.title("SSIM vs Denoising Steps")
plt.legend()
plt.xticks(x_pos, step_labels)





plt.tight_layout()
plt.savefig("DDIM_conditional_derived/Metrics_Test_Set/ssim_vs_steps.pdf")
plt.show()
