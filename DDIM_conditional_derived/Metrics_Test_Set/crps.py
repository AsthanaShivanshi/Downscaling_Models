import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import properscoring as ps

files = [
    ("DDIM_conditional_derived/output_inference/ddim_downscaled_50steps_test_set_5samples_eta_0.0.nc", 50),
    ("DDIM_conditional_derived/output_inference/ddim_downscaled_100steps_test_set_5samples_eta_0.0.nc", 100),
    ("DDIM_conditional_derived/output_inference/ddim_downscaled_250steps_test_set_5samples_eta_0.0.nc", 250),
    ("DDIM_conditional_derived/output_inference/UNet_downscaled_test_set_2011_2023.nc", None)
]




ref_temp_ds = xr.open_dataset("Dataset_Setup_I_Chronological_12km/TabsD_step1_latlon.nc")
ref_temp = ref_temp_ds["TabsD"].sel(time=slice("2011-01-01", "2023-12-31"))
ref_temp_ds.close()

ref_precip_ds = xr.open_dataset("Dataset_Setup_I_Chronological_12km/RhiresD_step1_latlon.nc")
ref_precip = ref_precip_ds["RhiresD"].sel(time=slice("2011-01-01", "2023-12-31"))
ref_precip_ds.close()



mask_temp = ~np.isnan(ref_temp.values)
mask_precip = ~np.isnan(ref_precip.values)





def crps_score(ref, pred, mask):
    if pred.shape[0] == 1:  # UNet mae
        abs_err = np.abs(pred[0] - ref.values)
        abs_err = np.where(mask, abs_err, np.nan)
        return np.nanmean(abs_err)
    else:  # DDIM, ensemble
        ens = pred[:, mask]  # shape: (n_samples, n_points)
        obs = ref.values[mask]  # shape: (n_points,)
        crps = ps.crps_ensemble(obs, ens.T)
        return np.nanmean(crps)

crps_temp = []
crps_precip = []
steps = []





for f, step in files:
    ds = xr.open_dataset(f)
    if step is None:  # UNet
        precip = ds['precip'].sel(time=slice("2011-01-01", "2023-12-31")).values
        temp = ds['temp'].sel(time=slice("2011-01-01", "2023-12-31")).values
        precip = precip[None, ...]
        temp = temp[None, ...]
    else:
        temp = ds['temp'].sel(time=slice("2011-01-01", "2023-12-31")).values
        temp = np.moveaxis(temp, 1, 0)  # (n_samples, time, lat, lon)
        precip = ds['precip'].sel(time=slice("2011-01-01", "2023-12-31")).values
        precip = np.where(precip < 0, 0, precip)
        precip = np.moveaxis(precip, 1, 0)
    ds.close()  # Close the dataset after loading variables

    crps_temp.append(crps_score(ref_temp, temp, mask_temp))
    crps_precip.append(crps_score(ref_precip, precip, mask_precip))
    steps.append("UNet" if step is None else step)






print("Steps:", steps)
print("CRPS/MAE Temperature:", crps_temp)
print("CRPS/MAE Precipitation:", crps_precip)

step_labels = [str(s) for s in steps]
x_pos = list(range(len(steps)))

plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
plt.plot(x_pos, crps_temp, marker='o', color='red', label='Temperature')
plt.ylabel(r"CRPS (DDIM) / MAE (UNet) $\downarrow$")
plt.title("CRPS/MAE vs Denoising Steps")
plt.legend()
plt.xticks(x_pos, step_labels)

plt.subplot(2,1,2)
plt.plot(x_pos, crps_precip, marker='s', color='blue', label='Precipitation')
plt.ylabel(r"CRPS (DDIM) / MAE (UNet) $\downarrow$")
plt.title("CRPS/MAE vs Denoising Steps")
plt.legend()
plt.xticks(x_pos, step_labels)



plt.tight_layout()
plt.savefig("DDIM_conditional_derived/Metrics_Test_Set/crps_vs_steps.pdf")
plt.show()