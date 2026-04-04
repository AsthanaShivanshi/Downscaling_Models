import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

# File paths
ref_path = "Dataset_Setup_I_Chronological_12km/TabsD_step1_latlon.nc"
etas = [0.0, 0.3, 0.5, 1.0]
downscaled_paths = [
    f"DDIM_conditional_derived/output_inference/ddim_downscaled_test_set_eta_{eta}.nc"
    for eta in etas
]

ref_ds = xr.open_dataset(ref_path).sel(time=slice("2011-01-01", "2023-12-31"))  # Test set (2011-2023)
ref_temp = ref_ds['TabsD'].values  




ref_temp_flat = ref_temp.flatten() #Pooling

# ECDF
ecdf_ref = ECDF(ref_temp_flat)


ref_pit = ecdf_ref(ref_temp_flat)


for eta, ds_path in zip(etas, downscaled_paths):
    ds = xr.open_dataset(ds_path)




    if 'sample' in ds['temp'].dims:
        temp = ds['temp'].values  #: (sample, time, lat, lon)
        temp_flat = temp.reshape(-1)
    else:
        temp_flat = ds['temp'].values.flatten()

    pit = ecdf_ref(temp_flat)


    plt.figure(figsize=(12, 8), dpi=1000)


    plt.hist(pit, bins=30, density=True, alpha=0.7, color='tab:blue', label=f"Downscaled (eta={eta})")


    plt.hist(ref_pit, bins=30, density=True, alpha=0.5, color='tab:orange', label="Test Set (2011-2023)")
    plt.title(f"Temperature : Calibration PIT Hist vs Reference Spatiotemporally Pooled (eta={eta})")
    plt.xlabel("Binned Cumulative Probability")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True)



    plt.tight_layout()
    plt.savefig(f"DDIM_conditional_derived/output_inference/PIT_histogram_eta_{eta}.pdf", format='pdf', dpi=1000)
    plt.close()