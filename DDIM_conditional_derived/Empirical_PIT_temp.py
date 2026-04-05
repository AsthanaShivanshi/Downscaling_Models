import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

ref_path = "Dataset_Setup_I_Chronological_12km/TabsD_step1_latlon.nc"
etas = [0.0, 0.3, 0.5, 1.0]
downscaled_paths = [
    f"DDIM_conditional_derived/output_inference/ddim_downscaled_test_set_eta_{eta}.nc"
    for eta in etas
]



ref_ds = xr.open_dataset(ref_path).sel(time=slice("2011-01-01", "2023-12-31"))
ref_temp = ref_ds['TabsD']

mask = ~np.isnan(ref_temp.values)



ref_temp_masked = np.where(mask, ref_temp.values, np.nan)
ref_temp_flat = ref_temp_masked[mask]
ecdf_ref = ECDF(ref_temp_flat)
ref_pit = ecdf_ref(ref_temp_flat)

for eta, ds_path in zip(etas, downscaled_paths):
    print(f"Processing eta={eta}...")



    ds = xr.open_dataset(ds_path)
    temp = ds['temp']  # shape: (time, sample, lat, lon)

    temp_values = temp.values 
    temp_values = np.moveaxis(temp_values, 1, 0) 

    mask_broadcast = np.broadcast_to(mask, temp_values.shape[1:])  # (time, lat, lon)
    temp_masked = np.where(mask_broadcast, temp_values, np.nan) 

    temp_flat = temp_masked[~np.isnan(temp_masked)]
    pit = ecdf_ref(temp_flat)

    bins = 50
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), dpi=300, sharex=True)

    # Ref: top
    axs[0].hist(
        ref_pit, bins=bins, density=True, 
        color='tab:blue', alpha=0.8, edgecolor='black'
    )
    axs[0].set_ylabel("Density", fontsize=14)
    axs[0].set_title("Reference Test Set (2011-2023)", fontsize=16)
    axs[0].grid(True, linestyle='--', alpha=0.5)

    # Diff: bottom
    axs[1].hist(
        pit, bins=bins, density=True, 
        color='tab:orange', alpha=0.8, edgecolor='black'
    )
    axs[1].set_xlabel("Binned Cumulative Probability", fontsize=14)
    axs[1].set_ylabel("Density", fontsize=14)
    axs[1].set_title(f"Diffusion Samples (eta={eta})", fontsize=16)
    axs[1].grid(True, linestyle='--', alpha=0.5)

    ylim = max(axs[0].get_ylim()[1], axs[1].get_ylim()[1])
    axs[0].set_ylim(0, ylim)
    axs[1].set_ylim(0, ylim)

    plt.suptitle("Temperature :PIT Histograms", fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"DDIM_conditional_derived/output_inference/Temperature_PIT_histogram_eta_{eta}.pdf", format='pdf', dpi=300)
    plt.close()

