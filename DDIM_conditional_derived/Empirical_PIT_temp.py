import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

ref_path = "Dataset_Setup_I_Chronological_12km/TabsD_step1_latlon.nc"
etas = [0.0, 0.3, 0.5, 0.8, 1.0]
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
    bin_edges = np.linspace(0, 1, bins + 1)

    ref_hist, _ = np.histogram(ref_pit, bins=bin_edges, density=True)
    pit_hist, _ = np.histogram(pit, bins=bin_edges, density=True)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = (bin_edges[1] - bin_edges[0]) * 0.4  # Bar width, <0.5 for side-by-side

    fig, ax = plt.subplots(figsize=(12, 6), dpi=1000)

    ax.bar(bin_centers - width/2, ref_hist, width=width, label="Reference", color='tab:blue', alpha=0.8, edgecolor='black')
    ax.bar(bin_centers + width/2, pit_hist, width=width, label=f"Diffusion (eta={eta})", color='tab:orange', alpha=0.8, edgecolor='black')

    ax.set_xlabel("Binned Cumulative Probability", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.set_title(f"Temperature : PIT Histogram (eta={eta})", fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=13)

    plt.tight_layout()
    plt.savefig(f"DDIM_conditional_derived/output_inference/Temperature_PIT_histogram_eta_{eta}.pdf", format='pdf', dpi=1000)
    plt.close()
