import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

ref_path = "Dataset_Setup_I_Chronological_12km/TabsD_step1_latlon.nc"
etas = [0.0]
downscaled_paths = [
    f"DDIM_conditional_derived/output_inference/ddim_downscaled_50steps_test_set_5samples_eta_{eta}.nc"
    for eta in etas
]

bins = 50



#------------------------------------------------------------------------#

ref_ds = xr.open_dataset(ref_path).sel(time=slice("2011-01-01", "2023-12-31"))
ref_temp = ref_ds['TabsD']
mask = ~np.isnan(ref_temp.values)
ref_temp_masked = np.where(mask, ref_temp.values, np.nan)
ref_temp_flat = ref_temp_masked[mask]
ecdf_ref = ECDF(ref_temp_flat)
ref_pit = ecdf_ref(ref_temp_flat)

#------------------------------------------------------------------------#

for eta, ds_path in zip(etas, downscaled_paths):
    print(f"Processing eta={eta}...")

    ds = xr.open_dataset(ds_path)
    temp = ds['temp']  # shape: (time, sample, lat, lon)
    temp_values = temp.values 
    temp_values = np.moveaxis(temp_values, 1, 0)  # (sample, time, lat, lon)
    mask_broadcast = np.broadcast_to(mask, temp_values.shape)  # (sample, time, lat, lon)
    temp_masked = np.where(mask_broadcast, temp_values, np.nan) 
    temp_flat = temp_masked[~np.isnan(temp_masked)]
    pit = ecdf_ref(temp_flat)

    bin_edges = np.linspace(0, 1, bins + 1)
    ref_hist, _ = np.histogram(ref_pit, bins=bin_edges, density=True)
    pit_hist, _ = np.histogram(pit, bins=bin_edges, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = (bin_edges[1] - bin_edges[0]) * 0.4


    #------------------------------------------------------------------------#

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.bar(bin_centers - width/2, ref_hist, width=width, label="Reference (2011–2023)", color='#4575b4', alpha=0.85, edgecolor='black')
    ax.bar(bin_centers + width/2, pit_hist, width=width, label=f"Diffusion (η={eta})", color='#d73027', alpha=0.85, edgecolor='black')
    ax.plot([0, 1], [1, 1], color='gray', linestyle='-', linewidth=2, label='Uniform Density (Ideal)')
    ax.set_xlabel("Probability Integral Transform (PIT)", fontsize=15)
    ax.set_ylabel("Density", fontsize=15)
    ax.set_title(f"PIT Histogram for Temperature (All Grid Cells)", fontsize=17, pad=15)
    ax.annotate(f"Time steps: {ref_temp_flat.shape[0]}\nSamples: {temp.shape[0]}", 
                xy=(0.99, 0.97), xycoords='axes fraction', ha='right', va='top', fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
    ax.grid(True, linestyle=':', linewidth=0.7, alpha=0.7)
    ax.legend(fontsize=13, frameon=True)
    plt.tight_layout()
    plt.savefig(f"DDIM_conditional_derived/Metrics_Test_Set/Temperature_PIT_histogram_50steps_5samples_eta_{eta}.pdf", format='pdf', dpi=1000)

    plt.close()