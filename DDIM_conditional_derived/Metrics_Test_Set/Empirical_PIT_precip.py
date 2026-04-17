import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF




#---------------------------------------------------------------------#

ref_path = "Dataset_Setup_I_Chronological_12km/RhiresD_step1_latlon.nc"
etas = [0.0]
downscaled_paths = [
    f"DDIM_conditional_derived/output_inference/ddim_downscaled_50steps_test_set_5samples_eta_{eta}.nc"
    for eta in etas
]

ref_ds = xr.open_dataset(ref_path).sel(time=slice("2011-01-01", "2023-12-31"))
ref_precip = ref_ds['RhiresD']
mask = ~np.isnan(ref_precip.values)
ref_precip_masked = np.where(mask, ref_precip.values, np.nan)
ref_precip_flat = ref_precip_masked[mask]
ecdf_ref = ECDF(ref_precip_flat)
ref_pit = ecdf_ref(ref_precip_flat)

bins = 50
bin_start = 0.4
bin_end = 1.0


#---------------------------------------------------------------------#


for eta, ds_path in zip(etas, downscaled_paths):
    print(f"Processing eta={eta}...")

    ds = xr.open_dataset(ds_path)
    precip = ds['precip']  # shape: (time, sample, lat, lon)
    precip_values = precip.values 
    precip_values = np.moveaxis(precip_values, 1, 0)  # (sample, time, lat, lon)
    mask_broadcast = np.broadcast_to(mask, precip_values.shape)  # (sample, time, lat, lon)
    precip_masked = np.where(mask_broadcast, precip_values, np.nan) 

    precip_masked = np.where(precip_masked < 0, 0, precip_masked) # No negative precip
    precip_flat = precip_masked[~np.isnan(precip_masked)]
    pit = ecdf_ref(precip_flat)

    bin_edges = np.linspace(0, 1, bins + 1)
    ref_hist, _ = np.histogram(ref_pit, bins=bin_edges, density=True)
    pit_hist, _ = np.histogram(pit, bins=bin_edges, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = (bin_edges[1] - bin_edges[0]) * 0.4


#---------------------------------------------------------------------#


    bin_edges = np.linspace(bin_start, bin_end, bins + 1)
    ref_hist, _ = np.histogram(ref_pit, bins=bin_edges, density=True)
    pit_hist, _ = np.histogram(pit, bins=bin_edges, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = (bin_edges[1] - bin_edges[0]) * 0.4

    fig, ax = plt.subplots(figsize=(12, 8), dpi=1000)

    ax.bar(bin_centers - width/2, ref_hist, width=width, 
           label="Reference (2011–2023)", color='#4575b4', alpha=0.85, edgecolor='black')
    ax.bar(bin_centers + width/2, pit_hist, width=width, 
           label=f"Diffusion (η={eta})", color='#d73027', alpha=0.85, edgecolor='black')



    ax.plot([bin_start, bin_end], [1, 1], color='gray', linestyle='-', linewidth=2, label='Uniform Density (Ideal)')

    ax.set_xlabel("Probability Integral Transform (PIT)", fontsize=15)
    ax.set_ylabel("Density", fontsize=15)
    ax.set_title(f"PIT Histogram for Precipitation (All Grid Cells)", fontsize=17, pad=15)
    ax.set_xlim(bin_start, bin_end)

    ax.annotate(f"Time steps: {ref_precip_flat.shape[0]}\nSamples: {precip.shape[0]}", 
                xy=(0.99, 0.97), xycoords='axes fraction', ha='right', va='top', fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
    ax.grid(True, linestyle=':', linewidth=0.7, alpha=0.7)
    ax.legend(fontsize=13, frameon=True)

    plt.tight_layout()
    plt.savefig(f"DDIM_conditional_derived/Metrics_Test_Set/Precip_PIT_histogram_50steps_5samples_eta_{eta}.pdf", format='pdf', dpi=1000)
    plt.close()
