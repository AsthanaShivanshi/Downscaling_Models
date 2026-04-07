import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

ref_path = "Dataset_Setup_I_Chronological_12km/RhiresD_step1_latlon.nc"
etas = [0.0] #Etas removed for 0.0 case : AsthanaSh
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


for eta, ds_path in zip(etas, downscaled_paths):
    print(f"Processing eta={eta}...")



    ds = xr.open_dataset(ds_path)
    precip = ds['precip']  # shape: (time, sample, lat, lon)


    precip_values = precip.values 
    precip_values = np.moveaxis(precip_values, 1, 0)  # (sample, time, lat, lon)

    mask_broadcast = np.broadcast_to(mask, precip_values.shape)  # (sample, time, lat, lon)
    precip_masked = np.where(mask_broadcast, precip_values, np.nan) 
    precip_masked= np.where(precip_masked < 0, 0, precip_masked) #Making sure no neg precip

    precip_flat = precip_masked[~np.isnan(precip_masked)]
    pit = ecdf_ref(precip_flat)


    bins = 50
    bin_edges = np.linspace(0, 1, bins + 1)

    ref_hist, _ = np.histogram(ref_pit, bins=bin_edges, density=True)
    pit_hist, _ = np.histogram(pit, bins=bin_edges, density=True)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = (bin_edges[1] - bin_edges[0]) * 0.4  # Bar width

    fig, ax = plt.subplots(figsize=(12, 6), dpi=1000)

    ax.bar(bin_centers - width/2, ref_hist, width=width, label="Reference", color='tab:blue', alpha=0.8, edgecolor='black')
    ax.bar(bin_centers + width/2, pit_hist, width=width, label=f"Diffusion (eta={eta})", color='tab:orange', alpha=0.8, edgecolor='black')

    ax.axhline(1, color='red', linestyle='--', linewidth=1, label='Uniform Density')

    ax.set_xlabel("Binned Cumulative Probability", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.set_title(f"Precipitation : PIT Histogram (eta={eta})", fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=13)

    plt.tight_layout()
    plt.savefig(f"DDIM_conditional_derived/output_inference/Precip_PIT_histogram_50steps_5samples_eta_{eta}.pdf", format='pdf', dpi=1000)
    plt.close()
