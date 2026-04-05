import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

ref_path = "Dataset_Setup_I_Chronological_12km/RhiresD_step1_latlon.nc"
etas = [0.0, 0.3, 0.5, 1.0]
downscaled_paths = [
    f"DDIM_conditional_derived/output_inference/ddim_downscaled_test_set_eta_{eta}.nc"
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
    ds = xr.open_dataset(ds_path)
    precip = ds['precip']  # shape: (time, sample, lat, lon)

    precip_values = precip.values 
   
    precip_values = np.moveaxis(precip_values, 1, 0) 

    mask_broadcast = np.broadcast_to(mask, precip_values.shape[1:]) 
    precip_masked = np.where(mask_broadcast, precip_values, np.nan) 

    precip_flat = precip_masked[~np.isnan(precip_masked)]

    pit = ecdf_ref(precip_flat)


    plt.figure(figsize=(12, 8), dpi=1000)
    bins = 20

    plt.hist(
        pit, bins=bins, range=(0,1), density=True, 
        histtype='step', linewidth=2, color='green', label=f"Diffusion samples (eta={eta})"
    )
    plt.hist(
        ref_pit, bins=bins, range=(0,1), density=True, 
        histtype='step', linewidth=2, color='black', label="Reference Test Set (2011-2023)"
    )
    plt.xlim(0, 1)
    plt.xlabel("Binned Cumulative Probability")
    plt.ylabel("PDF")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"DDIM_conditional_derived/output_inference/Precip_PIT_histogram_eta_{eta}.pdf", format='pdf', dpi=1000)
    plt.close()
