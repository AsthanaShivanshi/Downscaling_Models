import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import argparse
import sys
sys.path.append("../Processing_and_Analysis_Scripts/Prelim_Stats_Obs_only")
from closest_grid_cell import select_nearest_grid_cell

#--------------------------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument("--target_lat", type=float, required=True)
parser.add_argument("--target_lon", type=float, required=True)
parser.add_argument("--city", type=str, required=True)
args = parser.parse_args()
target_lat = args.target_lat
target_lon = args.target_lon
city = args.city


etas = [0.0]
bins = 30
#-----------------------------------------------------------------------#

ref_path = "Dataset_Setup_I_Chronological_12km/RhiresD_step1_latlon.nc"

downscaled_paths = [
    f"DDIM_conditional_derived/output_inference/ddim_downscaled_50steps_test_set_5samples_eta_{eta}.nc"
    for eta in etas
]

#-----------------------------------------------------------------------#

ref_ds = xr.open_dataset(ref_path).sel(time=slice("2011-01-01", "2023-12-31"))
ref_cell = select_nearest_grid_cell(ref_ds, target_lat, target_lon, var_name='RhiresD')
ref_precip = ref_cell['data'].values  # shape: (time,)
mask = ~np.isnan(ref_precip)
ref_precip_masked = np.where(mask, ref_precip, np.nan)
ref_precip_flat = ref_precip_masked[mask]
ecdf_ref = ECDF(ref_precip_flat)
ref_pit = ecdf_ref(ref_precip_flat)

#-----------------------------------------------------------------------#

for eta, ds_path in zip(etas, downscaled_paths):
    print(f"Processing eta={eta}...")

    ds = xr.open_dataset(ds_path)

    cell = select_nearest_grid_cell(ds, target_lat, target_lon, var_name='precip')
    precip = cell['data'].values  # shape: (time, sample) or (sample, time)



    if precip.shape[0] != ref_precip.shape[0]:
        precip = precip.T



    precip_flat = precip.reshape(-1)
    precip_flat = np.where(precip_flat < 0, 0, precip_flat)
    precip_flat = precip_flat[~np.isnan(precip_flat)]
    pit = ecdf_ref(precip_flat)

  
    bin_edges = np.linspace(0, 1, bins + 1)
    ref_hist, _ = np.histogram(ref_pit, bins=bin_edges, density=True)
    pit_hist, _ = np.histogram(pit, bins=bin_edges, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = (bin_edges[1] - bin_edges[0]) * 0.4


    fig, ax = plt.subplots(figsize=(12, 6), dpi=1000)
    ax.bar(bin_centers - width/2, ref_hist, width=width, label="Test Set Targets(2011-2023)", color='tab:blue', alpha=0.8, edgecolor='black')
    ax.bar(bin_centers + width/2, pit_hist, width=width, label=f"Diffusion (eta={eta})", color='tab:orange', alpha=0.8, edgecolor='black')
    ax.axhline(1, color='red', linestyle='--', linewidth=1, label='Uniform Density')
    ax.set_xlabel("Binned Cumulative Probability", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.set_title(f"(Evaluation) PIT Histogram (eta={eta}) for {city} lat={target_lat:.3f}, lon={target_lon:.3f}", fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=13)
    plt.tight_layout()
    plt.savefig(f"DDIM_conditional_derived/output_inference/Precip_PIT_histogram_50steps_5samples_eta_{eta}_lat_{target_lat:.3f}_lon_{target_lon:.3f}.pdf", format='pdf', dpi=1000)
    plt.close()