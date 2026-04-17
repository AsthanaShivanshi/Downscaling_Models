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

ref_path = "Dataset_Setup_I_Chronological_12km/TabsD_step1_latlon.nc"

downscaled_paths = [
    f"DDIM_conditional_derived/output_inference/ddim_downscaled_50steps_test_set_5samples_eta_{eta}.nc"
    for eta in etas
]

#-----------------------------------------------------------------------#

ref_ds = xr.open_dataset(ref_path).sel(time=slice("2011-01-01", "2023-12-31"))
ref_cell = select_nearest_grid_cell(ref_ds, target_lat, target_lon, var_name='TabsD')
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

    cell = select_nearest_grid_cell(ds, target_lat, target_lon, var_name='temp')
    temp = cell['data'].values  # shape: (time, sample) or (sample, time)



    if temp.shape[0] != ref_precip.shape[0]:
        temp = temp.T



    temp_flat = temp.reshape(-1)
    temp_flat = temp_flat[~np.isnan(temp_flat)]
    pit = ecdf_ref(temp_flat)

  

    bin_edges = np.linspace(0, 1, bins + 1)
    ref_hist, _ = np.histogram(ref_pit, bins=bin_edges, density=True)
    pit_hist, _ = np.histogram(pit, bins=bin_edges, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = (bin_edges[1] - bin_edges[0]) * 0.4

#-----------------------------------------------------------------------#

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.bar(bin_centers - width/2, ref_hist, width=width, label="Reference (2011–2023)", color='#4575b4', alpha=0.85, edgecolor='black')
    ax.bar(bin_centers + width/2, pit_hist, width=width, label=f"Diffusion (η={eta})", color='#d73027', alpha=0.85, edgecolor='black')
    ax.plot([0, 1], [1, 1], color='gray', linestyle='-', linewidth=2, label='Uniform Density (Ideal)')
    ax.set_xlabel("Probability Integral Transform (PIT)", fontsize=15)
    ax.set_ylabel("Density", fontsize=15)
    ax.set_title(f"PIT Histogram for Temperature\n{city} (lat={target_lat:.3f}, lon={target_lon:.3f})", fontsize=17, pad=15)
    ax.annotate(f"Grid cell: lat={target_lat:.3f}, lon={target_lon:.3f}\nTime steps: {ref_precip_flat.shape[0]}\nSamples: {temp.shape[1] if temp.ndim > 1 else 1}",
                xy=(0.99, 0.97), xycoords='axes fraction', ha='right', va='top', fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))



    ax.grid(True, linestyle=':', linewidth=0.7, alpha=0.7)
    ax.legend(fontsize=13, frameon=True)
    plt.tight_layout()
    plt.savefig(f"DDIM_conditional_derived/Metrics_Test_Set/outputs/Temp_PIT_histogram_50steps_5samples_eta_{eta}_lat_{target_lat:.3f}_lon_{target_lon:.3f}.pdf", format='pdf', dpi=1000)

    plt.close()
