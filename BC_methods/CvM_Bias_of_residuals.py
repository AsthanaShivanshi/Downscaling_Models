import argparse
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from scipy.stats import cramervonmises_2samp
import config

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

parser = argparse.ArgumentParser()
parser.add_argument("--city", type=str, required=True, help="City name, first letter uppercase")
parser.add_argument("--lat", type=float, required=True, help="City latitude")
parser.add_argument("--lon", type=float, required=True, help="City longitude")
args = parser.parse_args()

target_city = args.city
target_lat = args.lat
target_lon = args.lon

calib_start = "1981-01-01"
calib_end = "2010-12-31"
methods = ["QDM", "EQM", "DOTC"]
var_names = ["temp", "precip", "tmin", "tmax"]
obs_var_names = ["TabsD", "RhiresD", "TminD", "TmaxD"]

# For singlecell BC files
bc_files = {
    "QDM": f"{config.OUTPUTS_MODELS_DIR}/QDM_{target_city}_4vars_corrected.nc",
    "EQM": f"{config.OUTPUTS_MODELS_DIR}/EQM_{target_city}_4vars_corrected.nc",
    "DOTC": f"{config.OUTPUTS_MODELS_DIR}/DOTC_{target_city}_4vars_corrected.nc"
}
obs_files = [
    f"{config.DATASETS_TRAINING_DIR}/TabsD_step2_coarse.nc",
    f"{config.DATASETS_TRAINING_DIR}/RhiresD_step2_coarse.nc",
    f"{config.DATASETS_TRAINING_DIR}/TminD_step2_coarse.nc",
    f"{config.DATASETS_TRAINING_DIR}/TmaxD_step2_coarse.nc"
]

# closest grid
obs_datasets = [xr.open_dataset(p)[vn] for p, vn in zip(obs_files, obs_var_names)]
lat_vals = obs_datasets[0]['lat'].values
lon_vals = obs_datasets[0]['lon'].values
dist = np.sqrt((lat_vals - target_lat)**2 + (lon_vals - target_lon)**2)
i_city, j_city = np.unravel_index(np.argmin(dist), dist.shape)

#Calibration : 1981-2010
obs_calib = [ds.sel(time=slice(calib_start, calib_end))[:, i_city, j_city].values for ds in obs_datasets]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for v_idx, var in enumerate(var_names):
    ax = axes[v_idx]
    obs_vals = obs_calib[v_idx]
    obs_vals = obs_vals[~np.isnan(obs_vals)]
    for method in methods:
        ds_bc = xr.open_dataset(bc_files[method])
        bc_vals = ds_bc[var].sel(time=slice(calib_start, calib_end))[:, 0, 0].values
        bc_vals = bc_vals[~np.isnan(bc_vals)]

        all_vals= np.sort(np.unique(np.concatenate((obs_vals, bc_vals))))

        #CDFs
        obs_cdf = np.searchsorted(np.sort(obs_vals), all_vals, side='right') / len(obs_vals)
        bc_cdf = np.searchsorted(np.sort(bc_vals), all_vals, side='right') / len(bc_vals)
        
        # Difference
        cdf_diff = bc_cdf - obs_cdf
        
        # CvM statistic
        cvm = cramervonmises_2samp(obs_vals, bc_vals)
        
        ax.plot(all_vals, cdf_diff, label=f"{method} (CvM={cvm.statistic:.3f})")

    ax.set_title(var)
    ax.set_xlabel("Value")
    ax.set_ylabel("Corrected Model CDF - Obs CDF (1981-2010)")
    ax.legend()
    ax.grid(True)

fig.suptitle(f"CDF Difference (Corrected model - Observations) for {target_city} ({target_lat:.2f}, {target_lon:.2f}) - Calibration Period")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"{config.OUTPUTS_MODELS_DIR}/Residual_CDFs_{target_city}_1981_2010.png", dpi=1000)
print(f"Saved plot to {config.OUTPUTS_MODELS_DIR}/Residual_CDFs_{target_city}_1981_2010.png")
