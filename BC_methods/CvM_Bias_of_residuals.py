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


bc_files = {
    "QDM": f"{config.OUTPUTS_MODELS_DIR}/QDM_{target_city}_4vars_corrected.nc",
    "EQM": f"{config.OUTPUTS_MODELS_DIR}/EQM_{target_city}_4vars_corrected.nc",
    "DOTC": f"{config.OUTPUTS_MODELS_DIR}/DOTC_{target_city}_4vars_corrected.nc"
}
obs_files = [
    f"/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/data/TabsD_step2_coarse.nc",
    f"/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling_Models/data/RhiresD_step2_coarse.nc",
    f"/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling_Models/data/TminD_step2_coarse.nc",
    f"/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling_Models/data/TmaxD_step2_coarse.nc"
]

# --- Load obs and get closest grid cell ---
obs_datasets = [xr.open_dataset(p)[vn] for p, vn in zip(obs_files, obs_var_names)]
lat_vals = obs_datasets[0]['lat'].values
lon_vals = obs_datasets[0]['lon'].values
dist = np.sqrt((lat_vals - lat)**2 + (lon_vals - lon)**2)
i_city, j_city = np.unravel_index(np.argmin(dist), dist.shape)

# --- Load obs calibration data ---
obs_calib = [ds.sel(time=slice(calib_start, calib_end))[:, i_city, j_city].values for ds in obs_datasets]

# --- Prepare plot ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for v_idx, var in enumerate(var_names):
    ax = axes[v_idx]
    obs_vals = obs_calib[v_idx]
    obs_vals = obs_vals[~np.isnan(obs_vals)]
    for method in methods:
        # Load BC output
        ds_bc = xr.open_dataset(bc_files[method])
        bc_vals = ds_bc[var].sel(time=slice(calib_start, calib_end))[:, 0, 0].values
        bc_vals = bc_vals[~np.isnan(bc_vals)]
        # Residuals
        residuals = obs_vals - bc_vals
        # CDF
        sorted_res = np.sort(residuals)
        cdf = np.arange(1, len(sorted_res)+1) / len(sorted_res)
        # CvM statistic
        cvm = cramervonmises_2samp(obs_vals, bc_vals)
        ax.plot(sorted_res, cdf, label=f"{method} (CvM={cvm.statistic:.3f})")
    ax.set_title(var)
    ax.set_xlabel("Residual (Obs - BC output)")
    ax.set_ylabel("CDF")
    ax.legend()
    ax.grid(True)

fig.suptitle(f"Residual CDFs for {city} ({lat:.2f}, {lon:.2f}) - Calibration Period")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"{outputs_dir}/Residual_CDFs_{city}_calib.png", dpi=300)
print(f"Saved plot to {outputs_dir}/Residual_CDFs_{city}_calib.png")
