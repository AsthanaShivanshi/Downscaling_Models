import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from SBCK import QM
import config
import argparse

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 18,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

parser= argparse.ArgumentParser()
parser.add_argument("--city", type=str, required=True, help="City name with first letter in Upper Case")
parser.add_argument("--lat", type=float, required=True, help="City lat")
parser.add_argument("--lon", type=float, required=True, help="City lon")
args = parser.parse_args()

target_city = args.city
target_lat = args.lat
target_lon = args.lon

locations= {target_city: (target_lat, target_lon)}

model_path = f"{config.MODELS_DIR}/temp_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099/temp_r01_coarse_masked.nc" 
obs_path = f"{config.DATASETS_TRAINING_DIR}/TabsD_step2_coarse.nc"
output_path_template = f"{config.BC_DIR}/qm_temp_r01_singlecell_{{city}}_output.nc"
plot_path = f"{config.OUTPUTS_MODELS_DIR}/qm_correction_function_temp_r01_2cities_DJF.png"

print("Loading data")
model_output = xr.open_dataset(model_path)["temp"]
obs_output = xr.open_dataset(obs_path)["TabsD"]

#Control period
calib_obs = obs_output.sel(time=slice("1981-01-01", "2010-12-31"))
calib_mod = model_output.sel(time=slice("1981-01-01", "2010-12-31"))

lat_vals = model_output['lat'].values
lon_vals = model_output['lon'].values

calib_times = calib_mod['time'].values
calib_doys = xr.DataArray(calib_times).dt.dayofyear.values

def get_season(doy):
    if doy >= 335 or doy <= 59:
        return "DJF"
    elif 60 <= doy <= 151:
        return "MAM"
    elif 152 <= doy <= 243:
        return "JJA"
    elif 244 <= doy <= 334:
        return "SON"
    else:
        return None

season_linestyles = {
    "DJF": "-",
    "MAM": "--",
    "JJA": "-.",
    "SON": ":"
}

fig, ax = plt.subplots(figsize=(10, 7))

season_colors = {
    "DJF": "b",
    "MAM": "g",
    "JJA": "r",
    "SON": "orange"
}

fig, ax = plt.subplots(figsize=(10, 7))

print(f"\nProcessing {target_city}...")
dist = np.sqrt((lat_vals - target_lat)**2 + (lon_vals - target_lon)**2)
i_city, j_city = np.unravel_index(np.argmin(dist), dist.shape)
print(f"Closest grid cell to {target_city}: i={i_city}, j={j_city}")
print(f"Location: lat={lat_vals[i_city, j_city]}, lon={lon_vals[i_city, j_city]}")

for season in ["DJF", "MAM", "JJA", "SON"]:
    season_corrections = []
    for doy in range(1, 367):
        if get_season(doy) != season:
            continue
        window_doys = ((calib_doys - doy + 366) % 366)
        window_mask = (window_doys <= 45) | (window_doys >= (366 - 45))
        obs_window = calib_obs[:, i_city, j_city].values[window_mask]
        mod_window = calib_mod[:, i_city, j_city].values[window_mask]
        obs_window = obs_window[~np.isnan(obs_window)]
        mod_window = mod_window[~np.isnan(mod_window)]
        if obs_window.size == 0 or mod_window.size == 0:
            continue
        quantiles = np.linspace(0.01, 0.99, 99)
        obs_q = np.quantile(obs_window, quantiles)
        mod_q = np.quantile(mod_window, quantiles)
        obs_q = np.concatenate([[obs_q[0]], obs_q, [obs_q[-1]]])
        mod_q = np.concatenate([[mod_q[0]], mod_q, [mod_q[-1]]])
        correction = mod_q - obs_q
        season_corrections.append(correction)

    if season_corrections:
        mean_corr = np.mean(season_corrections, axis=0)
        ext_q = np.linspace(0, 1, 101)
        ax.plot(
            ext_q, mean_corr,
            label=f"{target_city} {season}",
            color=season_colors[season]
        )

ax.axhline(0, color="gray", linestyle="--")
ax.set_xlabel("Quantile")
ax.set_ylabel("Mean Correction: seasonwise")
ax.set_title(f"Seasonal Correction Fx of Daily Temperature for {target_city} at 12 kms resolution")
ax.legend(loc="lower left")
ax.grid(True)
fig.tight_layout()
plt.savefig(plot_path, dpi=1000)
print(f"Seasonal correction function plot saved to {plot_path}")