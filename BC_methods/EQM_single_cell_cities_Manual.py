import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from SBCK import QM
import config
import argparse
from scipy.interpolate import interp1d

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

parser= argparse.ArgumentParser()
parser.add_argument("--city", type=str, required=True, help="City name,,first letter upper case")
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
plot_path = f"{config.OUTPUTS_MODELS_DIR}/qm_corr_fx_temp_allseasons_{target_city}.png"

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

print(f"\nProcessing {target_city}...")
dist = np.sqrt((lat_vals - target_lat)**2 + (lon_vals - target_lon)**2)
i_city, j_city = np.unravel_index(np.argmin(dist), dist.shape)
print(f"Closest grid cell to {target_city}: i={i_city}, j={j_city}")
print(f"Location: lat={lat_vals[i_city, j_city]}, lon={lon_vals[i_city, j_city]}")


# Corr fx for each doy using 91d
doy_corrections = []
doy_seasons = []
quantiles_inner = np.linspace(0.01, 0.99, 99)
model_cell = calib_mod[:, i_city, j_city].values
obs_cell = calib_obs[:, i_city, j_city].values


for doy in range(1, 367):
    window_doys = ((calib_doys - doy + 366) % 366)
    window_mask = (window_doys <= 45) | (window_doys >= (366 - 45))
    obs_window = obs_cell[window_mask]
    mod_window = model_cell[window_mask]
    obs_window = obs_window[~np.isnan(obs_window)]
    mod_window = mod_window[~np.isnan(mod_window)]
    if obs_window.size == 0 or mod_window.size == 0:
        doy_corrections.append(np.full(101, np.nan))
        doy_seasons.append(get_season(doy))
        continue
    eqm = QM()
    eqm.fit(mod_window.reshape(-1, 1), obs_window.reshape(-1, 1))
    quantiles_inner = np.linspace(0.01, 0.99, 99)
    mod_q_inner = np.quantile(mod_window, quantiles_inner)
    obs_q_inner = np.quantile(obs_window, quantiles_inner)
    # Interpolation for tails
    correction_inner = obs_q_inner - mod_q_inner
    interp_corr = interp1d(
        quantiles_inner, correction_inner, kind='linear', fill_value='extrapolate'
    )
    quantiles = np.linspace(0, 1, 101)
    correction = interp_corr(quantiles)
    doy_corrections.append(correction)
    doy_seasons.append(get_season(doy))

doy_corrections = np.array(doy_corrections)  #(366, 101)
doy_seasons = np.array(doy_seasons)
quantiles = np.linspace(0, 1, 101)
for season in ["DJF", "MAM", "JJA", "SON"]:
    season_mask = (doy_seasons == season)
    if np.any(season_mask):
        mean_correction = np.nanmean(doy_corrections[season_mask], axis=0)
        ax.plot(
            quantiles, mean_correction,
            label=season,
            color=season_colors[season],
            linestyle=season_linestyles[season]
        )

ax.axhline(0, color="gray", linestyle="--")
ax.set_xlabel("Quantile")
ax.set_ylabel("Mean Correction: seasonwise")
ax.set_title(f"Correction Fx of daily temperature: {target_city}")
ax.legend(loc="lower left")
ax.grid(True)
fig.tight_layout()
plt.savefig(plot_path, dpi=1000)
print(f"Correction function plot saved to {plot_path}")
