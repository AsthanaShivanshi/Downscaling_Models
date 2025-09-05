import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from SBCK import QM
import config
import argparse
from scipy.interpolate import interp1d
import scipy.stats

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

    #NaN window: moving on
    if obs_window.size == 0 or mod_window.size == 0:
        doy_corrections.append(np.full(101, np.nan))
        doy_seasons.append(get_season(doy))
        continue

    # QM on inner quantiles
    quantiles_inner = np.linspace(0.01, 0.99, 99)
    mod_q_inner = np.quantile(mod_window, quantiles_inner)
    obs_q_inner = np.quantile(obs_window, quantiles_inner)
    eqm = QM()
    eqm.fit(mod_q_inner.reshape(-1, 1), obs_q_inner.reshape(-1, 1))

    #Correction using SBCK for inner quantiles
    correction_inner = obs_q_inner - mod_q_inner

#Interp between quant res and at the ends for final corr fx
    interp_corr = interp1d(
        quantiles_inner, correction_inner, kind='linear', fill_value='extrapolate'
    )
    quantiles= np.linspace(0, 1, 101)
    correction = interp_corr(quantiles)
    doy_corrections.append(correction)
    doy_seasons.append(get_season(doy))

doy_corrections = np.array(doy_corrections)  #(366, 101)
full_model_cell = model_output[:, i_city, j_city].values
full_times = model_output['time'].values
full_doys = xr.DataArray(full_times).dt.dayofyear.values

corrected_cell = np.full_like(full_model_cell, np.nan)

for idx, (val, doy) in enumerate(zip(full_model_cell, full_doys)):
    correction = doy_corrections[doy-1]  
    mod_window = model_cell[((calib_doys - doy + 366) % 366 <= 45) | ((calib_doys - doy + 366) % 366 >= (366 - 45))]
    mod_window = mod_window[~np.isnan(mod_window)]
    if mod_window.size == 0 or np.isnan(val):
        continue
    mod_q = np.quantile(mod_window, np.linspace(0, 1, 101))
    value_quantile = np.searchsorted(mod_q, val, side='right') / 100.0
    value_quantile = np.clip(value_quantile, 0, 1)
    interp_corr = interp1d(np.linspace(0, 1, 101), correction, kind='linear', fill_value='extrapolate')
    corrected_cell[idx] = val + interp_corr(value_quantile)
doy_seasons = np.array(doy_seasons)

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
ax.set_xlabel("Quantile Level")
ax.set_ylabel("Seasonal mean of Correction Fx (°C)")
ax.set_title(f"Correction Fx of daily temperature with EQM BC: {target_city}")
ax.legend(loc="lower left")
ax.grid(True)
fig.tight_layout()
plt.savefig(plot_path, dpi=1000)
print(f"Correction function plot saved to {plot_path}")

# CDFs from calibration period
calib_start = "1981-01-01"
calib_end = "2010-12-31"
model_vals = model_output.sel(time=slice(calib_start, calib_end))[:, i_city, j_city].values
obs_vals = obs_output.sel(time=slice(calib_start, calib_end))[:, i_city, j_city].values
model_vals = model_vals[~np.isnan(model_vals)]
obs_vals = obs_vals[~np.isnan(obs_vals)]
corr_vals = corrected_cell[(full_times >= np.datetime64(calib_start)) & (full_times <= np.datetime64(calib_end))]
corr_vals = corr_vals[~np.isnan(corr_vals)]

emd_model = scipy.stats.wasserstein_distance(obs_vals, model_vals)
emd_corr = scipy.stats.wasserstein_distance(obs_vals, corr_vals)

plt.figure(figsize=(8, 6))

for vals, label, color in [
    (model_vals, f"Model (Coarse) [Wasserstein={emd_model:.3f}]", "blue"),
    (obs_vals, "Observations", "green"),
    (corr_vals, f"Corrected Output [Wasserstein={emd_corr:.3f}]", "red")
]:
    sorted_vals = np.sort(vals)
    cdf = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)
    plt.plot(sorted_vals, cdf, label=label, color=color)

plt.xlabel("Mean Temperature (°C)")
plt.ylabel("CDF")
plt.title(f"CDFs for {target_city}: EQM BC")
plt.legend()
plt.grid(True)
plt.tight_layout()
cdf_plot_path = plot_path.replace("corr_fx_temp_allseasons", "cdf_temp_singlecell")
plt.savefig(cdf_plot_path, dpi=1000)
print(f"CDF plot saved to {cdf_plot_path}")
