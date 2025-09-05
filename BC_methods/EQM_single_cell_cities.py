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

model_path = f"{config.MODELS_DIR}/precip_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099/precip_r01_coarse_masked.nc" 
obs_path = f"{config.DATASETS_TRAINING_DIR}/RhiresD_step2_coarse.nc"
output_path_template = f"{config.BC_DIR}/qm_precip_r01_singlecell_{{city}}_output.nc"
plot_path = f"{config.OUTPUTS_MODELS_DIR}/qm_corr_fx_precip_allseasons_{target_city}.png"

print("Loading data")
model_output = xr.open_dataset(model_path)["precip"]
obs_output = xr.open_dataset(obs_path)["RhiresD"]

#Control 
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

    # QM inner quantiles
    quantiles_inner = np.linspace(0.01, 0.99, 99)
    mod_q_inner = np.quantile(mod_window, quantiles_inner)
    obs_q_inner = np.quantile(obs_window, quantiles_inner)
    eqm = QM()
    eqm.fit(mod_q_inner.reshape(-1, 1), obs_q_inner.reshape(-1, 1))

    #Correction using SBCK for inner quantiles
    correction_inner = (eqm.predict(mod_q_inner.reshape(-1, 1)).flatten()) - mod_q_inner

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
    mod_window = model_cell[((calib_doys - doy + 366) % 366 <= 45) | ((calib_doys - doy + 366) % 366 >= (366 - 45))]
    obs_window = obs_cell[((calib_doys - doy + 366) % 366 <= 45) | ((calib_doys - doy + 366) % 366 >= (366 - 45))]
    mod_window = mod_window[~np.isnan(mod_window)]
    obs_window = obs_window[~np.isnan(obs_window)]
    if mod_window.size == 0 or obs_window.size == 0 or np.isnan(val):
        continue
    mod_q = np.quantile(mod_window, np.linspace(0, 1, 101))
    obs_q = np.quantile(obs_window, np.linspace(0, 1, 101))
    value_quantile = np.searchsorted(mod_q, val, side='right') / 100.0
    value_quantile = np.clip(value_quantile, 0, 1)
    correction_fx = doy_corrections[doy - 1]  # doy runs from 1 to 366, array index from 0
    interp_corr = interp1d(np.linspace(0, 1, 101), correction_fx, kind='linear', fill_value='extrapolate')
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
ax.set_ylabel("Seasonal mean of Correction Fx (mm/day)")
ax.set_title(f"Correction Fx of daily precipitation with EQM BC: {target_city}")
ax.legend(loc="lower left")
ax.grid(True)
fig.tight_layout()
plt.savefig(plot_path, dpi=1000)
print(f"Correction function plot saved to {plot_path}")


# CDFs
calib_start = "1981-01-01"
calib_end = "2010-12-31"
scenario_start = "2011-01-01"
scenario_end = "2099-12-31"

model_vals_calib = model_output.sel(time=slice(calib_start, calib_end))[:, i_city, j_city].values
obs_vals_calib = obs_output.sel(time=slice(calib_start, calib_end))[:, i_city, j_city].values
model_vals_calib = model_vals_calib[~np.isnan(model_vals_calib)]
obs_vals_calib = obs_vals_calib[~np.isnan(obs_vals_calib)]
corr_vals_calib = corrected_cell[(full_times >= np.datetime64(calib_start)) & (full_times <= np.datetime64(calib_end))]
corr_vals_calib = corr_vals_calib[~np.isnan(corr_vals_calib)]

model_vals_scen = model_output.sel(time=slice(scenario_start, scenario_end))[:, i_city, j_city].values
model_vals_scen = model_vals_scen[~np.isnan(model_vals_scen)]
corr_vals_scen = corrected_cell[(full_times >= np.datetime64(scenario_start)) & (full_times <= np.datetime64(scenario_end))]
corr_vals_scen = corr_vals_scen[~np.isnan(corr_vals_scen)]

ks_model_calib = scipy.stats.ks_2samp(obs_vals_calib, model_vals_calib)
ks_corr_calib = scipy.stats.ks_2samp(obs_vals_calib, corr_vals_calib)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: Calib
for vals, label, color in [
    (model_vals_calib, f"Model (Coarse,1981-2010) [KS={ks_model_calib.statistic:.3f}]", "red"),
    (obs_vals_calib, "Observations (1981-2010)", "black"),
    (corr_vals_calib, f"Corrected Output (1981-2010) [KS={ks_corr_calib.statistic:.3f}]", "green")
]:
    sorted_vals = np.sort(vals)
    cdf = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)
    axes[0].plot(sorted_vals, cdf, label=label, color=color)

axes[0].set_xlabel("Precipitation (mm/day)")
axes[0].set_ylabel("CDF")
axes[0].set_title(f"CDFs (cal period : 1981-2010) for {target_city}: EQM BC")
axes[0].legend()
axes[0].grid(True)

# Right: Scenario 
for vals, label, color in [
    (model_vals_scen, f"Model (Coarse, 2011-2099)", "red"),
    (obs_vals_calib, "Observations (1981-2010)", "black"),
    (corr_vals_scen, f"Corrected Output (2011-2099)", "green")
]:
    sorted_vals = np.sort(vals)
    cdf = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)
    axes[1].plot(sorted_vals, cdf, label=label, color=color)

axes[1].set_xlabel("Precipitation (mm/day)")
axes[1].set_ylabel("CDF")
axes[1].set_title(f"CDFs (scenario period : 2011-2099) for {target_city}: EQM BC")
axes[1].legend()
axes[1].grid(True)

fig.tight_layout()
cdf_plot_path = plot_path.replace("corr_fx_precip_allseasons", "cdf_precip_singlecell_twopanel")
plt.savefig(cdf_plot_path, dpi=1000)
print(f"Two-panel CDF plot saved to {cdf_plot_path}")
