import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from SBCK import QM
import config

model_path = f"{config.SCRATCH_DIR}/tmax_r01_HR_masked.nc"
obs_path = f"{config.SCRATCH_DIR}/TmaxD_1971_2023.nc"
output_path = f"{config.BC_DIR}/qm_tmax_r01_singlecell_output.nc"
plot_path = f"{config.OUTPUTS_MODELS_DIR}/qm_correction_function_tmax_r01_zurich_window.png"

print("Loading data")
model_output = xr.open_dataset(model_path)["tmax"]
obs_output = xr.open_dataset(obs_path)["TmaxD"]
calib_obs = obs_output.sel(time=slice("1981-01-01", "2010-12-31"))
calib_mod = model_output.sel(time=slice("1981-01-01", "2010-12-31"))

lat_vals = model_output['lat'].values
lon_vals = model_output['lon'].values

target_lat = 47.3769
target_lon = 8.5417

dist = np.sqrt((lat_vals - target_lat)**2 + (lon_vals - target_lon)**2)
i_zurich, j_zurich = np.unravel_index(np.argmin(dist), dist.shape)
print(f"Closest grid cell to Zurich: i={i_zurich}, j={j_zurich}")
print(f"Location: lat={lat_vals[i_zurich, j_zurich]}, lon={lon_vals[i_zurich, j_zurich]}")

calib_times = calib_mod['time'].values
calib_doys = xr.DataArray(calib_times).dt.dayofyear.values

model_times = model_output['time'].values
model_doys = xr.DataArray(model_times).dt.dayofyear.values
valid_mask = (model_times >= np.datetime64("1981-01-01")) & (model_times <= np.datetime64("2010-12-31"))
qm_series = np.full(model_output.shape[0], np.nan, dtype=np.float32)
correction_functions = {}

print("Fitting EQM for Zurich with 91-day moving window")
for doy in range(1, 367):  # 1 to 366
    window_doys = ((calib_doys - doy + 366) % 366)
    window_mask = (window_doys <= 45) | (window_doys >= (366 - 45))
    obs_window = calib_obs[:, i_zurich, j_zurich].values[window_mask]
    mod_window = calib_mod[:, i_zurich, j_zurich].values[window_mask]
    obs_window = obs_window[~np.isnan(obs_window)]
    mod_window = mod_window[~np.isnan(mod_window)]
    if obs_window.size == 0 or mod_window.size == 0:
        continue
    eqm = QM()
    eqm.fit(mod_window.reshape(-1, 1), obs_window.reshape(-1, 1))
    quantiles = np.linspace(0.01, 0.99, 99)
    obs_q = np.quantile(obs_window, quantiles)
    mod_q = np.quantile(mod_window, quantiles)
    # End quantiles 
    obs_q = np.concatenate([[obs_q[0]], obs_q, [obs_q[-1]]])
    mod_q = np.concatenate([[mod_q[0]], mod_q, [mod_q[-1]]])
    correction = mod_q - obs_q
    ext_q = np.linspace(0, 1, 101)
    correction_functions[doy] = (ext_q, correction)
    indices = np.where((model_doys == doy) & valid_mask)[0]
    for idx in indices:
        value = model_output[idx, i_zurich, j_zurich]
        qm_series[idx] = eqm.predict(np.array([[value]])).flatten()[0]

qm_data = np.full(model_output.shape, np.nan, dtype=np.float32)
qm_data[:, i_zurich, j_zurich] = qm_series.astype(np.float32)

qm_ds = xr.Dataset(
    {"tmax": (model_output.dims, qm_data)},
    coords=model_output.coords
)
qm_ds.to_netcdf(output_path)
print(f"Single-cell output saved to {output_path}")

# Plot: seasonal correction functions
season_doys = {
    "DJF": list(range(335, 367)) + list(range(1, 60)),
    "MAM": list(range(60, 152)),
    "JJA": list(range(152, 244)),
    "SON": list(range(244, 335)),
}

plt.figure(figsize=(8, 6))
ax1 = plt.gca()
for season, doys in season_doys.items():
    season_corrs = []
    for doy in doys:
        if doy in correction_functions:
            ext_q, ext_corr = correction_functions[doy]
            season_corrs.append(ext_corr)
    if season_corrs:
        mean_corr = np.mean(season_corrs, axis=0)
        ax1.plot(ext_q, mean_corr, label=season)

ax1.axhline(0, color="gray", linestyle="--")
ax1.set_xlabel("Quantile")
ax1.set_ylabel("Correction (Model - Observation) in degrees C")
ax1.set_title("Seasonal Mean Correction Functions for Zürich")
ax1.legend()
ax1.grid(True)
plt.tight_layout()
plt.savefig(plot_path, dpi=1000)
print(f"Correction function plot saved to {plot_path}")