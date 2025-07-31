import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import config
import pandas as pd
from SBCK import QM

model_path = f"{config.SCRATCH_DIR}/temp_r01_HR_masked.nc"
obs_path = f"{config.SCRATCH_DIR}/TabsD_1971_2023.nc"
output_path = f"{config.BC_DIR}/qm_temp_r01_singlecell_output.nc"
plot_path = f"{config.OUTPUTS_MODELS_DIR}/qm_correction_function_temp_r01_zurich.png"

print("Loading data")
model_output = xr.open_dataset(model_path)["temp"]
obs_output = xr.open_dataset(obs_path)["TabsD"]

# Padding for calibration
buffer_start = pd.to_datetime("1981-01-01") - pd.Timedelta(days=45)
buffer_end = pd.to_datetime("2010-12-31") + pd.Timedelta(days=45)
calib_obs_ext = obs_output.sel(time=slice(buffer_start, buffer_end))
calib_mod_ext = model_output.sel(time=slice(buffer_start, buffer_end))

# Aligning time
common_times = np.intersect1d(calib_obs_ext['time'].values, calib_mod_ext['time'].values)
calib_obs_ext = calib_obs_ext.sel(time=common_times)
calib_mod_ext = calib_mod_ext.sel(time=common_times)
calib_times_ext = pd.to_datetime(common_times)
print("calib_obs_ext shape:", calib_obs_ext.shape)
print("calib_mod_ext shape:", calib_mod_ext.shape)
print("calib_times_ext shape:", calib_times_ext.shape)

lat_vals = model_output['lat'].values
lon_vals = model_output['lon'].values

target_lat = 47.3769
target_lon = 8.5417

dist = np.sqrt((lat_vals - target_lat)**2 + (lon_vals - target_lon)**2)
i_zurich, j_zurich = np.unravel_index(np.argmin(dist), dist.shape)
print(f"Closest grid cell to Zurich: i={i_zurich}, j={j_zurich}")
print(f"Location: lat={lat_vals[i_zurich, j_zurich]}, lon={lon_vals[i_zurich, j_zurich]}")

model_times = pd.to_datetime(model_output['time'].values)
model_doys = model_times.dayofyear
valid_mask = (model_times >= "1981-01-01") & (model_times <= "2010-12-31")
qm_series = np.full(model_output.shape[0], np.nan, dtype=np.float32)
correction_functions = {}

for doy in range(1, 367):  # 1 to 366
    calib_doys_ext = calib_times_ext.dayofyear
    window_doys = ((calib_doys_ext - doy + 366) % 366)
    window_mask = (window_doys <= 45) | (window_doys >= (366 - 45))
    if calib_obs_ext.shape[0] != window_mask.shape[0]:
        print(f"Warning: window_mask shape {window_mask.shape} does not match time axis {calib_obs_ext.shape[0]}")
        continue
    obs_window = calib_obs_ext[:, i_zurich, j_zurich].values[window_mask]
    mod_window = calib_mod_ext[:, i_zurich, j_zurich].values[window_mask]
    obs_window = obs_window[~np.isnan(obs_window)]
    mod_window = mod_window[~np.isnan(mod_window)]
    if obs_window.size == 0 or mod_window.size == 0:
        continue
    eqm = QM()
    eqm.fit(mod_window.reshape(-1, 1), obs_window.reshape(-1, 1))
    quantiles = np.linspace(0.01, 0.99, 99)
    obs_q = np.quantile(obs_window, quantiles)
    mod_q = np.quantile(mod_window, quantiles)
    correction = mod_q - obs_q
    ext_q = np.concatenate([[0.0], quantiles, [1.0]])
    ext_corr = np.concatenate(([correction[0]], correction, [correction[-1]]))
    correction_functions[doy] = (ext_q, ext_corr)
    indices = np.where((model_doys == doy) & valid_mask)[0]
    for idx in indices:
        value = model_output[idx, i_zurich, j_zurich]
        qm_series[idx] = eqm.predict(np.array([[value]])).flatten()[0]

qm_data = np.full(model_output.shape, np.nan, dtype=np.float32)
qm_data[:, i_zurich, j_zurich] = qm_series.astype(np.float32)
qm_ds = xr.Dataset(
    {"temp": (model_output.dims, qm_data)},
    coords=model_output.coords
)
qm_ds.to_netcdf(output_path)
print(f"Single-cell output saved to {output_path}")

print("Number of DOYs with correction:", len(correction_functions))
print("DOYs with correction:", list(correction_functions.keys()))

# Plot: seasonal corr fx 
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
ax1.set_title("Seasonal Correction Functions for Zürich")
ax1.legend()
ax1.grid(True)
plt.tight_layout()
plt.savefig(plot_path.replace(".png", "_seasons.png"), dpi=1000)
print(f"Saved to {plot_path.replace('.png', '_seasons.png')}")