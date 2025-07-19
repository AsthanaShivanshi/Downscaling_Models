import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import config
from SBCK import QDM

model_path = f"{config.SCRATCH_DIR}/temp_r01_HR_masked.nc"
obs_path = f"{config.SCRATCH_DIR}/TabsD_1971_2023.nc"
output_path = f"{config.BC_DIR}/qdm_temp_r01_singlecell_output.nc"

print("Loading data")
model_output = xr.open_dataset(model_path)["temp"]
obs_output = xr.open_dataset(obs_path)["TabsD"]
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

obs_valid = calib_obs[:, i_zurich, j_zurich].values[~np.isnan(calib_obs[:, i_zurich, j_zurich].values)]
mod_valid = calib_mod[:, i_zurich, j_zurich].values[~np.isnan(calib_mod[:, i_zurich, j_zurich].values)]
if obs_valid.size == 0 or mod_valid.size == 0:
    print("No valid data for Zurich grid cell. Exiting.")
    exit(1)

print("Fitting QDM for Zurich (additive for temperature, 45-day window)")
qdm = QDM(window_width=45, kind='additive')
qdm.fit(
    obs_valid.reshape(-1, 1),           # Y0: obs calibration for 1981-2010
    mod_valid.reshape(-1, 1),           # X0: model calibration for 1981-2010
    model_output[:, i_zurich, j_zurich].values.reshape(-1, 1)  # X1: projection (full series)
)

full_mod_series = model_output[:, i_zurich, j_zurich].values.reshape(-1, 1)
qdm_series = qdm.predict(full_mod_series).flatten()

qm_data = np.full(model_output.shape, np.nan, dtype=np.float32)
qm_data[:, i_zurich, j_zurich] = qdm_series.astype(np.float32)

qm_ds = xr.Dataset(
    {"temp": (model_output.dims, qm_data)},
    coords=model_output.coords
)
qm_ds.to_netcdf(output_path)
print(f"Single-cell QDM output saved to {output_path}")

#Correction fx plot: percentiles vs (model - obs)
quantiles = np.linspace(0, 1, 1001)
plot_obs_q = np.quantile(obs_valid, quantiles)
plot_mod_q = np.quantile(mod_valid, quantiles)
correction = plot_mod_q - plot_obs_q
lat_val = lat_vals[i_zurich, j_zurich]
lon_val = lon_vals[i_zurich, j_zurich]

plt.figure(figsize=(7, 5))
plt.plot(quantiles * 100, correction, label="Correction (model - obs)")
plt.axhline(0, color="gray", linestyle="--", label="No correction")
plt.xlabel("Percentile")
plt.ylabel("Correction (Model - Observation) in degrees C")
plt.title(f"QDM Correction Function for Daily Avg Temp\nZurich (lat={lat_val:.3f}, lon={lon_val:.3f})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{config.OUTPUTS_MODELS_DIR}/qdm_correction_function_temp_r01_zurich.png", dpi=500)
print("Correction function plot saved.")

# EXCDF
plt.figure(figsize=(7, 5))
obs_sorted = np.sort(obs_valid)
mod_sorted = np.sort(mod_valid)
obs_cdf = np.arange(1, len(obs_sorted)+1) / len(obs_sorted)
mod_cdf = np.arange(1, len(mod_sorted)+1) / len(mod_sorted)
plt.plot(obs_sorted, obs_cdf, label="Obs empirical CDF")
plt.plot(mod_sorted, mod_cdf, label="Model empirical CDF")
plt.xlabel("Value")
plt.ylabel("Cumulative Probability")
plt.title(f"Empirical CDFs for Daily Avg Temp\nZurich (lat={lat_val:.3f}, lon={lon_val:.3f})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{config.OUTPUTS_MODELS_DIR}/qdm_cdf_temp_r01_zurich.png", dpi=500)
print("CDF plot saved.")

print("QDM Zurich validation completed successfully.")