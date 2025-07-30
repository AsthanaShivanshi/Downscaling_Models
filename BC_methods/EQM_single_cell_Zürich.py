import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from SBCK import QM
import config

model_path = f"{config.SCRATCH_DIR}/precip_r01_HR_masked.nc"
obs_path = f"{config.SCRATCH_DIR}/RhiresD_1971_2023.nc"
output_path = f"{config.BC_DIR}/qm_precip_r01_singlecell_output.nc"
plot_path = f"{config.OUTPUTS_MODELS_DIR}/qm_correction_function_precip_r01_zürich.png"
cdf_plot_path = f"{config.OUTPUTS_MODELS_DIR}/qm_cdf_precip_r01_zürich.png"
map_plot_path = f"{config.OUTPUTS_MODELS_DIR}/qm_selected_gridcell_map_precip_r01_zürich.png"

print("Loading data")
model_output = xr.open_dataset(model_path)["precip"]
obs_output = xr.open_dataset(obs_path)["RhiresD"]
calib_obs = obs_output.sel(time=slice("1981-01-01", "2010-12-31"))
calib_mod = model_output.sel(time=slice("1981-01-01", "2010-12-31"))

lat_vals = model_output['lat'].values
lon_vals = model_output['lon'].values

print("model_output shape:", model_output.shape)
print("lat_vals shape:", lat_vals.shape)
print("lon_vals shape:", lon_vals.shape)

target_lat = 46.2044
target_lon = 6.1432
dist = np.sqrt((lat_vals - target_lat)**2 + (lon_vals - target_lon)**2)
i_zurich, j_zurich = np.unravel_index(np.argmin(dist), dist.shape)
print(f"Closest grid cell to Zurich: i={i_zurich}, j={j_zurich}")
print(f"Location: lat={lat_vals[i_zurich, j_zurich]}, lon={lon_vals[i_zurich, j_zurich]}")

obs_valid = calib_obs[:, i_zurich, j_zurich].values[~np.isnan(calib_obs[:, i_zurich, j_zurich].values)]
mod_valid = calib_mod[:, i_zurich, j_zurich].values[~np.isnan(calib_mod[:, i_zurich, j_zurich].values)]
if obs_valid.size == 0 or mod_valid.size == 0:
    print("No valid data for Zurich grid cell. Exiting.")
    exit(1)

print("Fitting EQM for Zurich")
eqm = QM()
eqm.fit(mod_valid.reshape(-1, 1), obs_valid.reshape(-1, 1))

full_mod_series = model_output[:, i_zurich, j_zurich].values.reshape(-1, 1)
qm_series = eqm.predict(full_mod_series).flatten()

qm_data = np.full(model_output.shape, np.nan, dtype=np.float32)
qm_data[:, i_zurich, j_zurich] = qm_series.astype(np.float32)

qm_ds = xr.Dataset(
    {"precip": (model_output.dims, qm_data)},
    coords=model_output.coords
)
qm_ds.to_netcdf(output_path)
print(f"Single-cell output saved to {output_path}")

# Correction fx with quantiles : (model - obs)
quantiles = np.linspace(0, 1, 1001)
plot_obs_q = np.quantile(obs_valid, quantiles)
plot_mod_q = np.quantile(mod_valid, quantiles)
correction = plot_mod_q - plot_obs_q
lat_val = lat_vals[i_zurich, j_zurich]
lon_val = lon_vals[i_zurich, j_zurich]

# Setting correction for quantiles >= 0.95 to the value at 0.95
murdered_correction = correction.copy()
tail_start_idx = np.where(quantiles >= 0.95)[0][0]
murdered_correction[tail_start_idx:] = correction[tail_start_idx]

plt.figure(figsize=(7, 5))
plt.plot(quantiles, correction, label="Original correction", color="blue")
plt.plot(quantiles, murdered_correction, color="red", linestyle="--", label="Correction with murdered tail")
plt.axhline(0, color="gray", linestyle="--", label="No correction")
plt.xlabel("Quantile")
plt.ylabel("Correction (Model - Observation) in mm/day")
plt.title(f"Correction Function for Daily Accumulated Precip for \nZürich (lat={lat_val:.3f}, lon={lon_val:.3f})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_path, dpi=1000)
print(f"Correction function plot saved to {plot_path}")


# ECDF
plt.figure(figsize=(7, 5))
obs_sorted = np.sort(obs_valid)
mod_sorted = np.sort(mod_valid)
obs_cdf = np.arange(1, len(obs_sorted)+1) / len(obs_sorted)
mod_cdf = np.arange(1, len(mod_sorted)+1) / len(mod_sorted)
plt.plot(obs_sorted, obs_cdf, label="Obs empirical CDF")
plt.plot(mod_sorted, mod_cdf, label="Model empirical CDF")
plt.xlabel("Value")
plt.ylabel("Cumulative Probability")
plt.title(f"Empirical CDFs for Daily Accumulated Precip for \nZürich (lat={lat_val:.3f}, lon={lon_val:.3f})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(cdf_plot_path, dpi=1000)
print(f"CDF plot saved to {cdf_plot_path}")

print("EQM Zürich validation completed successfully.")