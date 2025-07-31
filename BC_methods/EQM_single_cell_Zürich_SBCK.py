import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from SBCK import QM
import config
import pandas as pd

model_path = f"{config.SCRATCH_DIR}/temp_r01_HR_masked.nc"
obs_path = f"{config.SCRATCH_DIR}/TabsD_1971_2023.nc"
output_path = f"{config.BC_DIR}/qm_temp_r01_singlecell_output.nc"
plot_path = f"{config.OUTPUTS_MODELS_DIR}/qm_correction_function_temp_r01_zurich.png"
cdf_plot_path = f"{config.OUTPUTS_MODELS_DIR}/qm_cdf_temp_r01_zurich.png"
map_plot_path = f"{config.OUTPUTS_MODELS_DIR}/qm_selected_gridcell_map_temp_r01_zurich.png"

print("Loading data")
model_output = xr.open_dataset(model_path)["temp"]
obs_output = xr.open_dataset(obs_path)["TabsD"]
calib_obs = obs_output.sel(time=slice("1981-01-01", "2010-12-31"))
calib_mod = model_output.sel(time=slice("1981-01-01", "2010-12-31"))

lat_vals = model_output['lat'].values
lon_vals = model_output['lon'].values

print("model_output shape:", model_output.shape)
print("lat_vals shape:", lat_vals.shape)
print("lon_vals shape:", lon_vals.shape)

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

print("Fitting EQM for Zurich")
eqm = QM()
eqm.fit(mod_valid.reshape(-1, 1), obs_valid.reshape(-1, 1))

full_mod_series = model_output[:, i_zurich, j_zurich].values.reshape(-1, 1)
qm_series = eqm.predict(full_mod_series).flatten()

qm_data = np.full(model_output.shape, np.nan, dtype=np.float32)
qm_data[:, i_zurich, j_zurich] = qm_series.astype(np.float32)

qm_ds = xr.Dataset(
    {"temp": (model_output.dims, qm_data)},
    coords=model_output.coords
)
qm_ds.to_netcdf(output_path)
print(f"Single-cell output saved to {output_path}")

# Correction fx with quantiles
quantiles = np.linspace(0.01, 0.99, 99) # 99 quantiles from 1 to 99th percentiles
plot_obs_q = np.quantile(obs_valid, quantiles)
plot_mod_q = np.quantile(mod_valid, quantiles)
correction = plot_mod_q - plot_obs_q
lat_val = lat_vals[i_zurich, j_zurich]
lon_val = lon_vals[i_zurich, j_zurich]

#Extended correction 
extended_quantiles= np.concatenate([[0.0],quantiles,[1.0]])
extended_correction= np.concatenate(([correction[0]],correction,[correction[-1]]))


plt.figure(figsize=(7, 5))
plt.plot(extended_quantiles, extended_correction, label="correction function", color="blue")
plt.axhline(0, color="gray", linestyle="--", label="No correction")
plt.xlabel("Quantile")
plt.ylabel("Correction (Model - Observation) in degrees C")
plt.title(f"Correction Function for Daily Average Temperature for \nZürich (lat={lat_val:.3f}, lon={lon_val:.3f})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_path, dpi=1000)
print(f"Correction function plot saved to {plot_path}")
