import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from SBCK import QM
import os
import config

model_path =f"{config.SCRATCH_DIR}/tmax_r01_HR_masked.nc"
obs_path = f"{config.SCRATCH_DIR}/TmaxD_1971_2023.nc"
output_path = f"{config.BC_DIR}/qm_tmax_r01_singlecell_output.nc"
plot_path = f"{config.BC_DIR}/qm_correction_function_tmax_r01_randomcell.png"

print("Loading data...")
model_output = xr.open_dataset(model_path)["tmax"]
obs_output = xr.open_dataset(obs_path)["TmaxD"]
calib_obs = obs_output.sel(time=slice("1981-01-01", "2010-12-31"))
calib_mod = model_output.sel(time=slice("1981-01-01", "2010-12-31"))

lat_vals = model_output['lat'].values
lon_vals = model_output['lon'].values

print("model_output shape:", model_output.shape)
print("lat_vals shape:", lat_vals.shape)
print("lon_vals shape:", lon_vals.shape)

i_rand = np.random.randint(0, lat_vals.shape[0])
j_rand = np.random.randint(0, lon_vals.shape[0])
print(f"Random grid cell indices: i={i_rand}, j={j_rand}")
print(f"Random grid cell location: lat={lat_vals[i_rand]}, lon={lon_vals[j_rand]}")

obs_valid = calib_obs[:, i_rand, j_rand].values[~np.isnan(calib_obs[:, i_rand, j_rand].values)]
mod_valid = calib_mod[:, i_rand, j_rand].values[~np.isnan(calib_mod[:, i_rand, j_rand].values)]

if obs_valid.size == 0 or mod_valid.size == 0:
    print("No valid data for random grid cell. Exiting.")
    exit(1)

print("Fitting EQM for random grid cell...")
eqm = QM()
eqm.fit(mod_valid.reshape(-1, 1), obs_valid.reshape(-1, 1))

full_mod_series = model_output[:, i_rand, j_rand].values.reshape(-1, 1)
qm_series = eqm.predict(full_mod_series).flatten()

qm_data = np.full(model_output.shape, np.nan, dtype=np.float32)
qm_data[:, i_rand, j_rand] = qm_series.astype(np.float32)

qm_ds = xr.Dataset(
    {"tmax": (model_output.dims, qm_data)},
    coords=model_output.coords
)
qm_ds.to_netcdf(output_path)
print(f"Single-cell output saved to {output_path}")

quantiles = np.linspace(0, 1, 1001)
plot_obs_q = np.quantile(obs_valid, quantiles)
plot_mod_q = np.quantile(mod_valid, quantiles)
lat_val = lat_vals[i_rand, j_rand]
lon_val = lon_vals[i_rand, j_rand]


plt.figure(figsize=(7, 5))
plt.plot(plot_mod_q, plot_obs_q, label="Correction function (obs vs model)")
plt.plot(plot_mod_q, plot_mod_q, "--", color="gray", label="1:1 line")
plt.xlabel("Model quantiles")
plt.ylabel("Observed quantiles")
plt.title(f"Quantile Mapping Correction Function\nRandom Cell (lat={lat_val:.3f}, lon={lon_val:.3f})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_path, dpi=500)
print(f"Correction function plot saved to {plot_path}")

print("EQM random-cell EQM done")