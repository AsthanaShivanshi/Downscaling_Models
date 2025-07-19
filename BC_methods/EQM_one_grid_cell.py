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
cdf_plot_path = f"{config.BC_DIR}/qm_cdf_tmax_r01_randomcell.png"
map_plot_path = f"{config.BC_DIR}/qm_selected_gridcell_map.png"

print("Loading data")
model_output = xr.open_dataset(model_path)["tmax"]
obs_output = xr.open_dataset(obs_path)["TmaxD"]
calib_obs = obs_output.sel(time=slice("1981-01-01", "2010-12-31"))
calib_mod = model_output.sel(time=slice("1981-01-01", "2010-12-31"))

lat_vals = model_output['lat'].values
lon_vals = model_output['lon'].values

print("model_output shape:", model_output.shape)
print("lat_vals shape:", lat_vals.shape)
print("lon_vals shape:", lon_vals.shape)

# Loop until a valid cell is found
max_attempts = 1000
for attempt in range(max_attempts):
    i_rand = np.random.randint(0, lat_vals.shape[0])
    j_rand = np.random.randint(0, lon_vals.shape[1])
    obs_valid = calib_obs[:, i_rand, j_rand].values[~np.isnan(calib_obs[:, i_rand, j_rand].values)]
    mod_valid = calib_mod[:, i_rand, j_rand].values[~np.isnan(calib_mod[:, i_rand, j_rand].values)]
    if obs_valid.size > 0 and mod_valid.size > 0:
        print(f"Valid grid cell found after {attempt+1} attempts: i={i_rand}, j={j_rand}")
        print(f"Location: lat={lat_vals[i_rand, j_rand]}, lon={lon_vals[i_rand, j_rand]}")
        break
else:
    print("Could not find a valid grid cell after 1000 attempts. Exiting.")
    exit(1)

print("Fitting EQM for valid grid cell...")
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
plt.title(f"Quantile Mapping Correction Function\nValid Cell (lat={lat_val:.3f}, lon={lon_val:.3f})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_path, dpi=500)
print(f"Correction function plot saved to {plot_path}")

plt.figure(figsize=(7, 5))
obs_sorted = np.sort(obs_valid)
mod_sorted = np.sort(mod_valid)
obs_cdf = np.arange(1, len(obs_sorted)+1) / len(obs_sorted)
mod_cdf = np.arange(1, len(mod_sorted)+1) / len(mod_sorted)
plt.plot(obs_sorted, obs_cdf, label="Obs empirical CDF")
plt.plot(mod_sorted, mod_cdf, label="Model empirical CDF")
plt.xlabel("Value")
plt.ylabel("Cumulative Probability")
plt.title(f"Empirical CDFs\nValid Cell (lat={lat_val:.3f}, lon={lon_val:.3f})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(cdf_plot_path, dpi=500)
print(f"CDF plot saved to {cdf_plot_path}")

#CH map with highlighted grid cell
plt.figure(figsize=(7, 7))
plt.pcolormesh(lon_vals, lat_vals, np.zeros_like(lat_vals), cmap="Greys", shading="auto")
plt.scatter(lon_val, lat_val, color="red", marker="*", s=400, label="Selected grid cell")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Selected Grid Cell on Switzerland Map")
plt.legend()
plt.tight_layout()
plt.savefig(map_plot_path, dpi=500)
print(f"Map plot with selected grid cell saved to {map_plot_path}")

print("EQM valid-cell test complete!")