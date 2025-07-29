import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from SBCK import QM
import config
import cartopy.crs as ccrs
import cartopy.feature as cfeature

model_path = f"{config.SCRATCH_DIR}/precip_r01_HR_masked.nc"
obs_path = f"{config.SCRATCH_DIR}/RhiresD_1971_2023.nc"
output_path = f"{config.BC_DIR}/qm_precip_r01_singlecell_output.nc"
plot_path = f"{config.OUTPUTS_MODELS_DIR}/qm_correction_function_precip_r01_geneva.png"
cdf_plot_path = f"{config.OUTPUTS_MODELS_DIR}/qm_cdf_precip_r01_geneva.png"
map_plot_path = f"{config.OUTPUTS_MODELS_DIR}/qm_selected_gridcell_map_precip_r01_geneva.png"

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
i_geneva, j_geneva = np.unravel_index(np.argmin(dist), dist.shape)
print(f"Closest grid cell to Geneva: i={i_geneva}, j={j_geneva}")
print(f"Location: lat={lat_vals[i_geneva, j_geneva]}, lon={lon_vals[i_geneva, j_geneva]}")

obs_valid = calib_obs[:, i_geneva, j_geneva].values[~np.isnan(calib_obs[:, i_geneva, j_geneva].values)]
mod_valid = calib_mod[:, i_geneva, j_geneva].values[~np.isnan(calib_mod[:, i_geneva, j_geneva].values)]
if obs_valid.size == 0 or mod_valid.size == 0:
    print("No valid data for Geneva grid cell. Exiting.")
    exit(1)

print("Fitting EQM for Geneva")
eqm = QM()
eqm.fit(mod_valid.reshape(-1, 1), obs_valid.reshape(-1, 1))

full_mod_series = model_output[:, i_geneva, j_geneva].values.reshape(-1, 1)
qm_series = eqm.predict(full_mod_series).flatten()

qm_data = np.full(model_output.shape, np.nan, dtype=np.float32)
qm_data[:, i_geneva, j_geneva] = qm_series.astype(np.float32)

qm_ds = xr.Dataset(
    {"tmin": (model_output.dims, qm_data)},
    coords=model_output.coords
)
qm_ds.to_netcdf(output_path)
print(f"Single-cell output saved to {output_path}")

# Correction fx with quantiles : (model - obs)
quantiles = np.linspace(0, 1, 1001)
plot_obs_q = np.quantile(obs_valid, quantiles)
plot_mod_q = np.quantile(mod_valid, quantiles)
correction = plot_mod_q - plot_obs_q
lat_val = lat_vals[i_geneva, j_geneva]
lon_val = lon_vals[i_geneva, j_geneva]

plt.figure(figsize=(7, 5))
plt.plot(quantiles, correction, label="Correction (model - obs)")
plt.axhline(0, color="gray", linestyle="--", label="No correction")
plt.xlabel("Quantile")
plt.ylabel("Correction (Model - Observation) in degrees C")
plt.title(f"Correction Function for Daily Min Temp for \nGeneva (lat={lat_val:.3f}, lon={lon_val:.3f})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_path, dpi=500)
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
plt.title(f"Empirical CDFs for Daily Min Temp for \nZurich (lat={lat_val:.3f}, lon={lon_val:.3f})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(cdf_plot_path, dpi=500)
print(f"CDF plot saved to {cdf_plot_path}")

plt.figure(figsize=(8, 8))
ax = plt.axes(projection=ccrs.Mercator())
ax.set_extent([5.5, 10.5, 45.5, 47.9], crs=ccrs.PlateCarree()) 

ax.add_feature(cfeature.BORDERS, linewidth=1)
ax.add_feature(cfeature.LAND, facecolor='lightgray')
background = np.nanmean(model_output.values, axis=0)
mesh = ax.pcolormesh(lon_vals, lat_vals, background, cmap="coolwarm", shading="auto", transform=ccrs.PlateCarree())
plt.colorbar(mesh, ax=ax, orientation='vertical', label="degrees C")

ax.plot(lon_val, lat_val, marker="*", color="black", markersize=18, markeredgewidth=2, label="Geneva grid cell", transform=ccrs.PlateCarree())
plt.title("Geneva Grid Cell on Switzerland Map", fontsize=15)
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(map_plot_path, dpi=1000)
print(f"Map plot with Geneva grid cell saved to {map_plot_path}")

print("EQM Geneva validation completed successfully.")