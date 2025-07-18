import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import config

model_path = "/scratch/sasthana/temp_r01_HR_masked.nc"
obs_path = "/scratch/sasthana/TabsD_1971_2023.nc"
output_path = "/scratch/sasthana/qm_temp_r01_output.nc"

print("Data")
model_output = xr.open_dataset(model_path,chunks={"time":2000})["temp"]
obs_output = xr.open_dataset(obs_path,chunks={"time":2000})["TabsD"]

print("Calibration:1981-2010")
calib_obs = obs_output.sel(time=slice("1981-01-01", "2010-12-31"))
calib_mod = model_output.sel(time=slice("1981-01-01", "2010-12-31"))

quantiles = np.linspace(0, 1, 1001)
ntime, nlat, nlon = model_output.shape

zurich_lat = 47.3769
zurich_lon = 8.5417
lat_vals = model_output['lat'].values
lon_vals = model_output['lon'].values
i_zurich = np.argmin(np.abs(lat_vals - zurich_lat))
j_zurich = np.argmin(np.abs(lon_vals - zurich_lon))
plot_obs_q = plot_mod_q = None

qm_data = np.full(model_output.shape, np.nan, dtype=np.float32)

print("EQM")
for i in range(nlat):
    for j in range(nlon):
        obs_series = calib_obs[:, i, j].values
        mod_series = calib_mod[:, i, j].values
        obs_valid = obs_series[~np.isnan(obs_series)]
        mod_valid = mod_series[~np.isnan(mod_series)]
        if obs_valid.size== 0 or mod_valid.size== 0:
            continue
        obs_q = np.quantile(obs_valid, quantiles)
        mod_q = np.quantile(mod_valid, quantiles)
        full_mod_series = model_output[:, i, j].values
        qm_series = np.interp(full_mod_series, mod_q, obs_q)
        qm_data[:, i, j] = qm_series.astype(np.float32)
        if i == i_zurich and j == j_zurich:
            plot_obs_q = obs_q
            plot_mod_q = mod_q
    if i % 10 == 0:
        print(f"Processed latitude {i}/{nlat}")

print("Writing O/P")
qm_ds = xr.Dataset(
    {"temp": (model_output.dims, qm_data)},
    coords=model_output.coords
)
qm_ds.to_netcdf(output_path)
print(f"Output saved to {output_path}")

# Correction quantile prob fx 
if plot_obs_q is not None and plot_mod_q is not None:
    plt.figure(figsize=(7, 5))
    plt.plot(plot_mod_q, plot_obs_q, label="Correction function (obs vs model)")
    plt.plot(plot_mod_q, plot_mod_q, "--", color="gray", label="1:1 line")
    plt.xlabel("Model quantiles")
    plt.ylabel("Observed quantiles")
    plt.title(f"Quantile Mapping Correction Function\nZürich for Daily Avg Temp (lat={lat_vals[i_zurich]:.3f}, lon={lon_vals[j_zurich]:.3f})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUTS_MODELS_DIR}/qm_correction_function_temp_r01_zurich.png", dpi=500)