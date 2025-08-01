import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from SBCK import QM
import config

model_path = f"{config.SCRATCH_DIR}/tmin_r01_HR_masked.nc"
obs_path = f"{config.SCRATCH_DIR}/TminD_1971_2023.nc"
output_path_template = f"{config.BC_DIR}/qm_tmin_r01_singlecell_{{city}}_output.nc"
plot_path = f"{config.OUTPUTS_MODELS_DIR}/qm_correction_function_tmin_r01_3cities_window.png"

print("Loading data")
model_output = xr.open_dataset(model_path)["tmin"]
obs_output = xr.open_dataset(obs_path)["TminD"]
calib_obs = obs_output.sel(time=slice("1981-01-01", "2010-12-31"))
calib_mod = model_output.sel(time=slice("1981-01-01", "2010-12-31"))

lat_vals = model_output['lat'].values
lon_vals = model_output['lon'].values

locations = {
    "Zurich": (47.3769, 8.5417),
    "Geneva": (46.2044, 6.1432),
    "Locarno": (46.1670, 8.7943),
}

calib_times = calib_mod['time'].values
calib_doys = xr.DataArray(calib_times).dt.dayofyear.values

model_times = model_output['time'].values
model_doys = xr.DataArray(model_times).dt.dayofyear.values
valid_mask = (model_times >= np.datetime64("1981-01-01")) & (model_times <= np.datetime64("2010-12-31"))

# For plotting mean correction functions
plt.figure(figsize=(8, 6))

for city, (target_lat, target_lon) in locations.items():
    print(f"\nProcessing {city}...")
    dist = np.sqrt((lat_vals - target_lat)**2 + (lon_vals - target_lon)**2)
    i_city, j_city = np.unravel_index(np.argmin(dist), dist.shape)
    print(f"Closest grid cell to {city}: i={i_city}, j={j_city}")
    print(f"Location: lat={lat_vals[i_city, j_city]}, lon={lon_vals[i_city, j_city]}")

    qm_series = np.full(model_output.shape[0], np.nan, dtype=np.float32)
    city_correction_functions = {}

    for doy in range(1, 367):  # 1 to 366
        window_doys = ((calib_doys - doy + 366) % 366)
        window_mask = (window_doys <= 45) | (window_doys >= (366 - 45))
        obs_window = calib_obs[:, i_city, j_city].values[window_mask]
        mod_window = calib_mod[:, i_city, j_city].values[window_mask]
        obs_window = obs_window[~np.isnan(obs_window)]
        mod_window = mod_window[~np.isnan(mod_window)]
        if obs_window.size == 0 or mod_window.size == 0:
            continue
        eqm = QM()
        eqm.fit(mod_window.reshape(-1, 1), obs_window.reshape(-1, 1))
        quantiles = np.linspace(0.01, 0.99, 99)
        obs_q = np.quantile(obs_window, quantiles)
        mod_q = np.quantile(mod_window, quantiles)
        obs_q = np.concatenate([[obs_q[0]], obs_q, [obs_q[-1]]])
        mod_q = np.concatenate([[mod_q[0]], mod_q, [mod_q[-1]]])
        correction = mod_q - obs_q
        ext_q = np.linspace(0, 1, 101)
        city_correction_functions[doy] = (ext_q, correction)
        indices = np.where((model_doys == doy) & valid_mask)[0]
        for idx in indices:
            value = model_output[idx, i_city, j_city]
            qm_series[idx] = eqm.predict(np.array([[value]])).flatten()[0]

    # Save corrected series for this city
    qm_data = np.full(model_output.shape, np.nan, dtype=np.float32)
    qm_data[:, i_city, j_city] = qm_series.astype(np.float32)
    qm_ds = xr.Dataset(
        {"tmin": (model_output.dims, qm_data)},
        coords=model_output.coords
    )
    output_path = output_path_template.format(city=city.lower())
    qm_ds.to_netcdf(output_path)
    print(f"Single-cell output for {city} saved to {output_path}")

    # Plot mean correction function for this city
    all_corrs = []
    for doy in city_correction_functions:
        ext_q, ext_corr = city_correction_functions[doy]
        all_corrs.append(ext_corr)
    if all_corrs:
        all_corrs = np.stack(all_corrs, axis=0)
        mean_corr = np.mean(all_corrs, axis=0)
        plt.plot(ext_q, mean_corr, label=city)

plt.axhline(0, color="gray", linestyle="--")
plt.xlabel("Quantile")
plt.ylabel("Correction (Model - Observation)")
plt.title("Mean Correction Function (All DOYs)\nZurich, Geneva, Locarno")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_path, dpi=1000)
print(f"Overall mean correction function plot for 3 cities saved to {plot_path}")