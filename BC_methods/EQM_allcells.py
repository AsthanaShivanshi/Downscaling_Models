import xarray as xr
import numpy as np
from SBCK import QM
import config
import pandas as pd

model_path = f"{config.SCRATCH_DIR}/temp_r01_HR_masked.nc"
obs_path = f"{config.SCRATCH_DIR}/TabsD_1971_2023.nc"
output_path = f"{config.BIAS_CORRECTED_DIR}/EQM/eqm_temp_r01.nc"

model_output = xr.open_dataset(model_path)["temp"]
obs_output = xr.open_dataset(obs_path)["TabsD"]

calib_start = "1981-01-01"
calib_end = "2010-12-31"
calib_obs = obs_output.sel(time=slice(calib_start, calib_end))
calib_mod = model_output.sel(time=slice(calib_start, calib_end))

# Align calibration time axes
common_times = np.intersect1d(calib_obs['time'].values, calib_mod['time'].values)
calib_obs = calib_obs.sel(time=common_times)
calib_mod = calib_mod.sel(time=common_times)
calib_times = pd.to_datetime(common_times)
calib_doys = calib_times.dayofyear

model_times = pd.to_datetime(model_output['time'].values)
model_doys = model_times.dayofyear

ntime, nlat, nlon = model_output.shape
qm_data = np.full(model_output.shape, np.nan, dtype=np.float32)

quantiles = np.linspace(0.01, 0.99, 99) # 99 quantiles for correction

for i in range(nlat):
    for j in range(nlon):
        obs_cell = calib_obs[:, i, j].values
        mod_cell = calib_mod[:, i, j].values
        # NaN handling
        if np.all(np.isnan(obs_cell)) or np.all(np.isnan(mod_cell)):
            continue

        cell_series = np.full(ntime, np.nan, dtype=np.float32)

        for doy in range(1, 367):
            # 91-day window, wrap-around
            window_doys = ((calib_doys - doy + 366) % 366)
            window_mask = (window_doys <= 45) | (window_doys >= (366 - 45))
            obs_window = obs_cell[window_mask]
            mod_window = mod_cell[window_mask]
            obs_window = obs_window[~np.isnan(obs_window)]
            mod_window = mod_window[~np.isnan(mod_window)]
            if obs_window.size == 0 or mod_window.size == 0:
                continue

            eqm = QM()
            eqm.fit(mod_window.reshape(-1, 1), obs_window.reshape(-1, 1))

            quantiles = np.linspace(0.01, 0.99, 99)
            obs_q = np.quantile(obs_window, quantiles)
            mod_q = np.quantile(mod_window, quantiles)
            # Set 0th and 100th percentiles to 1st and 99th
            obs_q = np.concatenate([[obs_q[0]], obs_q, [obs_q[-1]]])
            mod_q = np.concatenate([[mod_q[0]], mod_q, [mod_q[-1]]])
            correction = mod_q - obs_q
            ext_q = np.linspace(0, 1, 101)

            indices = np.where(model_doys == doy)[0]
            for idx in indices:
                value = model_output[idx, i, j]
                cell_series[idx] = eqm.predict(np.array([[value]])).flatten()[0]

        qm_data[:, i, j] = cell_series

qm_ds = xr.Dataset(
    {"temp": (model_output.dims, qm_data)},
    coords=model_output.coords
)
qm_ds.to_netcdf(output_path)
print(f"Bias-corrected EQM output saved to {output_path}")