import xarray as xr
import numpy as np
from SBCK import QM
import config
from joblib import Parallel, delayed
import cProfile, pstats
pr = cProfile.Profile()
pr.enable()
import pandas as pd


model_path = f"{config.SCRATCH_DIR}/temp_r01_HR_masked.nc"
obs_path = f"{config.SCRATCH_DIR}/TabsD_1971_2023.nc"
output_path = f"{config.BIAS_CORRECTED_DIR}/EQM/eqm_temp_r01.nc"

print("Loading data")
model_output = xr.open_dataset(model_path)["temp"]
obs_output = xr.open_dataset(obs_path)["TabsD"]
calib_obs = obs_output.sel(time=slice("1981-01-01", "2010-12-31"))
calib_mod = model_output.sel(time=slice("1981-01-01", "2010-12-31"))

lat_vals = model_output['lat'].values
lon_vals = model_output['lon'].values

calib_times = pd.to_datetime(calib_mod['time'].values)
model_times = pd.to_datetime(model_output['time'].values)
model_doys = model_times.dayofyear

ntime, nlat, nlon = model_output.shape
qm_data = np.full(model_output.shape, np.nan, dtype=np.float32)

quantiles = np.linspace(0.01, 0.99, 99)

for i in range(nlat):
    for j in range(nlon):
        obs_cell = calib_obs[:, i, j].values
        mod_cell = calib_mod[:, i, j].values
        obs_cell = obs_cell[~np.isnan(obs_cell)]
        mod_cell = mod_cell[~np.isnan(mod_cell)]
        if obs_cell.size == 0 or mod_cell.size == 0:
            continue

        cell_series = np.full(ntime, np.nan, dtype=np.float32)

        # Corr fx
        for doy in range(1, 367):
            calib_doys = calib_times.dayofyear
            window_mask = ((calib_doys >= doy - 45) & (calib_doys <= doy + 45)) | \
                          ((doy - 45 < 1) & (calib_doys >= 365 + (doy - 45))) | \
                          ((doy + 45 > 366) & (calib_doys <= (doy + 45) - 366))
            obs_window = calib_obs[:, i, j].values[window_mask]
            mod_window = calib_mod[:, i, j].values[window_mask]
            obs_window = obs_window[~np.isnan(obs_window)]
            mod_window = mod_window[~np.isnan(mod_window)]
            if obs_window.size == 0 or mod_window.size == 0:
                continue

            eqm = QM()
            eqm.fit(mod_window.reshape(-1, 1), obs_window.reshape(-1, 1))

            indices = np.where(model_doys == doy)[0]
            for idx in indices:
                value = model_output[idx, i, j]
                cell_series[idx] = eqm.predict(np.array([[value]])).flatten()[0]

        qm_data[:, i, j] = cell_series

qm_ds = xr.Dataset(
    {"tmax": (model_output.dims, qm_data)},
    coords=model_output.coords
)
qm_ds.to_netcdf(output_path)
print(f"Bias-corrected seasonal QM output saved to {output_path}")