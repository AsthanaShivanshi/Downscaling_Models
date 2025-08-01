import xarray as xr
import numpy as np
from SBCK import QM
import config

model_path = f"{config.SCRATCH_DIR}/tmax_r01_HR_masked.nc"
obs_path = f"{config.SCRATCH_DIR}/TmaxD_1971_2023.nc"
output_path = f"{config.BIAS_CORRECTED_DIR}/EQM/eqm_tmax_r01.nc"

print("Loading data")
model_output = xr.open_dataset(model_path)["tmax"]
obs_output = xr.open_dataset(obs_path)["TmaxD"]
calib_obs = obs_output.sel(time=slice("1981-01-01", "2010-12-31"))
calib_mod = model_output.sel(time=slice("1981-01-01", "2010-12-31"))

ntime, nlat, nlon = model_output.shape

calib_times = calib_mod['time'].values
calib_doys = xr.DataArray(calib_times).dt.dayofyear.values

model_times = model_output['time'].values
model_doys = xr.DataArray(model_times).dt.dayofyear.values
valid_mask = (model_times >= np.datetime64("1981-01-01")) & (model_times <= np.datetime64("2010-12-31"))

qm_data = np.full(model_output.shape, np.nan, dtype=np.float32)

for i in range(nlat):
    for j in range(nlon):
        qm_series = np.full(ntime, np.nan, dtype=np.float32)
        for doy in range(1, 367):  # 1 to 366
            window_doys = ((calib_doys - doy + 366) % 366)
            window_mask = (window_doys <= 45) | (window_doys >= (366 - 45))
            obs_window = calib_obs[:, i, j].values[window_mask]
            mod_window = calib_mod[:, i, j].values[window_mask]
            obs_window = obs_window[~np.isnan(obs_window)]
            mod_window = mod_window[~np.isnan(mod_window)]
            if obs_window.size == 0 or mod_window.size == 0:
                continue
            eqm = QM()
            eqm.fit(mod_window.reshape(-1, 1), obs_window.reshape(-1, 1))
            quantiles = np.linspace(0.01, 0.99, 99)
            obs_q = np.quantile(obs_window, quantiles)
            mod_q = np.quantile(mod_window, quantiles)
            # End quantiles 
            obs_q = np.concatenate([[obs_q[0]], obs_q, [obs_q[-1]]])
            mod_q = np.concatenate([[mod_q[0]], mod_q, [mod_q[-1]]])
            ext_q = np.linspace(0, 1, 101)
            indices = np.where((model_doys == doy) & valid_mask)[0]
            for idx in indices:
                value = model_output[idx, i, j]
                if np.isnan(value):
                    continue
                qm_series[idx] = eqm.predict(np.array([[value]])).flatten()[0]
        qm_data[:, i, j] = qm_series.astype(np.float32)

qm_ds = xr.Dataset(
    {"tmax": (model_output.dims, qm_data)},
    coords=model_output.coords
)
qm_ds.to_netcdf(output_path)
print(f"AllCells EQM BC completed and saved to {output_path}")