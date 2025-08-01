import xarray as xr
import numpy as np
from SBCK import QM
import config
from joblib import Parallel, delayed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_jobs', type=int, default=1)
args = parser.parse_args()
n_jobs = args.n_jobs

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

def process_cell(i, j):
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
        quantiles = np.linspace(0.01, 0.99, 99)
        obs_q = np.quantile(obs_window, quantiles)
        mod_q = np.quantile(mod_window, quantiles)
        # Extend to 0th and 100th using 1st and 99th
        ext_q = np.concatenate([[0], quantiles, [1]])
        ext_obs_q = np.concatenate([[obs_q[0]], obs_q, [obs_q[-1]]])
        ext_mod_q = np.concatenate([[mod_q[0]], mod_q, [mod_q[-1]]])
        indices = np.where((model_doys == doy) & valid_mask)[0]
        for idx in indices:
            value = model_output[idx, i, j]
            if np.isnan(value):
                continue
            q = np.interp(value, ext_mod_q, ext_q)
            # Correction 
            corr = np.interp(q, ext_q, ext_mod_q - ext_obs_q)
            qm_series[idx] = value - corr
    return (i, j, qm_series.astype(np.float32))

print(f"Running with {n_jobs} parallel jobs")
results = Parallel(n_jobs=n_jobs)(
    delayed(process_cell)(i, j) for i in range(nlat) for j in range(nlon)
)

qm_data = np.full(model_output.shape, np.nan, dtype=np.float32)
for i, j, series in results:
    qm_data[:, i, j] = series

qm_ds = xr.Dataset(
    {"tmax": (model_output.dims, qm_data)},
    coords=model_output.coords
)
qm_ds.to_netcdf(output_path)
print(f"AllCells EQM BC completed and saved to {output_path}")