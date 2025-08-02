import xarray as xr
import numpy as np
from SBCK import QM
import config
import dask
from dask.diagnostics import ProgressBar
import time

def eqm_cell(model_cell, obs_cell, calib_start, calib_end, model_times, obs_times):
    if model_cell.ndim != 1 or obs_cell.ndim != 1:
        ntime = len(model_times)
        return np.full(ntime, np.nan, dtype=np.float32)
    ntime = model_cell.shape[0]
    qm_series = np.full(ntime, np.nan, dtype=np.float32)

    if ntime == 0 or np.all(np.isnan(model_cell)) or np.all(np.isnan(obs_cell)):
        return qm_series

    model_times = xr.DataArray(model_times)
    obs_times = xr.DataArray(obs_times)
    calib_mask_mod = (model_times >= calib_start) & (model_times <= calib_end)
    calib_mask_obs = (obs_times >= calib_start) & (obs_times <= calib_end)

    if not calib_mask_mod.any() or not calib_mask_obs.any():
        return qm_series

    calib_mod_cell = model_cell[calib_mask_mod.values]
    calib_obs_cell = obs_cell[calib_mask_obs.values]
    calib_doys = model_times[calib_mask_mod].dt.dayofyear.values
    model_doys = model_times.dt.dayofyear.values

    for doy in range(1, 367):
        window_doys = ((calib_doys - doy + 366) % 366)
        window_mask = (window_doys <= 45) | (window_doys >= (366 - 45))
        obs_window = calib_obs_cell[window_mask]
        mod_window = calib_mod_cell[window_mask]
        obs_window = obs_window[~np.isnan(obs_window)]
        mod_window = mod_window[~np.isnan(mod_window)]
        if obs_window.size == 0 or mod_window.size == 0:
            continue
        eqm = QM()
        eqm.fit(mod_window.reshape(-1, 1), obs_window.reshape(-1, 1))
        indices = np.where(model_doys == doy)[0]
        for idx in indices:
            value = model_cell[idx]
            if np.isnan(value):
                continue
            qm_series[idx] = eqm.predict(np.array([[value]])).flatten()[0]
    return qm_series

def main():
    start = time.time()
    model_path = f"{config.SCRATCH_DIR}/tmax_r01_HR_masked.nc"
    obs_path = f"{config.SCRATCH_DIR}/TmaxD_1971_2023.nc"
    output_path = f"{config.BIAS_CORRECTED_DIR}/EQM/eqm_tmax_r01_allcells_DOY.nc"

    print("Loading data")
    model_output = xr.open_dataset(model_path,chunks={"lat": 20, "lon":20})["tmax"]
    obs_output = xr.open_dataset(obs_path,chunks={"lat": 20,"lon":20})["TmaxD"]

    model_output = model_output.sel(time=slice("1981-01-01", "2010-12-31"))
    obs_output = obs_output.sel(time=slice("1981-01-01", "2010-12-31"))

    # Ensure time is a single chunk for each grid cell (required for EQM)
    model_output = model_output.chunk({'time': -1})
    obs_output = obs_output.chunk({'time': -1})

    print("First 10 model times:", model_output['time'].values[:10])
    print("First 10 obs times:", obs_output['time'].values[:10])
    print("Model dtype:", model_output['time'].dtype)
    print("Obs dtype:", obs_output['time'].dtype)
    print("Are all times equal?", np.array_equal(model_output['time'].values, obs_output['time'].values))

    if not np.array_equal(model_output['time'].values, obs_output['time'].values):
        if len(model_output['time']) == len(obs_output['time']):
            print("Time coordinates do not match exactly but lengths match. Forcibly assigning obs time to model time.")
            obs_output = obs_output.assign_coords(time=model_output['time'])
        else:
            print("ERROR: Time coordinates do not match and lengths differ. Cannot proceed.")
            print("Model times:", model_output['time'].values)
            print("Obs times:", obs_output['time'].values)
            return

    print("Data loading took", time.time() - start, "seconds")
    start = time.time()

    # Dask parallelism with progress bar
    print("Starting EQM correction for all grid cells (Dask parallelism)...")
    qm_data = xr.apply_ufunc(
        eqm_cell,
        model_output,
        obs_output,
        input_core_dims=[['time'], ['time']],
        kwargs={
            'calib_start': np.datetime64("1981-01-01"),
            'calib_end': np.datetime64("2010-12-31"),
            'model_times': model_output['time'].values,
            'obs_times': obs_output['time'].values
        },
        output_core_dims=[['time']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.float32]
    )

    with ProgressBar():
        qm_data = qm_data.compute()

    print("EQM correction took", time.time() - start, "seconds")

    qm_ds = xr.Dataset(
        {"tmax": qm_data},
        coords=model_output.coords
    )
    qm_ds.to_netcdf(output_path)
    print(f"Bias-corrected tmax saved to {output_path}")

if __name__ == "__main__":
    main()