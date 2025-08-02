import xarray as xr
import numpy as np
from SBCK import QM
import config
import argparse
import cProfile
import dask
#dask.config.set(scheduler='single-threaded') #For pirnt statements in log files

def eqm_cell(model_cell, obs_cell, calib_start, calib_end, model_times, obs_times, quantiles):
    print("  model_cell.shape:", model_cell.shape, "obs_cell.shape:", obs_cell.shape)
    print("  calib_start:", calib_start, "calib_end:", calib_end)
    print("  model_times.shape:", model_times.shape, "obs_times.shape:", obs_times.shape)

    if model_cell.ndim != 1 or obs_cell.ndim != 1:
        print("Returning NaNs")
        ntime = len(model_times)
        return np.full(ntime, np.nan, dtype=np.float32)
    ntime = model_cell.shape[0]
    qm_series = np.full(ntime, np.nan, dtype=np.float32)

    if ntime == 0 or np.all(np.isnan(model_cell)) or np.all(np.isnan(obs_cell)):
        print(" Returning NaNs")
        return qm_series

    model_times = xr.DataArray(model_times)
    obs_times = xr.DataArray(obs_times)
    calib_mask_mod = (model_times >= calib_start) & (model_times <= calib_end)
    calib_mask_obs = (obs_times >= calib_start) & (obs_times <= calib_end)

    print("  calib_mask_mod.sum():", calib_mask_mod.sum().item(), "calib_mask_obs.sum():", calib_mask_obs.sum().item())

    if not calib_mask_mod.any() or not calib_mask_obs.any():
        print(" Returning NaNs")
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_jobs', type=int, default=1)
    args = parser.parse_args()

    model_path = f"{config.SCRATCH_DIR}/tmin_r01_HR_masked.nc"
    obs_path = f"{config.SCRATCH_DIR}/TminD_1971_2023.nc"
    output_path = f"{config.BIAS_CORRECTED_DIR}/EQM/eqm_tmin_r01.nc"

    print("Loading data")
    model_output = xr.open_dataset(model_path)["tmin"]
    obs_output = xr.open_dataset(obs_path)["TminD"]

    print("Model time range:", str(model_output['time'].values[0]), "to", str(model_output['time'].values[-1]), "len:", len(model_output['time']))
    print("Obs time range:", str(obs_output['time'].values[0]), "to", str(obs_output['time'].values[-1]), "len:", len(obs_output['time']))

    model_output = model_output.sel(time=slice("1981-01-01", "2010-12-31"))
    obs_output = obs_output.sel(time=slice("1981-01-01", "2010-12-31"))

    print("After selection:")
    print("  Model time range:", str(model_output['time'].values[0]), "to", str(model_output['time'].values[-1]), "len:", len(model_output['time']))
    print("  Obs time range:", str(obs_output['time'].values[0]), "to", str(obs_output['time'].values[-1]), "len:", len(obs_output['time']))

    if not np.array_equal(model_output['time'].values, obs_output['time'].values):
        if len(model_output['time']) == len(obs_output['time']):
            print("Time coordinates do not match exactly but lengths match. Forcibly assigning obs time to model time.")
            obs_output = obs_output.assign_coords(time=model_output['time'])
        else:
            print("ERROR: Time coordinates do not match and lengths differ. Cannot proceed.")
            print("Model times:", model_output['time'].values)
            print("Obs times:", obs_output['time'].values)
            return

    quantiles = np.linspace(0.01, 0.99, 99)
    qm_data = xr.apply_ufunc(
        eqm_cell,
        model_output,
        obs_output,
        input_core_dims=[['time'], ['time']],
        kwargs={
            'calib_start': np.datetime64("1981-01-01"),
            'calib_end': np.datetime64("2010-12-31"),
            'model_times': model_output['time'].values,
            'obs_times': obs_output['time'].values,
            'quantiles': quantiles
        },
        output_core_dims=[['time']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.float32]
    )

    qm_ds = xr.Dataset(
        {"tmin": qm_data},
        coords=model_output.coords
    )
    qm_ds.to_netcdf(output_path)
    print(f"AllCells EQM BC completed and saved to {output_path}")

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    profiler.dump_stats("eqm_allcells.prof")
    print("Profiling complete. Stats saved to eqm_allcells.prof")