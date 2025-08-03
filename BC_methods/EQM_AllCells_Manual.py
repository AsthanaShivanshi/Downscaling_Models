import xarray as xr
import numpy as np
from SBCK import QM
import config
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

    quantiles = np.linspace(0.01, 0.99, 99)  # 99 quantiles between 0.01 and 0.99

    for doy in range(1, 367):
        window_doys = ((calib_doys - doy + 366) % 366)
        window_mask = (window_doys <= 45) | (window_doys >= (366 - 45))
        obs_window = calib_obs_cell[window_mask]
        mod_window = calib_mod_cell[window_mask]
        obs_window = obs_window[~np.isnan(obs_window)]
        mod_window = mod_window[~np.isnan(mod_window)]
        if obs_window.size == 0 or mod_window.size == 0:
            continue

        obs_q = np.quantile(obs_window, quantiles)
        mod_q = np.quantile(mod_window, quantiles)

        indices = np.where(model_doys == doy)[0]
        if indices.size > 0:
            values = model_cell[indices]
            # Linear interpolation between 1st and 99th percentiles,
            # assign 1st percentile correction below, 99th above
            corrected = np.interp(values, mod_q, obs_q, left=obs_q[0], right=obs_q[-1])
            qm_series[indices] = corrected

    return qm_series

def main():
    from dask.distributed import Client, performance_report
    import logging

    # Set up logging for progress tracking
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")

    client = Client(n_workers=8, threads_per_worker=1)
    logging.info(f"Dask dashboard: {client.dashboard_link}")

    start = time.time()
    model_path = f"{config.SCRATCH_DIR}/tmax_r01_HR_masked.nc"
    obs_path = f"{config.SCRATCH_DIR}/TmaxD_1971_2023.nc"
    output_path = f"{config.BIAS_CORRECTED_DIR}/EQM/eqm_tmax_r01_allcells_DOY.nc"

    model_output = xr.open_dataset(model_path)["tmax"]
    obs_output = xr.open_dataset(obs_path)["TmaxD"]
    model_output = model_output.sel(time=slice("1981-01-01", "2010-12-31"))
    obs_output = obs_output.sel(time=slice("1981-01-01", "2010-12-31"))

    logging.info(f"Model dims: {model_output.dims}")
    logging.info(f"Obs dims: {obs_output.dims}")

    # Use actual dimension names for chunking
    spatial_dims = [d for d in model_output.dims if d != "time"]
    chunk_dict = {d: 10 for d in spatial_dims}
    chunk_dict["time"] = -1
    model_output = model_output.chunk(chunk_dict)
    obs_output = obs_output.chunk(chunk_dict)

    logging.info(f"First 10 model times: {model_output['time'].values[:10]}")
    logging.info(f"First 10 obs times: {obs_output['time'].values[:10]}")
    logging.info(f"Model dtype: {model_output['time'].dtype}")
    logging.info(f"Obs dtype: {obs_output['time'].dtype}")
    logging.info(f"Are all times equal? {np.array_equal(model_output['time'].values, obs_output['time'].values)}")

    # Print chunk structure for debugging Dask parallelism
    logging.info(f"Model data chunk structure: {model_output.chunks}")
    logging.info(f"Obs data chunk structure: {obs_output.chunks}")

    if not np.array_equal(model_output['time'].values, obs_output['time'].values):
        if len(model_output['time']) == len(obs_output['time']):
            logging.warning("Time coordinates do not match exactly but lengths match. Forcibly assigning obs time to model time.")
            obs_output = obs_output.assign_coords(time=model_output['time'])
        else:
            logging.error("ERROR: Time coordinates do not match and lengths differ. Cannot proceed.")
            logging.error(f"Model times: {model_output['time'].values}")
            logging.error(f"Obs times: {obs_output['time'].values}")
            return

    logging.info(f"Data loading took {time.time() - start:.2f} seconds")
    start = time.time()

    # Dask parallelism with progress bar and performance report
    logging.info("Starting EQM correction for all grid cells (Dask parallelism)")
    with performance_report(filename="dask_performance_report.html"):
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

        logging.info("About to start Dask compute with ProgressBar")
        with ProgressBar():
            qm_data = qm_data.compute()
        logging.info("Dask compute finished")

    logging.info(f"EQM correction took {time.time() - start:.2f} seconds")

    # Check output shape and coords before saving
    logging.info(f"qm_data shape: {qm_data.shape}")
    logging.info(f"qm_data dims: {qm_data.dims}")
    logging.info(f"qm_data coords: {list(qm_data.coords)}")

    qm_ds = xr.Dataset(
        {"tmax": qm_data},
        coords=model_output.coords
    )
    qm_ds.to_netcdf(output_path)
    logging.info(f"Bias-corrected tmax saved to {output_path}")

# Profiler
if __name__ == "__main__":
    import cProfile
    import pstats
    profile_file = "EQM_allcells_progress.prof"
    cProfile.run('main()', profile_file)
    with open("EQM_allcells_progress_cumstat.txt", "w") as f:
        stats = pstats.Stats(profile_file, stream=f)
        stats.sort_stats("cumulative").print_stats(10)
    print("Profiling complete. Results saved to EQM_allcells_progress_cumstat.txt", flush=True)