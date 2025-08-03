from pyexpat import model
import dask
import dask.array as da
import xarray as xr
import numpy as np
import config
from dask.diagnostics import ProgressBar
from dask.distributed import Client

def eqm_cell(model_cell, obs_cell, calib_start, calib_end, model_times, obs_times):
    # Return all-NaN if input is invalid
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

    quantiles = np.linspace(0.01, 0.99, 99)  # 99quantiles

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
            # Linear interp between 1st and 99th
            corrected = np.interp(values, mod_q, obs_q, left=obs_q[0], right=obs_q[-1])
            qm_series[indices] = corrected
    return qm_series


def main():
    client = Client(n_workers=4) 
    print(f"Dashboard: {client.dashboard_link}")

    model_path = f"{config.SCRATCH_DIR}/tmax_r01_HR_masked.nc"
    obs_path = f"{config.SCRATCH_DIR}/TmaxD_1971_2023.nc"
    output_path = f"{config.BIAS_CORRECTED_DIR}/EQM/eqm_tmax_r01_allcells_DOY.nc"


    # Open with chunking for Dask
    model_ds = xr.open_dataset(model_path)
    obs_ds = xr.open_dataset(obs_path)
    model = model_ds["tmax"].sel(time=slice("1981-01-01", "2010-12-31")).chunk({"time": -1, "N": 10, "E": 10})
    obs = obs_ds["TmaxD"].sel(time=slice("1981-01-01", "2010-12-31")).chunk({"time": -1, "N": 10, "E": 10})
    print("model variable dims:", model.dims)
    print("obs variable dims:", obs.dims)

    # After slicing model and obs:
    if not np.array_equal(model['time'].values, obs['time'].values):
        print("WARNING: Forcing obs time axis to match model time axis for alignment. Indexing different but time alignment works")
        obs = obs.assign_coords(time=model['time'])

    # Dims
    spatial_dims = [d for d in model.dims if d != "time"]

    # UFunc for parallelization
    qm_da = xr.apply_ufunc(
        eqm_cell,
        model,
        obs,
        input_core_dims=[['time'], ['time']],
        output_core_dims=[['time']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.float32],
        kwargs={
            'calib_start': np.datetime64("1981-01-01"),
            'calib_end': np.datetime64("2010-12-31"),
            'model_times': model['time'].values,
            'obs_times': obs['time'].values
        }
    )

    with ProgressBar():
        qm_da = qm_da.compute()
    print("qm_da dims:", qm_da.dims)
    print("qm_da shape:", qm_da.shape)
    print("qm_da coords:", qm_da.coords)

    qm_ds = xr.Dataset({"tmax": qm_da})
    qm_ds.to_netcdf(output_path)
    print(f"Bias-corrected tmax saved to {output_path}")

if __name__ == "__main__":
    main()
    print("Running EQM bias correction for all cells...")
    print("Done.")