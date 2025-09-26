import xarray as xr
import numpy as np
import config
from SBCK import QDM
from joblib import Parallel, delayed

def qdm_cell(model_cell, obs_cell, calib_start, calib_end, model_times, obs_times, var_name):
    ntime = model_cell.shape[0]
    qdm_series = np.full(ntime, np.nan, dtype=np.float32)
    if ntime == 0 or np.all(np.isnan(model_cell)) or np.all(np.isnan(obs_cell)):
        return qdm_series

    model_dates = np.array(model_times, dtype='datetime64[D]')
    obs_dates = np.array(obs_times, dtype='datetime64[D]')

    calib_dates = np.arange(calib_start, calib_end + np.timedelta64(1, 'D'), dtype='datetime64[D]')
    model_calib_idx = np.in1d(model_dates, calib_dates)
    obs_calib_idx = np.in1d(obs_dates, calib_dates)

    common_dates = np.intersect1d(model_dates[model_calib_idx], obs_dates[obs_calib_idx])
    if len(common_dates) == 0:
        return qdm_series

    model_common_idx = np.in1d(model_dates, common_dates)
    obs_common_idx = np.in1d(obs_dates, common_dates)

    calib_mod_cell = model_cell[model_common_idx]
    calib_obs_cell = obs_cell[obs_common_idx]

    # Clip precipitation to non-negative values
    if var_name == "precip":
        calib_mod_cell = np.clip(calib_mod_cell, 0, None)
        calib_obs_cell = np.clip(calib_obs_cell, 0, None)
        model_cell = np.clip(model_cell, 0, None)

    def get_doy(d): return (np.datetime64(d, 'D') - np.datetime64(str(d)[:4] + '-01-01', 'D')).astype(int) + 1
    calib_doys = np.array([get_doy(d) for d in common_dates])
    model_doys = np.array([get_doy(d) for d in model_dates])

    for doy in range(1, 367):
        window_doys = ((calib_doys - doy + 366) % 366)
        window_mask = (window_doys <= 45) | (window_doys >= (366 - 45))
        obs_window = calib_obs_cell[window_mask]
        mod_window = calib_mod_cell[window_mask]
        
        # Remove NaN values
        obs_window = obs_window[~np.isnan(obs_window)]
        mod_window = mod_window[~np.isnan(mod_window)]
        
        if obs_window.size < 10 or mod_window.size < 10:
            continue

        # Find indices for this DOY in the full model time series
        indices = np.where(model_doys == doy)[0]
        if indices.size == 0:
            continue

        values = model_cell[indices]
        valid_values_mask = ~np.isnan(values)
        valid_indices = indices[valid_values_mask]
        valid_values = values[valid_values_mask]
        
        if valid_values.size == 0:
            continue

        try:
            # Set delta method based on variable
            delta_method = "multiplicative" if var_name == "precip" else "additive"
            
            # Check for zero variance
            if np.std(mod_window) < 1e-10 or np.std(obs_window) < 1e-10:
                # Use mean adjustment
                mean_diff = np.nanmean(obs_window) - np.nanmean(mod_window)
                corrected = valid_values + mean_diff
            else:
                # Apply QDM
                qdm = QDM(delta=delta_method)
                qdm.fit(obs_window.reshape(-1, 1), mod_window.reshape(-1, 1), valid_values.reshape(-1, 1))
                corrected = qdm.predict(valid_values.reshape(-1, 1)).flatten()
            
            # Clip precipitation results
            if var_name == "precip":
                corrected = np.clip(corrected, 0, None)
            
            qdm_series[valid_indices] = corrected
            
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            print(f"QDM failed for DOY {doy}, var {var_name}: {e}. Using mean adjustment.")
            mean_diff = np.nanmean(obs_window) - np.nanmean(mod_window)
            corrected = valid_values + mean_diff
            if var_name == "precip":
                corrected = np.clip(corrected, 0, None)
            qdm_series[valid_indices] = corrected

    return qdm_series

def process_variable(var_name, obs_var_name):
    print(f"Processing {var_name}...")
    
    model_path = f"{config.MODELS_DIR}/{var_name}_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099/{var_name}_r01_coarse_masked.nc"
    obs_path = f"{config.DATASETS_TRAINING_DIR}/{obs_var_name}_step2_coarse.nc"
    output_path = f"{config.BIAS_CORRECTED_DIR}/QDM/{var_name}_QDM_BC_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099_r01.nc"

    model_ds = xr.open_dataset(model_path)
    obs_ds = xr.open_dataset(obs_path)
    model = model_ds[var_name]
    obs = obs_ds[obs_var_name]

    ntime, nN, nE = model.shape
    qdm_data = np.full(model.shape, np.nan, dtype=np.float32)

    # Parallelising over grid
    def process_cell(i, j):
        model_cell = model[:, i, j].values
        obs_cell = obs[:, i, j].values
        return qdm_cell(
            model_cell, obs_cell,
            np.datetime64("1981-01-01"), np.datetime64("2010-12-31"),
            model['time'].values, obs['time'].values, var_name
        )

    print(f"Starting gridwise QDM correction for {var_name}...")
    results = Parallel(n_jobs=8)(
        delayed(process_cell)(i, j)
        for i in range(nN) for j in range(nE)
    )

    idx = 0
    for i in range(nN):
        for j in range(nE):
            qdm_data[:, i, j] = results[idx]
            idx += 1

    out_ds = model_ds.copy()
    out_ds[var_name] = (("time", "N", "E"), qdm_data)
    out_ds.to_netcdf(output_path)
    print(f"Bias-corrected {var_name} saved to {output_path}")

def main():
    print("QDM for All Cells started")
    
    var_names = ["temp", "precip", "tmin", "tmax"]
    obs_var_names = ["TabsD", "RhiresD", "TminD", "TmaxD"]
    
    for var_name, obs_var_name in zip(var_names, obs_var_names):
        process_variable(var_name, obs_var_name)
    
    print("QDM for All Cells finished")

if __name__ == "__main__":
    main()