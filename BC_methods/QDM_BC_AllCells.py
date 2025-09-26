import xarray as xr
import numpy as np
import config
from SBCK import QDM
from joblib import Parallel, delayed

var_names = ["temp", "precip", "tmin", "tmax"]
obs_var_names = ["TabsD", "RhiresD", "TminD", "TmaxD"]

model_paths = [
    f"{config.MODELS_DIR}/temp_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099/temp_r01_coarse_masked.nc",
    f"{config.MODELS_DIR}/precip_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099/precip_r01_coarse_masked.nc",
    f"{config.MODELS_DIR}/tmin_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099/tmin_r01_coarse_masked.nc",
    f"{config.MODELS_DIR}/tmax_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099/tmax_r01_coarse_masked.nc"
]

obs_paths = [
    f"{config.DATASETS_TRAINING_DIR}/TabsD_step2_coarse.nc",
    f"{config.DATASETS_TRAINING_DIR}/RhiresD_step2_coarse.nc",
    f"{config.DATASETS_TRAINING_DIR}/TminD_step2_coarse.nc",
    f"{config.DATASETS_TRAINING_DIR}/TmaxD_step2_coarse.nc"
]

model_datasets = [xr.open_dataset(p)[vn] for p, vn in zip(model_paths, var_names)]
obs_datasets = [xr.open_dataset(p)[ovn] for p, ovn in zip(obs_paths, obs_var_names)]

ntime, nN, nE = model_datasets[0].shape
nvars = len(var_names)

# Use xarray's built-in dayofyear instead of custom function
model_times = model_datasets[0]['time']
obs_times = obs_datasets[0]['time']
model_doys = model_times.dt.dayofyear.values
obs_doys = obs_times.dt.dayofyear.values

def process_cell(i, j):
    # Extract data for this cell using xarray indexing
    full_mod_cells = [ds.isel(lat=i, lon=j).values for ds in model_datasets]
    full_mod_stack = np.stack(full_mod_cells, axis=1)
    full_times = model_times.values
    full_doys = model_doys
    
    # Check for too many NaNs
    nan_fraction = np.isnan(full_mod_stack).sum() / full_mod_stack.size
    if nan_fraction > 0.5:
        return np.full_like(full_mod_stack, np.nan)
    
    # Use xarray for time slicing
    calib_mod_cells = [ds.sel(time=slice("1981-01-01", "2010-12-31")).isel(lat=i, lon=j).values for ds in model_datasets]
    calib_obs_cells = [ds.sel(time=slice("1981-01-01", "2010-12-31")).isel(lat=i, lon=j).values for ds in obs_datasets]
    calib_times = model_datasets[0].sel(time=slice("1981-01-01", "2010-12-31"))['time']
    
    calib_mod_stack = np.stack(calib_mod_cells, axis=1)
    calib_obs_stack = np.stack(calib_obs_cells, axis=1)
    calib_doys = calib_times.dt.dayofyear.values

    # Clip precipitation values
    if "precip" in var_names:
        precip_idx = var_names.index("precip")
        calib_obs_stack[:, precip_idx] = np.clip(calib_obs_stack[:, precip_idx], 0, None)
        calib_mod_stack[:, precip_idx] = np.clip(calib_mod_stack[:, precip_idx], 0, None)
        full_mod_stack[:, precip_idx] = np.clip(full_mod_stack[:, precip_idx], 0, None)

    full_corrected_stack = np.full_like(full_mod_stack, np.nan)

    # Process each day of year
    for doy in range(1, 367):
        # Create window mask for calibration period
        window_diffs = (calib_doys - doy + 366) % 366
        window_mask = (window_diffs <= 45) | (window_diffs >= (366 - 45))
        calib_mod_win = calib_mod_stack[window_mask]
        calib_obs_win = calib_obs_stack[window_mask]
        
        # Find days in full period matching this DOY
        full_mask = (full_doys == doy)
        full_mod_win_for_pred = full_mod_stack[full_mask]

        if calib_mod_win.shape[0] == 0 or calib_obs_win.shape[0] == 0 or full_mod_win_for_pred.shape[0] == 0:
            continue

        valid_mask_mod = ~np.any(np.isnan(calib_mod_win), axis=1)
        valid_mask_obs = ~np.any(np.isnan(calib_obs_win), axis=1)
        valid_mask_pred = ~np.any(np.isnan(full_mod_win_for_pred), axis=1)
        
        calib_mod_win_clean = calib_mod_win[valid_mask_mod]
        calib_obs_win_clean = calib_obs_win[valid_mask_obs]
        full_mod_win_clean = full_mod_win_for_pred[valid_mask_pred]
        
        if (calib_mod_win_clean.shape[0] < 10 or 
            calib_obs_win_clean.shape[0] < 10 or 
            full_mod_win_clean.shape[0] == 0):
            continue
            
        corrected_full = np.full_like(full_mod_win_clean, np.nan)
        
        for var_idx in range(nvars):
            calib_mod_var = calib_mod_win_clean[:, var_idx].reshape(-1, 1)
            calib_obs_var = calib_obs_win_clean[:, var_idx].reshape(-1, 1)
            full_mod_var = full_mod_win_clean[:, var_idx].reshape(-1, 1)
            
            if (np.all(calib_mod_var == calib_mod_var[0]) or 
                np.all(calib_obs_var == calib_obs_var[0]) or
                np.std(calib_mod_var) < 1e-10 or
                np.std(calib_obs_var) < 1e-10):
                mean_diff = np.nanmean(calib_obs_var) - np.nanmean(calib_mod_var)
                corrected_full[:, var_idx] = full_mod_var.flatten() + mean_diff
            else:
                try:
                    delta_method = "multiplicative" if var_names[var_idx] == "precip" else "additive"
                    
                    qdm = QDM(bin_width=None, bin_origin=None, delta=delta_method)
                    qdm.fit(calib_obs_var, calib_mod_var, full_mod_var)
                    corrected_var = qdm.predict(full_mod_var)
                    corrected_full[:, var_idx] = corrected_var.flatten()
                    
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                    print(f"QDM failed for cell ({i},{j}), DOY {doy}, var {var_names[var_idx]}: {e}. Using mean adjustment.")
                    mean_diff = np.nanmean(calib_obs_var) - np.nanmean(calib_mod_var)
                    corrected_full[:, var_idx] = full_mod_var.flatten() + mean_diff

        if "precip" in var_names:
            precip_idx = var_names.index("precip")
            corrected_full[:, precip_idx] = np.clip(corrected_full[:, precip_idx], 0, None)

        full_corrected_indices = np.where(full_mask)[0][valid_mask_pred]
        full_corrected_stack[full_corrected_indices] = corrected_full

    return full_corrected_stack


print("Starting gridwise QDM correction...")
results = Parallel(n_jobs=8)(
    delayed(process_cell)(i, j)
    for i in range(nN) for j in range(nE)
)

corrected_data = {var: np.full((ntime, nN, nE), np.nan, dtype=np.float32) for var in var_names}

idx = 0
for i in range(nN):
    for j in range(nE):
        result = results[idx]
        if result is not None and not np.all(np.isnan(result)):
            for v, var in enumerate(var_names):
                corrected_data[var][:, i, j] = result[:, v]
        idx += 1

coords = {
    "time": model_times.values,
    "lat": (("lat", "lon"), model_datasets[0]['lat'].values),
    "lon": (("lat", "lon"), model_datasets[0]['lon'].values)
}

data_vars = {
    var: (("time", "lat", "lon"), corrected_data[var])
    for var in var_names
}

ds_out = xr.Dataset(data_vars, coords=coords)
output_path = f"{config.BIAS_CORRECTED_DIR}/QDM/QDM_BC_AllCells_4vars.nc"
ds_out.to_netcdf(output_path)
print(f"Bias-corrected data saved to {output_path}")