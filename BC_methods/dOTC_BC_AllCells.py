import xarray as xr
import numpy as np
import config
from SBCK import dOTC
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

# Prepare time and DOY arrays
model_times = model_datasets[0]['time'].values
obs_times = obs_datasets[0]['time'].values
def get_doy(d): return (np.datetime64(d, 'D') - np.datetime64(str(d)[:4] + '-01-01', 'D')).astype(int) + 1
model_doys = np.array([get_doy(d) for d in model_times])
obs_doys = np.array([get_doy(d) for d in obs_times])

calib_start = np.datetime64("1981-01-01")
calib_end = np.datetime64("2010-12-31")
scenario_start = np.datetime64("2011-01-01")
scenario_end = np.datetime64("2099-12-31")

def process_cell(i, j):
    # Extract FULL time series for this cell (1971-2099)
    full_mod_cells = [ds[:, i, j].values for ds in model_datasets]  # Full period
    full_mod_stack = np.stack(full_mod_cells, axis=1)
    full_times = model_times
    full_doys = np.array([get_doy(d) for d in full_times])
    
    # Extract CALIBRATION period only for training
    calib_mod_cells = [ds.sel(time=slice("1981-01-01", "2010-12-31"))[:, i, j].values for ds in model_datasets]
    calib_obs_cells = [ds.sel(time=slice("1981-01-01", "2010-12-31"))[:, i, j].values for ds in obs_datasets]
    calib_times = model_datasets[0].sel(time=slice("1981-01-01", "2010-12-31"))['time'].values
    
    calib_mod_stack = np.stack(calib_mod_cells, axis=1)
    calib_obs_stack = np.stack(calib_obs_cells, axis=1)
    calib_doys = np.array([get_doy(d) for d in calib_times])

    # Precip clipping
    if "precip" in var_names:
        precip_idx = var_names.index("precip")
        calib_obs_stack[:, precip_idx] = np.clip(calib_obs_stack[:, precip_idx], 0, None)
        calib_mod_stack[:, precip_idx] = np.clip(calib_mod_stack[:, precip_idx], 0, None)
        full_mod_stack[:, precip_idx] = np.clip(full_mod_stack[:, precip_idx], 0, None)

    # Apply correction to FULL period (1971-2099)
    full_corrected_stack = np.full_like(full_mod_stack, np.nan)

    for doy in range(1, 367):
        # Use calibration window for training
        window_diffs = (calib_doys - doy + 366) % 366
        window_mask = (window_diffs <= 45) | (window_diffs >= (366 - 45))
        calib_mod_win = calib_mod_stack[window_mask]
        calib_obs_win = calib_obs_stack[window_mask]
        
        # Apply to full period for prediction
        full_mask = (full_doys == doy)
        full_mod_win_for_pred = full_mod_stack[full_mask]

        if calib_mod_win.shape[0] == 0 or calib_obs_win.shape[0] == 0 or full_mod_win_for_pred.shape[0] == 0:
            continue

        dotc = dOTC(bin_width=None, bin_origin=None)
        dotc.fit(calib_obs_win, calib_mod_win, full_mod_win_for_pred)
        corrected_full = dotc.predict(full_mod_win_for_pred)

        if "precip" in var_names:
            corrected_full[:, precip_idx] = np.clip(corrected_full[:, precip_idx], 0, None)

        full_corrected_stack[full_mask] = corrected_full

    return full_corrected_stack


print("Starting gridwise dOTC correction...")
results = Parallel(n_jobs=8)(
    delayed(process_cell)(i, j)
    for i in range(nN) for j in range(nE)
)

# Recon
corrected_data = {var: np.full((ntime, nN, nE), np.nan, dtype=np.float32) for var in var_names}
idx = 0
for i in range(nN):
    for j in range(nE):
        for v, var in enumerate(var_names):
            corrected_data[var][:, i, j] = results[idx][:, v]
        idx += 1

coords = {
    "time": model_times,
    "lat": (("lat", "lon"), model_datasets[0]['lat'].values), #2D
    "lon": (("lat", "lon"), model_datasets[0]['lon'].values)
}

data_vars = {
    var: (("time", "lat", "lon"), corrected_data[var])
    for var in var_names
}
ds_out = xr.Dataset(data_vars, coords=coords)
output_path = f"{config.BIAS_CORRECTED_DIR}/dOTC/dOTC_BC_AllCells_4vars.nc"
ds_out.to_netcdf(output_path)
print(f"Bias-corrected data saved to {output_path}")