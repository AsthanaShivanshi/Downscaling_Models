import xarray as xr
import numpy as np
import config
from SBCK import QM
from joblib import Parallel, delayed

def eqm_cell(model_cell, obs_cell, calib_start, calib_end, model_times, obs_times):
    ntime = model_cell.shape[0]
    qm_series = np.full(ntime, np.nan, dtype=np.float32)
    if ntime == 0 or np.all(np.isnan(model_cell)) or np.all(np.isnan(obs_cell)):
        return qm_series

    # Convert to date only (ignore time-of-day)
    model_dates = np.array(model_times, dtype='datetime64[D]')
    obs_dates = np.array(obs_times, dtype='datetime64[D]')

    # Calibration period mask for both
    calib_dates = np.arange(calib_start, calib_end + np.timedelta64(1, 'D'), dtype='datetime64[D]')
    model_calib_idx = np.in1d(model_dates, calib_dates)
    obs_calib_idx = np.in1d(obs_dates, calib_dates)

    # Only keeping overlapping dates during calibration
    common_dates = np.intersect1d(model_dates[model_calib_idx], obs_dates[obs_calib_idx])
    if len(common_dates) == 0:
        return qm_series

    model_common_idx = np.in1d(model_dates, common_dates)
    obs_common_idx = np.in1d(obs_dates, common_dates)

    calib_mod_cell = model_cell[model_common_idx]
    calib_obs_cell = obs_cell[obs_common_idx]

    # DOY for calibration and full model period
    def get_doy(d): return (np.datetime64(d, 'D') - np.datetime64(str(d)[:4] + '-01-01', 'D')).astype(int) + 1
    calib_doys = np.array([get_doy(d) for d in common_dates])
    model_doys = np.array([get_doy(d) for d in model_dates])

    for doy in range(1, 367):
        # 91-d with wraparound
        window_doys = ((calib_doys - doy + 366) % 366)
        window_mask = (window_doys <= 45) | (window_doys >= (366 - 45))
        obs_window = calib_obs_cell[window_mask]
        mod_window = calib_mod_cell[window_mask]
        obs_window = obs_window[~np.isnan(obs_window)]
        mod_window = mod_window[~np.isnan(mod_window)]
        if obs_window.size == 0 or mod_window.size == 0:
            continue

        #quantile res: 99 with linear interp in between
        quantiles= np.linspace(0.01,0.99,99)
        mod_q= np.quantile(mod_window, quantiles)
        obs_q= np.quantile(obs_window, quantiles)
        eqm = QM()
        eqm.fit(mod_window.reshape(-1, 1), obs_window.reshape(-1, 1))
        indices = np.where(model_doys == doy)[0]
        if indices.size > 0:
            values = model_cell[indices]
            mapped= eqm.predict(values.reshape(-1, 1)).flatten()

            #Flat extrapolation for tails
            p1_mod=mod_q[0]
            p99_obs=obs_q[-1]
            p1_obs=obs_q[0]
            p99_mod=mod_q[-1]

            mapped[values < p1_mod] = p1_obs
            mapped[values > p99_mod] = p99_obs
            qm_series[indices] = mapped

    return qm_series

def main():
    print("EQM for All Cells started")

    model_path = f"{config.MODELS_DIR}/temp_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099/temp_r01_coarse_masked.nc" 
    obs_path = f"{config.DATASETS_TRAINING_DIR}/TabsD_step2_coarse.nc"
    output_path = f"{config.BIAS_CORRECTED_DIR}/EQM/eqm_temp_r01_allcells.nc"

    model_ds = xr.open_dataset(model_path)
    obs_ds = xr.open_dataset(obs_path)
    model = model_ds["temp"]
    obs = obs_ds["TabsD"]

    ntime, nN, nE = model.shape
    qm_data = np.full(model.shape, np.nan, dtype=np.float32)

    # Parallelising over grid
    def process_cell(i, j):
        model_cell = model[:, i, j].values
        obs_cell = obs[:, i, j].values
        return eqm_cell(
            model_cell, obs_cell,
            np.datetime64("1981-01-01"), np.datetime64("2010-12-31"),
            model['time'].values, obs['time'].values
        )

    print("Starting gridwise EQM correction...")
    results = Parallel(n_jobs=8)(
        delayed(process_cell)(i, j)
        for i in range(nN) for j in range(nE)
    )

    idx = 0
    for i in range(nN):
        for j in range(nE):
            qm_data[:, i, j] = results[idx]
            idx += 1

    out_ds = model_ds.copy()
    out_ds["temp"] = (("time", "N", "E"), qm_data)
    out_ds.to_netcdf(output_path)
    print(f"Bias-corrected temp saved to {output_path}")
    print("EQM for All Cells finished")

if __name__ == "__main__":
    main()