import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import config
from joblib import Parallel, delayed
import argparse

#Not using SBCK here, for experimental purpises

model_path = f"{config.MODELS_DIR}/precip_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099/precip_r01_HR_masked.nc"
obs_path = f"{config.TARGET_DIR}/RhiresD_1971_2023.nc"
output_path = f"{config.BC_DIR}/qm_precip_r01_output.nc"

parser = argparse.ArgumentParser()
parser.add_argument('--n_jobs', type=int, default=1)
args = parser.parse_args()

print("Data")
model_output = xr.open_dataset(model_path, chunks={"time":2000})["precip"]
obs_output = xr.open_dataset(obs_path, chunks={"time":2000})["RhiresD"]

print("Calibration:1981-2010")
calib_obs = obs_output.sel(time=slice("1981-01-01", "2010-12-31"))
calib_mod = model_output.sel(time=slice("1981-01-01", "2010-12-31"))

quantiles = np.linspace(0, 1, 1001)
ntime, nlat, nlon = model_output.shape

zurich_lat = 47.3769
zurich_lon = 8.5417
lat_vals = model_output['lat'].values
lon_vals = model_output['lon'].values
i_zurich = np.argmin(np.abs(lat_vals - zurich_lat))
j_zurich = np.argmin(np.abs(lon_vals - zurich_lon))
plot_obs_q = plot_mod_q = None

qm_data = np.full(model_output.shape, np.nan, dtype=np.float32)

def process_lat(i):
    row = np.full((ntime, nlon), np.nan, dtype=np.float32)
    local_plot_obs_q = local_plot_mod_q = None
    for j in range(nlon):
        obs_valid = calib_obs[:, i, j].values[~np.isnan(calib_obs[:, i, j].values)]
        mod_valid = calib_mod[:, i, j].values[~np.isnan(calib_mod[:, i, j].values)]
        if obs_valid.size == 0 or mod_valid.size == 0:
            continue
        obs_q = np.quantile(obs_valid, quantiles)
        mod_q = np.quantile(mod_valid, quantiles)
        full_mod_series = model_output[:, i, j].values
        qm_series = np.interp(full_mod_series, mod_q, obs_q)
        row[:, j] = qm_series.astype(np.float32)
        if i == i_zurich and j == j_zurich:
            local_plot_obs_q = obs_q
            local_plot_mod_q = mod_q
    print(f"Processed latitude {i}/{nlat}")
    return i, row, local_plot_obs_q, local_plot_mod_q

results = Parallel(n_jobs=args.n_jobs)(delayed(process_lat)(i) for i in range(nlat))

for i, row, local_plot_obs_q, local_plot_mod_q in results:
    qm_data[:, i, :] = row
    if local_plot_obs_q is not None and local_plot_mod_q is not None:
        plot_obs_q = local_plot_obs_q
        plot_mod_q = local_plot_mod_q

print("Writing O/P")
qm_ds = xr.Dataset(
    {"precip": (model_output.dims, qm_data)},
    coords=model_output.coords
)
qm_ds.to_netcdf(output_path)
print(f"Output saved to {output_path}")

# Correction quantile prob fx 
if plot_obs_q is not None and plot_mod_q is not None:
    plt.figure(figsize=(7, 5))
    plt.plot(plot_mod_q, plot_obs_q, label="Correction function (obs vs model)")
    plt.plot(plot_mod_q, plot_mod_q, "--", color="gray", label="1:1 line")
    plt.xlabel("Model quantiles")
    plt.ylabel("Observed quantiles")
    plt.title(f"Quantile Mapping Correction Function\nZürich for Daily Accumulated Precipitation (lat={lat_vals[i_zurich]:.3f}, lon={lon_vals[j_zurich]:.3f})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUTS_MODELS_DIR}/qm_correction_function_precip_r01_zurich.png", dpi=500)