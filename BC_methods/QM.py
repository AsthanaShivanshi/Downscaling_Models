import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import config

# --- File paths ---
model_path = f"{config.MODELS_DIR}/precip_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099/precip_r01_HR_masked.nc"
obs_path = f"{config.TARGET_DIR}/RhiresD_1971_2023.nc"
output_path = f"{config.BC_DIR}/qm_output.nc"
plot_path = f"{config.OUTPUTS_MODELS_DIR}/qm_correction_plot.png"

print("Data loading")
model_output = xr.open_dataset(model_path)["precip"]
obs_output = xr.open_dataset(obs_path)["RhiresD"]

# Calibration
print("Selecting calibration period...")
calib_obs = obs_output.sel(time=slice("1981-01-01", "2010-12-31"))
calib_mod = model_output.sel(time=slice("1981-01-01", "2010-12-31"))

quantiles = np.linspace(0, 1, 1001)
ntime, nlat, nlon = model_output.shape
chunk_size = 100

print("Preparing output file...")
qm_ds = xr.Dataset(
    {"precip": (model_output.dims, np.full(model_output.shape, np.nan, dtype=np.float32))},
    coords=model_output.coords
)
qm_ds.to_netcdf(output_path)
del qm_ds

# --- Pick a random grid cell for plotting ---
i_rand = np.random.randint(0, nlat)
j_rand = np.random.randint(0, nlon)
plot_obs_q = plot_mod_q = None

# --- Chunked processing and writing ---
print("Starting chunked EQM processing...")
with xr.open_dataset(output_path, mode="r+") as ds_out:
    for i_start in range(0, nlat, chunk_size):
        i_end = min(i_start + chunk_size, nlat)
        print(f"Processing latitudes {i_start} to {i_end-1}")
        for i in range(i_start, i_end):
            for j in range(nlon):
                obs_series = calib_obs[:, i, j].values
                mod_series = calib_mod[:, i, j].values
                obs_valid = obs_series[~np.isnan(obs_series)]
                mod_valid = mod_series[~np.isnan(mod_series)]
                if len(obs_valid) < 10 or len(mod_valid) < 10:
                    continue
                obs_q = np.quantile(obs_valid, quantiles)
                mod_q = np.quantile(mod_valid, quantiles)
                full_mod_series = model_output[:, i, j].values
                qm_series = np.interp(full_mod_series, mod_q, obs_q)
                ds_out["precip_qmapped"][:, i, j] = qm_series.astype(np.float32)

                # Save for plotting if this is the random cell
                if i == i_rand and j == j_rand:
                    plot_obs_q = obs_q
                    plot_mod_q = mod_q

print("EQM processing complete. Output saved to:", output_path)

# --- Plot the correction quantile probability function for the random grid cell ---
if plot_obs_q is not None and plot_mod_q is not None:
    plt.figure(figsize=(7,5))
    plt.plot(plot_mod_q, plot_obs_q, label="Correction function (obs vs model)")
    plt.plot(plot_mod_q, plot_mod_q, "--", color="gray", label="1:1 line")
    plt.xlabel("Model quantiles (calib period)")
    plt.ylabel("Observed quantiles (calib period)")
    plt.title(f"Quantile Mapping Correction Function\nGrid cell (i={i_rand}, j={j_rand})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print("Correction plot saved to:", plot_path)
else:
    print("Random grid cell had insufficient data for plotting.")