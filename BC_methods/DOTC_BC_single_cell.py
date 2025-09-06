import scipy
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from SBCK import dOTC 
import config
import argparse
import scipy.stats

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

parser = argparse.ArgumentParser()
parser.add_argument("--city", type=str, required=True, help="City name, first letter upper case")
parser.add_argument("--lat", type=float, required=True, help="City lat")
parser.add_argument("--lon", type=float, required=True, help="City lon")
args = parser.parse_args()

target_city = args.city
target_lat = args.lat
target_lon = args.lon

locations = {target_city: (target_lat, target_lon)}

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

var_names = ["temp", "precip", "tmin", "tmax"]
obs_var_names = ["TabsD", "RhiresD", "TminD", "TmaxD"]

model_datasets = [xr.open_dataset(p)[vn] for p, vn in zip(model_paths, var_names)]
obs_datasets = [xr.open_dataset(p)[ovn] for p, ovn in zip(obs_paths, obs_var_names)]

lat_vals = model_datasets[0]['lat'].values
lon_vals = model_datasets[0]['lon'].values

dist = np.sqrt((lat_vals - target_lat)**2 + (lon_vals - target_lon)**2)

#Grid cell : Euclidean
i_city, j_city = np.unravel_index(np.argmin(dist), dist.shape)
print(f"Closest grid cell to {target_city}: i={i_city}, j={j_city}")
print(f"Location: lat={lat_vals[i_city, j_city]}, lon={lon_vals[i_city, j_city]}")


calib_start = "1981-01-01"
calib_end = "2010-12-31"
scenario_start = "2011-01-01"
scenario_end = "2099-12-31"


calib_mod_cells = [ds.sel(time=slice(calib_start, calib_end))[:, i_city, j_city].values for ds in model_datasets]
calib_obs_cells = [ds.sel(time=slice(calib_start, calib_end))[:, i_city, j_city].values for ds in obs_datasets]
calib_times = model_datasets[0].sel(time=slice(calib_start, calib_end))['time'].values

scenario_mod_cells = [ds.sel(time=slice(scenario_start, scenario_end))[:, i_city, j_city].values for ds in model_datasets]
scenario_times = model_datasets[0].sel(time=slice(scenario_start, scenario_end))['time'].values

calib_mod_stack = np.stack(calib_mod_cells, axis=1)
calib_obs_stack = np.stack(calib_obs_cells, axis=1)
scenario_mod_stack = np.stack(scenario_mod_cells, axis=1)

full_mod_stack = np.concatenate([calib_mod_stack, scenario_mod_stack], axis=0)
full_times = np.concatenate([calib_times, scenario_times])
full_doys = xr.DataArray(full_times).dt.dayofyear.values
calib_doys = xr.DataArray(calib_times).dt.dayofyear.values

full_corrected_stack = np.full_like(full_mod_stack, np.nan) #Nan array for storing corrected values

# Apply dOTC for doy
for doy in range(1, 367):
    window_diffs = (calib_doys - doy + 366) % 366
    window_mask = (window_diffs <= 45) | (window_diffs >= (366 - 45))
    calib_mod_win = calib_mod_stack[window_mask]
    calib_obs_win = calib_obs_stack[window_mask]
    full_mask = (full_doys == doy)
    full_mod_win_for_pred = full_mod_stack[full_mask]

    if calib_mod_win.shape[0] == 0 or calib_obs_win.shape[0] == 0 or full_mod_win_for_pred.shape[0] == 0:
        continue

    dotc = dOTC(bin_width=None, bin_origin=None)
    dotc.fit(calib_obs_win, calib_mod_win, full_mod_win_for_pred)
    corrected_full = dotc.predict(full_mod_win_for_pred) #not only the 2011-2099 scenario period, corr applied to full period
    full_corrected_stack[full_mask] = corrected_full

coords = {
    "time": full_times,
    "lat": [lat_vals[i_city, j_city]],
    "lon": [lon_vals[i_city, j_city]]
}


data_vars = {
    var: (("time", "lat", "lon"), full_corrected_stack[:, idx].reshape(-1, 1, 1))
    for idx, var in enumerate(var_names)
}


ds_out = xr.Dataset(data_vars, coords=coords)
output_path = f"{config.OUTPUTS_MODELS_DIR}/DOTC_{target_city}_4vars_corrected.nc"
ds_out.to_netcdf(output_path)



# CDFs, calib and scenario
for idx, var in enumerate(var_names):
    # 1981-2010
    model_vals_calib = calib_mod_stack[:, idx]
    obs_vals_calib = calib_obs_stack[:, idx]
    corr_vals_calib = full_corrected_stack[:calib_mod_stack.shape[0], idx]

    # 2011-2099
    scenario_model_vals = scenario_mod_stack[:, idx]
    scenario_corr_vals = full_corrected_stack[calib_mod_stack.shape[0]:, idx]
    model_vals_calib = model_vals_calib[~np.isnan(model_vals_calib)]
    obs_vals_calib = obs_vals_calib[~np.isnan(obs_vals_calib)]
    corr_vals_calib = corr_vals_calib[~np.isnan(corr_vals_calib)]
    scenario_model_vals = scenario_model_vals[~np.isnan(scenario_model_vals)]
    scenario_corr_vals = scenario_corr_vals[~np.isnan(scenario_corr_vals)]

    # KS
    ks_model_calib = scipy.stats.ks_2samp(obs_vals_calib, model_vals_calib)
    ks_corr_calib = scipy.stats.ks_2samp(obs_vals_calib, corr_vals_calib)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: 1981-2010
    for vals, label, color in [
        (model_vals_calib, f"Model (Coarse,1981-2010) [KS={ks_model_calib.statistic:.3f}]", "red"),
        (obs_vals_calib, "Observations (1981-2010)", "black"),
        (corr_vals_calib, f"Corrected Output (1981-2010) [KS={ks_corr_calib.statistic:.3f}]", "green")
    ]:
        if len(vals) == 0:
            continue
        sorted_vals = np.sort(vals)
        cdf = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)
        axes[0].plot(sorted_vals, cdf, label=label, color=color)

    axes[0].set_xlabel("Mean Temperature (°C)" if var == "temp" else
                       "Precipitation (mm/day)" if var == "precip" else
                       "Minimum Temperature (°C)" if var == "tmin" else
                       "Maximum Temperature (°C)")
    axes[0].set_ylabel("CDF")
    axes[0].set_title(f"CDFs (calibration period) for {target_city} - {var}: dOTC BC")
    axes[0].legend()
    axes[0].grid(True)

    # Right: Scenario
    for vals, label, color in [
        (scenario_model_vals, f"Model (Coarse, 2011-2099)", "red"),
        (obs_vals_calib, "Observations (1981-2010)", "black"),
        (scenario_corr_vals, f"Corrected Output (2011-2099)", "green")
    ]:
        if len(vals) == 0:
            continue
        sorted_vals = np.sort(vals)
        cdf = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)
        axes[1].plot(sorted_vals, cdf, label=label, color=color)

    axes[1].set_xlabel("Mean Temperature (°C)" if var == "temp" else
                       "Precipitation (mm/day)" if var == "precip" else
                       "Minimum Temperature (°C)" if var == "tmin" else
                       "Maximum Temperature (°C)")
    axes[1].set_ylabel("CDF")
    axes[1].set_title(f"CDFs (scenario period) for {target_city} - {var}: dOTC BC")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    cdf_plot_path = output_path.replace(".nc", f"_cdf_twopanel_{var}.png")
    plt.savefig(cdf_plot_path, dpi=1000)
    print(f"Two-panel CDF plot for {var} saved to {cdf_plot_path}")