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

print("calib_mod_stack shape:", calib_mod_stack.shape)
print("calib_obs_stack shape:", calib_obs_stack.shape)
print("scenario_mod_stack shape:", scenario_mod_stack.shape)

n_features = len(var_names)
corrected_stack = np.full_like(scenario_mod_stack, np.nan)

calib_doys = xr.DataArray(calib_times).dt.dayofyear.values
scenario_doys = xr.DataArray(scenario_times).dt.dayofyear.values

for doy in range(1, 367):
    window_diffs = (calib_doys - doy + 366) % 366
    window_mask = (window_diffs <= 45) | (window_diffs >= (366 - 45))
    calib_mod_win = calib_mod_stack[window_mask]
    calib_obs_win = calib_obs_stack[window_mask]

    # scenario indices for this DOY
    scenario_mask = (scenario_doys == doy)
    scenario_mod_win = scenario_mod_stack[scenario_mask]

    if calib_mod_win.shape[0] == 0 or calib_obs_win.shape[0] == 0 or scenario_mod_win.shape[0] == 0:
        continue

    dotc = dOTC(bin_width=None, bin_origin=None) #Autoestimation bin size
    dotc.fit(calib_obs_win, calib_mod_win, scenario_mod_win)
    corrected_win = dotc.predict(scenario_mod_win)

    corrected_stack[scenario_mask] = corrected_win

output_path = f"{config.OUTPUTS_MODELS_DIR}/DOTC_{target_city}_4vars_corrected.nc"

coords = {
    "time": model_datasets[0].sel(time=slice(scenario_start, scenario_end))['time'].values,
    "lat": [lat_vals[i_city, j_city]],
    "lon": [lon_vals[i_city, j_city]]
}

data_vars = {
    var: (("time", "lat", "lon"), corrected_stack[:, idx].reshape(-1, 1, 1))
    for idx, var in enumerate(var_names)
}
ds_out = xr.Dataset(data_vars, coords=coords)
ds_out.to_netcdf(output_path)
print(f"Corrected output saved to {output_path}")



for idx, var in enumerate(var_names):
    plt.figure(figsize=(8, 6))
    model_vals = scenario_mod_stack[:, idx][~np.isnan(scenario_mod_stack[:, idx])]
    obs_vals = obs_datasets[idx].sel(time=slice(calib_start, calib_end))[:, i_city, j_city].values
    obs_vals = obs_vals[~np.isnan(obs_vals)]
    corr_vals = corrected_stack[:, idx][~np.isnan(corrected_stack[:, idx])]

    # Wasserstein 
    emd_model = scipy.stats.wasserstein_distance(obs_vals, model_vals)
    emd_corr = scipy.stats.wasserstein_distance(obs_vals, corr_vals)

    for vals, label, color in [
        (model_vals, f"Model (Coarse) [Wasserstein={emd_model:.3f}]", "blue"),
        (obs_vals, "Observations", "green"),
        (corr_vals, f"Corrected Output [Wasserstein={emd_corr:.3f}]", "red")
    ]:
        sorted_vals = np.sort(vals)
        cdf = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)
        plt.plot(sorted_vals, cdf, label=label, color=color)

    plt.xlabel("Mean Temperature (°C)" if var == "temp" else
                "Precipitation (mm/day)" if var == "precip" else
                "Minimum Temperature (°C)" if var == "tmin" else
               "Maximum Temperature (°C)")
    
    plt.ylabel("CDF")
    plt.title(f"CDFs for {target_city} - {var}: DOTC BC")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    cdf_plot_path = output_path.replace(".nc", f"_cdf_{var}.png")
    plt.savefig(cdf_plot_path, dpi=1000)
    print(f"CDF plot saved to {cdf_plot_path}")