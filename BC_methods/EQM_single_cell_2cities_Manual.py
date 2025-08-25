import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from SBCK import QM
import config

#Fontsize and name specs
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 18,
    "axes.labelsize": 22,
    "axes.titlesize": 24,
    "legend.fontsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
})

model_path = f"{config.SCRATCH_DIR}/tmax_r01_HR_masked.nc"
obs_path = f"{config.SCRATCH_DIR}/TmaxD_1971_2023.nc"
output_path_template = f"{config.BC_DIR}/qm_tmax_r01_singlecell_{{city}}_output.nc"
plot_path = f"{config.OUTPUTS_MODELS_DIR}/qm_correction_function_tmax_r01_2cities_DJF.png"

print("Loading data")
model_output = xr.open_dataset(model_path)["tmax"]
obs_output = xr.open_dataset(obs_path)["TmaxD"]
calib_obs = obs_output.sel(time=slice("1981-01-01", "2010-12-31"))
calib_mod = model_output.sel(time=slice("1981-01-01", "2010-12-31"))

lat_vals = model_output['lat'].values
lon_vals = model_output['lon'].values

locations = {
    "Zurich": (47.3769, 8.5417),
    "Geneva": (46.2044, 6.1432),
    "Locarno": (46.6502, 8.7943)
}

calib_times = calib_mod['time'].values
calib_doys = xr.DataArray(calib_times).dt.dayofyear.values

def get_season(doy):
    if (doy >= 335 or doy <= 59):
        return "DJF"
    else:
        return None

city_colors = {"Zurich": "b", "Geneva": "g", "Locarno": "r"}

fig, ax = plt.subplots(figsize=(10, 7))

for city, (target_lat, target_lon) in locations.items():
    print(f"\nProcessing {city}...")
    dist = np.sqrt((lat_vals - target_lat)**2 + (lon_vals - target_lon)**2)
    i_city, j_city = np.unravel_index(np.argmin(dist), dist.shape)
    print(f"Closest grid cell to {city}: i={i_city}, j={j_city}")
    print(f"Location: lat={lat_vals[i_city, j_city]}, lon={lon_vals[i_city, j_city]}")

    djf_corrections = []

    for doy in range(1, 367):  # 1 to 366
        if get_season(doy) != "DJF":
            continue
        window_doys = ((calib_doys - doy + 366) % 366)
        window_mask = (window_doys <= 45) | (window_doys >= (366 - 45))
        obs_window = calib_obs[:, i_city, j_city].values[window_mask]
        mod_window = calib_mod[:, i_city, j_city].values[window_mask]
        obs_window = obs_window[~np.isnan(obs_window)]
        mod_window = mod_window[~np.isnan(mod_window)]
        if obs_window.size == 0 or mod_window.size == 0:
            continue
        quantiles = np.linspace(0.01, 0.99, 99)
        obs_q = np.quantile(obs_window, quantiles)
        mod_q = np.quantile(mod_window, quantiles)
        obs_q = np.concatenate([[obs_q[0]], obs_q, [obs_q[-1]]])
        mod_q = np.concatenate([[mod_q[0]], mod_q, [mod_q[-1]]])
        correction = mod_q - obs_q
        djf_corrections.append(correction)

    if djf_corrections:
        mean_corr = np.mean(djf_corrections, axis=0)
        ext_q = np.linspace(0, 1, 101)
        ax.plot(ext_q, mean_corr, label=f"{city} DJF", color=city_colors[city], linestyle='-')

ax.axhline(0, color="gray", linestyle="--")
ax.set_xlabel("Quantile")
ax.set_ylabel("Correction (Model - Observations) in degrees C")
ax.set_title("Winter (DJF, seasonal) Mean Correction Function of Maximum Daily Temperatures for \nZurich and Geneva")
ax.legend(loc="upper left")
ax.grid(True)
fig.tight_layout()
plt.savefig(plot_path, dpi=1000)
print(f"Winter correction function plot for Zurich, Geneva, and Locarno saved to {plot_path}")