import xarray as xr
import numpy as np
from SBCK import QM
import config
from joblib import Parallel, delayed

model_path = f"{config.SCRATCH_DIR}/temp_r01_HR_masked.nc"
obs_path = f"{config.SCRATCH_DIR}/TabsD_1971_2023.nc"
output_path = f"{config.BIAS_CORRECTED_DIR}/EQM/eqm_temp_r01.nc"

print("Loading data")
model_output = xr.open_dataset(model_path)["temp"]
obs_output = xr.open_dataset(obs_path)["TabsD"]
calib_obs = obs_output.sel(time=slice("1981-01-01", "2010-12-31"))
calib_mod = model_output.sel(time=slice("1981-01-01", "2010-12-31"))

lat_vals = model_output['lat'].values
lon_vals = model_output['lon'].values

print("model_output shape:", model_output.shape)
print("lat_vals shape:", lat_vals.shape)
print("lon_vals shape:", lon_vals.shape)

qm_data = np.full(model_output.shape, np.nan, dtype=np.float32)

def process_cell(i, j):
    obs_valid = calib_obs[:, i, j].values[~np.isnan(calib_obs[:, i, j].values)]
    mod_valid = calib_mod[:, i, j].values[~np.isnan(calib_mod[:, i, j].values)]
    if obs_valid.size == 0 or mod_valid.size == 0:
        return None
    qm = QM()
    qm.fit(mod_valid.reshape(-1, 1), obs_valid.reshape(-1, 1))
    full_mod_series = model_output[:, i, j].values.reshape(-1, 1)
    qm_series = qm.predict(full_mod_series).flatten()
    return (i, j, qm_series.astype(np.float32))

results = Parallel(n_jobs=-1, backend="loky")(
    delayed(process_cell)(i, j)
    for i in range(lat_vals.shape[0])
    for j in range(lon_vals.shape[1])
)

for result in results:
    if result is not None:
        i, j, qm_series = result
        qm_data[:, i, j] = qm_series

qm_ds = xr.Dataset(
    {"temp": (model_output.dims, qm_data)},
    coords=model_output.coords
)
qm_ds.to_netcdf(output_path)
print(f"Bias-corrected QM output saved to {output_path}")