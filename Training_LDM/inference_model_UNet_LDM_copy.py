import sys
sys.path.append("..")
sys.path.append("../..")
import torch
import numpy as np
import config
import xarray as xr
from tqdm import tqdm

from models.components.unet import DownscalingUnetLightning
from DownscalingDataModule import DownscalingDataModule

# QDM runs
qdm_input_paths = {
    'precip': 'BC_Model_Runs/dOTC/precip_temp_tmin_tmax_bicubic_r01.nc',
    'temp': 'BC_Model_Runs/dOTC/precip_temp_tmin_tmax_bicubic_r01.nc',
    'temp_min': 'BC_Model_Runs/dOTC/precip_temp_tmin_tmax_bicubic_r01.nc',
    'temp_max': 'BC_Model_Runs/dOTC/precip_temp_tmin_tmax_bicubic_r01.nc'
}
qdm_target_paths = qdm_input_paths.copy()  # dummy, not used: AsthanaSh

elevation_path = f'{config.BASE_DIR}/sasthana/Downscaling/Downscaling_Models/elevation.tif'

# DataModule 
dm = DownscalingDataModule(
    train_input={}, train_target={},
    val_input={}, val_target={},
    test_input=qdm_input_paths,
    test_target=qdm_target_paths,
    elevation=elevation_path,
    batch_size=1,
    num_workers=0,
    preprocessing={
        'variables': {
            'input': {
                'precip': 'precip',
                'temp': 'temp',
                'temp_min': 'tmin',
                'temp_max': 'tmax'
            },
            'target': {
                'precip': 'precip',
                'temp': 'temp',
                'temp_min': 'tmin',
                'temp_max': 'tmax'
            }
        },
        'preprocessing': {
            'nan_to_num': True,
            'nan_value': 0.0
        }
    }
)
dm.setup('test')
test_loader = dm.test_dataloader()

# Load reference for time/lat/lon
ref_ds = xr.open_dataset(qdm_input_paths['precip'])
dates = ref_ds['time'].values
lat2d = ref_ds["lat"].values
lon2d = ref_ds["lon"].values
ref_ds.close()

# Model
ckpt_unet = "Training_LDM/trained_ckpts/Training_LDM.models.components.unet.DownscalingUnetLightning_checkpoint.ckpt"
model_UNet = DownscalingUnetLightning(
    in_ch=5, out_ch=4, features=[64, 128, 256, 512],
    channel_names=["precip", "temp", "temp_min", "temp_max"]
)


unet_state_dict = torch.load(ckpt_unet, map_location="cpu")["state_dict"]
model_UNet.load_state_dict(unet_state_dict, strict=False)

model_UNet.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_UNet.to(device)



# Inference 
unet_baseline = []

with tqdm(total=len(dates), desc="Frame") as pbar:
    for batch in test_loader:
        input_tensor, _ = batch  # ignore target
        input_tensor = input_tensor.to(device)
        with torch.no_grad():
            unet_pred = model_UNet(input_tensor)
        unet_pred_np = unet_pred[0].cpu().numpy()  # (4, H, W)
        unet_baseline.append(unet_pred_np)
        pbar.update(1)



unet_baseline_np = np.array(unet_baseline)  # (time, 4, H, W)
unet_baseline_np = np.transpose(unet_baseline_np, (0, 2, 3, 1))  # (time, H, W, 4)

var_names = ["precip", "temp", "temp_min", "temp_max"]


ds_out = xr.Dataset(
    {
        var: (("time", "N", "E"), unet_baseline_np[:, :, :, i])
        for i, var in enumerate(var_names)
    },
    coords={
        "time": dates,
        "N": np.arange(lat2d.shape[0]),
        "E": np.arange(lat2d.shape[1]),
        "lat": (("N", "E"), lat2d),
        "lon": (("N", "E"), lon2d),
    }
)


encoding = {var: {"_FillValue": np.nan} for var in var_names}
ds_out.to_netcdf("dOTC_BC_modelrun_1981_2010_samples_UNet_baseline.nc", encoding=encoding)
print(f"UNet baseline saved with shape: {unet_baseline_np.shape}")
print("Final unet_baseline_np NaNs:", np.isnan(unet_baseline_np).sum())