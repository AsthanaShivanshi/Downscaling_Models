import sys
sys.path.append("..") #parent (required)
sys.path.append("../..") #grandparent (required)
import torch
import numpy as np
import json
import xarray as xr
from models.components.ldm.denoiser import UNetModel
from Training_LDM.DownscalingDataModule import DownscalingDataModule
from models.components.unet import DownscalingUnetLightning
from models.ae_module import AutoencoderKL
from models.components.ae import SimpleConvEncoder, SimpleConvDecoder
from models.ldm_module import LatentDiffusion
from models.components.ldm.denoiser.ddim import DDIMSampler
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Scaling params
with open(config.DATASETS_TRAINING_DIR) as f:
    pr_params = json.load(f)
with open("../Training_Chronological_Dataset/TabsD_scaling_params_chronological.json") as f:
    temp_params = json.load(f)
with open("../Training_Chronological_Dataset/TminD_scaling_params_chronological.json") as f:
    temp_min_params = json.load(f)
with open("../Training_Chronological_Dataset/TmaxD_scaling_params_chronological.json") as f:
    temp_max_params = json.load(f)

def denorm_pr(x):
    return x * (pr_params["max"] - pr_params["min"]) + pr_params["min"]

def denorm_temp(x, params):
    return x * params["std"] + params["mean"]

# Instantiate models
encoder = SimpleConvEncoder(in_dim=5, levels=1, min_ch=64, ch_mult=1)
decoder = SimpleConvDecoder(in_dim=64, levels=2, min_ch=16)
ckpt_vae = "trained_ckpts/Training_LDM.models.ae_module.AutoencoderKL_checkpoint.ckpt"
model_VAE = AutoencoderKL.load_from_checkpoint(
    ckpt_vae,
    encoder=encoder,
    decoder=decoder,
    kl_weight=0.01,
    strict=False
).to(device)

denoiser = UNetModel(
    in_channels=32,
    out_channels=32,
    model_channels=64,
    num_res_blocks=2,
    attention_resolutions=[1, 2, 4],
    context_ch=None,
    channel_mult=[1, 2, 4, 4],
    conv_resample=True,
    dims=2,
    use_fp16=False,
    num_heads=4
)

ckpt_ldm = "trained_ckpts/LDM_checkpoint.ckpt"
ldm_ckpt = torch.load(ckpt_ldm, map_location="cpu")
remapped_ldm_state_dict = {}
for k, v in ldm_ckpt["state_dict"].items():
    if k.startswith("autoencoder.unet_regr.unet."):
        new_key = "autoencoder.unet." + k[len("autoencoder.unet_regr.unet."):]
    elif k.startswith("autoencoder.unet_regr."):
        new_key = "autoencoder.unet." + k[len("autoencoder.unet_regr."):]
    else:
        new_key = k
    remapped_ldm_state_dict[new_key] = v

model_LDM = LatentDiffusion(
    denoiser=denoiser,
    autoencoder=model_VAE,
    timesteps=1000,
    beta_schedule="linear",
    loss_type="l2",
    use_ema=True,
    lr=1e-4,
    lr_warmup=0,
    linear_start=1e-4,
    linear_end=2e-2,
    cosine_s=8e-3,
    parameterization="eps"
).to(device)
model_LDM.load_state_dict(remapped_ldm_state_dict, strict=False)

# DataModule setup
dm = DownscalingDataModule(
    train_input={
        "precip": "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/Training_Chronological_Dataset/RhiresD_input_test_chronological_scaled.nc",
        "temp": "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/Training_Chronological_Dataset/TabsD_input_test_chronological_scaled.nc",
        "temp_min": "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/Training_Chronological_Dataset/TminD_input_test_chronological_scaled.nc",
        "temp_max": "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/Training_Chronological_Dataset/TmaxD_input_test_chronological_scaled.nc"
    },
    train_target={
        "precip": "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/Training_Chronological_Dataset/RhiresD_target_test_chronological_scaled.nc",
        "temp": "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/Training_Chronological_Dataset/TabsD_target_test_chronological_scaled.nc",
        "temp_min": "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/Training_Chronological_Dataset/TminD_target_test_chronological_scaled.nc",
        "temp_max": "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/Training_Chronological_Dataset/TmaxD_target_test_chronological_scaled.nc"
    },
    elevation="/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/elevation.tif",
    batch_size=32,
    num_workers=4,
    preprocessing={
        "variables": {
            "input": {
                "precip": "RhiresD",
                "temp": "TabsD",
                "temp_min": "TminD",
                "temp_max": "TmaxD"
            },
            "target": {
                "precip": "RhiresD",
                "temp": "TabsD",
                "temp_min": "TminD",
                "temp_max": "TmaxD"
            }
        },
        "preprocessing": {
            "nan_to_num": True,
            "nan_value": 0.0
        }
    }
)
dm.setup()

# Inference loop for test set
num_samples = 5
test_loader = dm.test_dataloader() if hasattr(dm, "test_dataloader") else dm.train_dataloader()

precip_samples = []
temp_samples = []
temp_min_samples = []
temp_max_samples = []

for batch_idx, test_batch in enumerate(test_loader):
    test_inputs, test_targets = test_batch  # (batch_size, channels, H, W)
    batch_size = test_inputs.shape[0]
    for t in range(batch_size):
        input_frame = test_inputs[t].unsqueeze(0).to(device)  # (1, ch, H, W) on GPU
        input_batch = input_frame.repeat(num_samples, 1, 1, 1).to(device)  # (num_samples, ch, H, W) on GPU
        with torch.no_grad():
            mean, log_var = model_VAE.encode(input_batch)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            latent = mean + eps * std

            sampler = DDIMSampler(model_LDM)
            shape = latent.shape[1:]
            samples, _ = sampler.sample(
                S=50,
                batch_size=num_samples,
                shape=shape,
                x_T=latent,
                eta=0.2,
                verbose=False,
                progbar=False
            )
            recon = model_VAE.decode(samples)

        # Move outputs to CPU before saving
        precip_samples.append([denorm_pr(recon[i, 0].cpu().numpy()) for i in range(num_samples)])
        temp_samples.append([denorm_temp(recon[i, 1].cpu().numpy(), temp_params) for i in range(num_samples)])
        temp_min_samples.append([denorm_temp(recon[i, 2].cpu().numpy(), temp_min_params) for i in range(num_samples)])
        temp_max_samples.append([denorm_temp(recon[i, 3].cpu().numpy(), temp_max_params) for i in range(num_samples)])

# Convert to numpy arrays: (timesteps, samples, H, W)
precip_arr = np.array(precip_samples)
temp_arr = np.array(temp_samples)
temp_min_arr = np.array(temp_min_samples)
temp_max_arr = np.array(temp_max_samples)

# Save as NetCDF using xarray
ds_pr = xr.DataArray(precip_arr, dims=["time", "sample", "y", "x"])
ds_temp = xr.DataArray(temp_arr, dims=["time", "sample", "y", "x"])
ds_temp_min = xr.DataArray(temp_min_arr, dims=["time", "sample", "y", "x"])
ds_temp_max = xr.DataArray(temp_max_arr, dims=["time", "sample", "y", "x"])

ds_pr.to_netcdf("outputs/precip_samples.nc")
ds_temp.to_netcdf("outputs/temp_samples.nc")
ds_temp_min.to_netcdf("outputs/temp_min_samples.nc")
ds_temp_max.to_netcdf("outputs/temp_max_samples.nc")

print("NetCDF files saved in outputs/")