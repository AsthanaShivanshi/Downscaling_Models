import sys
sys.path.append("..")
sys.path.append("../..")
import torch
import numpy as np
import config
import json
import xarray as xr
import argparse
from tqdm import tqdm

from models.components.unet import DownscalingUnetLightning
from models.ae_module import AutoencoderKL
from models.components.ae import SimpleConvEncoder, SimpleConvDecoder
from models.components.ldm.denoiser.ddim import DDIMSampler
from models.ldm_module import LatentDiffusion
from models.components.ldm.denoiser import UNetModel

# dOTC runs
model_input_paths = {
    'precip': ('BC_Model_Runs/QDM/precip_BC_bicubic_r01.nc', 'precip'),   #first element : path, second,,variable
    'temp': ('BC_Model_Runs/QDM/temp_BC_bicubic_r01.nc', 'temp'),
    'temp_min': ('BC_Model_Runs/QDM/tmin_BC_bicubic_r01.nc', 'tmin'),  
    'temp_max': ('BC_Model_Runs/QDM/tmax_BC_bicubic_r01.nc', 'tmax'), 
}


# scaling params
with open(f'{config.DATASETS_TRAINING_DIR}/RhiresD_scaling_params_chronological.json', 'r') as f:
    pr_params = json.load(f)
with open(f'{config.DATASETS_TRAINING_DIR}/TabsD_scaling_params_chronological.json', 'r') as f:
    temp_params = json.load(f)
with open(f'{config.DATASETS_TRAINING_DIR}/TminD_scaling_params_chronological.json', 'r') as f:
    temp_min_params = json.load(f)
with open(f'{config.DATASETS_TRAINING_DIR}/TmaxD_scaling_params_chronological.json', 'r') as f:
    temp_max_params = json.load(f)




def standardise(var, ds, params, start, end, ds_varname=None):
    if ds_varname is None:
        ds_varname = var
    arr = ds[ds_varname].sel(time=slice(start, end)).values
    if var == "precip":
        arr = np.where(arr < 0, 0, arr)  #neg value handling
        arr = np.nan_to_num(arr, nan=0.0)  
        arr = np.log(arr + params['epsilon'])
        arr = (arr - params['mean']) / params['std']
    else:
        arr = (arr - params['mean']) / params['std']
    return arr




def denorm_pr(x):
    return np.exp(x * pr_params['std'] + pr_params['mean']) - pr_params['epsilon']

def denorm_temp(x, params):
    return x * params['std'] + params['mean']



def denorm_sample(sample):
    out = np.empty_like(sample)
    out[0] = denorm_pr(sample[0])
    out[1] = denorm_temp(sample[1], temp_params)
    out[2] = denorm_temp(sample[2], temp_min_params)
    out[3] = denorm_temp(sample[3], temp_max_params)
    return out



#elevation
elevation_path = f'{config.BASE_DIR}/sasthana/Downscaling/Downscaling_Models/elevation.tif'
elev_da = xr.open_dataarray(elevation_path)
print(elev_da.rio.crs)

print(elev_da.dims)
print(elev_da.coords)

ref_ds = xr.open_dataset(model_input_paths['precip'][0])
target_lat = ref_ds['lat']
target_lon = ref_ds['lon']
# Interpolating elevation
elev_da = elev_da.rename({'x': 'lon', 'y': 'lat'})
elev_interp = elev_da.interp(lat=target_lat, lon=target_lon)
elev = elev_interp.squeeze().values 

ds_ref = xr.open_dataset(model_input_paths['precip'][0])
dates = ds_ref['time'].sel(time=slice("1981-01-01", "2010-12-31")).values

# With elevation
inputs_norm = []


datasets = {var: xr.open_dataset(path[0]) for var, path in model_input_paths.items()}

for t in dates:
    frame = []
    for var, path in model_input_paths.items():
        ds = datasets[var]
        ds_varname = path[1]
        if var == "precip":
            arr = standardise(var, ds, pr_params, start=str(t), end=str(t), ds_varname=ds_varname)
        elif var == "temp":
            arr = standardise(var, ds, temp_params, start=str(t), end=str(t), ds_varname=ds_varname)
        elif var == "temp_min":
            arr = standardise(var, ds, temp_min_params, start=str(t), end=str(t), ds_varname=ds_varname)
        elif var == "temp_max":
            arr = standardise(var, ds, temp_max_params, start=str(t), end=str(t), ds_varname=ds_varname)
        frame.append(arr.squeeze())
    frame.append(elev)




    # Debug print
    for i, arr in enumerate(frame):
        print(f"Frame {i} shape: {arr.shape}")
    print(f"Elevation shape: {elev.shape}")
    inputs_norm.append(np.stack(frame))





# Closing for memory issues : DEBUG 
for ds in datasets.values():
    ds.close()

    
ds_ref.close()


# Models
ckpt_unet = "Training_LDM/trained_ckpts/Training_LDM.models.components.unet.DownscalingUnetLightning_checkpoint.ckpt"
#ckpt_vae = "Training_LDM/trained_ckpts/Training_LDM.models.ae_module.AutoencoderKL_checkpoint.ckpt"
#ckpt_ldm = "Training_LDM/trained_ckpts/LDM_checkpoint.ckpt"

model_UNet = DownscalingUnetLightning(
    in_ch=5, out_ch=4, features=[64, 128, 256, 512],
    channel_names=["precip", "temp", "temp_min", "temp_max"]
)
unet_state_dict = torch.load(ckpt_unet, map_location="cpu")["state_dict"]
model_UNet.load_state_dict(unet_state_dict, strict=False)
model_UNet.eval()

#encoder = SimpleConvEncoder(in_dim=4, levels=1, min_ch=64, ch_mult=1)
#decoder = SimpleConvDecoder(in_dim=64, levels=1, min_ch=16)
#model_VAE = AutoencoderKL.load_from_checkpoint(
#    ckpt_vae, encoder=encoder, decoder=decoder, kl_weight=0.01, strict=False
#)
#model_VAE.eval()

#ldm_ckpt = torch.load(ckpt_ldm, map_location="cpu")
#remapped_ldm_state_dict = {}
#for k, v in ldm_ckpt["state_dict"].items():
#    if k.startswith("autoencoder.unet_regr.unet."):
#        new_key = "autoencoder.unet." + k[len("autoencoder.unet_regr.unet."):]
#    elif k.startswith("autoencoder.unet_regr."):
#        new_key = "autoencoder.unet." + k[len("autoencoder.unet_regr."):]
#    else:
#        new_key = k
#    remapped_ldm_state_dict[new_key] = v

#denoiser = UNetModel(
#    in_channels=32, out_channels=32, model_channels=64, num_res_blocks=2,
#    attention_resolutions=[1,2,4], context_ch=None, channel_mult=[1,2,4,4],
#    conv_resample=True, dims=2, use_fp16=False, num_heads=4
#)
#model_LDM = LatentDiffusion(denoiser=denoiser, autoencoder=model_VAE)
#model_LDM.load_state_dict(remapped_ldm_state_dict, strict=False)
#model_LDM.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_UNet.to(device)
#model_VAE.to(device)
#model_LDM.to(device)

#ddim_num_steps = 129
#ddim_eta = 0.0
#sampler = DDIMSampler(model_LDM, schedule="linear")
#sampler.make_schedule(ddim_num_steps=ddim_num_steps, ddim_eta=ddim_eta, verbose=False)

#def ddim_sample_from_t(sampler, model, x_t, t_start, t_end=0, shape=None, **kwargs):
#    device = x_t.device
#    timesteps = torch.arange(t_start, t_end-1, -1, device=device)
#    x = x_t
#    for i, t in enumerate(timesteps):
#        x, _ = sampler.p_sample_ddim(x, None, t.repeat(x.shape[0]), i, **kwargs)
#    return x

#def pipeline(input_sample, seed=None):
#    with torch.no_grad():
#        if seed is not None:
#            torch.manual_seed(seed)
#            np.random.seed(seed)
#        # UNet prediction
#        unet_prediction = model_UNet(input_sample)
#        # Bicubic input: first 4 channels (exclude elevation)
#        input_bicubic = input_sample[:, :4, :, :]
#        # Residuals: input_bicubic - unet_prediction
#        residuals = input_bicubic - unet_prediction
#        # Encode residuals to VAE latent
#        mean, log_var = model_VAE.encode(residuals)
#        std = torch.exp(0.5 * log_var)
#        eps = torch.randn_like(std)
#        latent = mean + eps * std
#        # LDM sampling
#        t = torch.tensor([ddim_num_steps-1], device=latent.device).long()
#        noise = torch.randn_like(latent)
#        noisy_latent = model_LDM.q_sample(latent, t, noise=noise)
#        denoised_latent = ddim_sample_from_t(sampler, model_LDM, noisy_latent, t_start=t.item())
#        # Decode denoised residuals
#        refined_residuals = model_VAE.decode(denoised_latent)
#        # Add back to UNet prediction
#        final_prediction = unet_prediction + refined_residuals
#        return final_prediction[0].cpu().numpy()

parser = argparse.ArgumentParser()
#parser.add_argument("--n_samples", type=int, default=10, help="N(Samples) per t")
args = parser.parse_args()
#n_samples = args.n_samples

#all_samples = []
unet_baseline = []

inputs_norm = np.stack(inputs_norm)

with tqdm(total=len(dates), desc="Frame") as pbar:
    for idx in range(inputs_norm.shape[0]):
        input_sample = torch.tensor(inputs_norm[idx], dtype=torch.float32).unsqueeze(0).to(device)
        #frame_samples = []
        with torch.no_grad():
            unet_pred = model_UNet(input_sample)
            unet_pred_np = denorm_sample(unet_pred[0].cpu().numpy())
            unet_baseline.append(unet_pred_np)
        #for seed in range(n_samples):
        #    sample = pipeline(input_sample, seed=seed)
        #    sample_denorm = denorm_sample(sample)
        #    frame_samples.append(sample_denorm)
        #all_samples.append(np.stack(frame_samples))
        pbar.update(1)

#all_samples = np.stack(all_samples)  # shape: (N, n_samples, 4, H, W)
unet_baseline = np.stack(unet_baseline)  # shape: (N, 4, H, W)

#da_ldm = xr.DataArray(
#    all_samples,
#    dims=["time", "sample", "variable", "y", "x"],
#    coords={
#        "time": dates,
#        "variable": ["precip", "temp", "temp_min", "temp_max"]
#    },
#    name="ldm_samples"
#)
#da_ldm.to_netcdf("EQM_BC_modelrun_1981_2010_samples_LDM.nc")

da_unet = xr.DataArray(
    unet_baseline,
    dims=["time", "variable", "y", "x"],
    coords={
        "time": dates,
        "variable": ["precip", "temp", "temp_min", "temp_max"]
    },
    name="unet_baseline"
)
da_unet.to_netcdf("QDM_BC_modelrun_1981_2010_samples_UNet_baseline.nc")