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
from DownscalingDataModule import DownscalingDataModule

# Loading dates
datetime_ref_path = "Training_Chronological_Dataset/RhiresD_target_test_chronological_scaled.nc"
ds = xr.open_dataset(datetime_ref_path)
times = ds["time"].values
dates = np.array([str(np.datetime64(t)) for t in times])
lat= ds["lat"].values
lon= ds["lon"].values

# Model checkpoints
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

encoder = SimpleConvEncoder(in_dim=4, levels=2, min_ch=16, ch_mult=4)
decoder = SimpleConvDecoder(in_dim=64, levels=2, min_ch=16)
#model_VAE = AutoencoderKL.load_from_checkpoint(
#    ckpt_vae, encoder=encoder, decoder=decoder, kl_weight=0.001, strict=False
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




#ddim_num_steps = 129
#ddim_eta = 0.0
#sampler = DDIMSampler(model_LDM, schedule="linear")
#sampler.make_schedule(ddim_num_steps=ddim_num_steps, ddim_eta=ddim_eta, verbose=False)



#def ddim_sample_from_t(sampler, model, x_t, t_start, t_end=0, shape=None, **kwargs):
#   device = x_t.device
#    timesteps = torch.arange(t_start, t_end-1, -1, device=device)
#    x = x_t
#    for i, t in enumerate(timesteps):
#        x, _ = sampler.p_sample_ddim(x, None, t.repeat(x.shape[0]), i, **kwargs)
#    return x




#def get_latent_shape(input_sample):
#    dummy_residual = torch.zeros((1, 4, input_sample.shape[-2], input_sample.shape[-1]), device=input_sample.device)
#    mean, _ = model_VAE.encode(dummy_residual)
#    return mean.shape


def pipeline(input_sample, seed=None):
    with torch.no_grad():
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        unet_prediction = model_UNet(input_sample)
        #latent_shape = get_latent_shape(input_sample)
        #latent = torch.randn(latent_shape, device=unet_prediction.device)
        #global ddim_num_steps
        #t = torch.tensor([ddim_num_steps-1], device=latent.device).long()
        #noisy_latent = model_LDM.q_sample(latent, t, noise=torch.randn_like(latent))
        #denoised_latent = ddim_sample_from_t(sampler, model_LDM, noisy_latent, t_start=t.item())
        #sampled_residuals = model_VAE.decode(denoised_latent)
        #final_prediction = unet_prediction + sampled_residuals
        #return final_prediction[0].cpu().numpy()  # shape: (4, H, W)

# Data paths
test_input_paths = {
    'precip': f'{config.DATASETS_TRAINING_DIR}/RhiresD_input_test_chronological_scaled.nc',
    'temp': f'{config.DATASETS_TRAINING_DIR}/TabsD_input_test_chronological_scaled.nc',
    'temp_min': f'{config.DATASETS_TRAINING_DIR}/TminD_input_test_chronological_scaled.nc',
    'temp_max': f'{config.DATASETS_TRAINING_DIR}/TmaxD_input_test_chronological_scaled.nc'
}
test_target_paths = {
    'precip': f'{config.DATASETS_TRAINING_DIR}/RhiresD_target_test_chronological_scaled.nc',
    'temp': f'{config.DATASETS_TRAINING_DIR}/TabsD_target_test_chronological_scaled.nc',
    'temp_min': f'{config.DATASETS_TRAINING_DIR}/TminD_target_test_chronological_scaled.nc',
    'temp_max': f'{config.DATASETS_TRAINING_DIR}/TmaxD_target_test_chronological_scaled.nc'
}
elevation_path = f'{config.BASE_DIR}/sasthana/Downscaling/Downscaling_Models/elevation.tif'



dm = DownscalingDataModule(
    train_input={}, train_target={},
    test_input=test_input_paths, test_target=test_target_paths,
    elevation=elevation_path, batch_size=32, num_workers=4,
    preprocessing={
        'variables': {
            'input': {'precip': 'RhiresD', 'temp': 'TabsD', 'temp_min': 'TminD', 'temp_max': 'TmaxD'},
            'target': {'precip': 'RhiresD', 'temp': 'TabsD', 'temp_min': 'TminD', 'temp_max': 'TmaxD'}
        },
        'preprocessing': {'nan_to_num': True, 'nan_value': 0.0}
    }
)
dm.setup('test')
test_loader = dm.test_dataloader()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_UNet.to(device)
#model_VAE.to(device)
#model_LDM.to(device)




# Destandardisation
with open(f'{config.DATASETS_TRAINING_DIR}/RhiresD_scaling_params_chronological.json', 'r') as f:
    pr_params = json.load(f)
with open(f'{config.DATASETS_TRAINING_DIR}/TabsD_scaling_params_chronological.json', 'r') as f:
    temp_params = json.load(f)
with open(f'{config.DATASETS_TRAINING_DIR}/TminD_scaling_params_chronological.json', 'r') as f:
    temp_min_params = json.load(f)
with open(f'{config.DATASETS_TRAINING_DIR}/TmaxD_scaling_params_chronological.json', 'r') as f:
    temp_max_params = json.load(f)

def denorm_pr(x):
    return np.exp(x * pr_params['std'] + pr_params['mean']) - pr_params['epsilon']

def denorm_temp(x, params):
    return x * params['std'] + params['mean']

def denorm_sample(sample):
    out = np.empty_like(sample)
    out[0] = denorm_pr(sample[0])  # Precip
    out[1] = denorm_temp(sample[1], temp_params)
    out[2] = denorm_temp(sample[2], temp_min_params)
    out[3] = denorm_temp(sample[3], temp_max_params)
    return out


orig_inputs = []
var_to_dsvar = {
    "precip": "RhiresD",
    "temp": "TabsD",
    "temp_min": "TminD",
    "temp_max": "TmaxD"
}


for var in ["precip", "temp", "temp_min", "temp_max"]:
    ds_in = xr.open_dataset(test_input_paths[var])
    orig_inputs.append(ds_in[var_to_dsvar[var]].values)  # shape: (time, N, E)
    ds_in.close()
orig_inputs = np.stack(orig_inputs, axis=1)  # shape: (time, 4, N, E)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--n_samples", type=int, default=10, help="Number of samples per frame")
    args = parser.parse_args()

    #n_samples = args.n_samples
    all_samples = []
    unet_baseline = []

    total_frames = len(test_loader.dataset)

    with tqdm(total=total_frames, desc="Frames") as pbar:
        for batch in test_loader:
            test_inputs, test_targets = batch
            batch_size = test_inputs.shape[0]
            for idx in range(batch_size):
                input_sample = test_inputs[idx].unsqueeze(0).to(device)
                frame_samples = []

                # baseline pred
                with torch.no_grad():
                    unet_pred = model_UNet(input_sample)
             
                    input_np = test_inputs[idx].cpu().numpy()
                    frame_idx = len(unet_baseline)  # This assumes you append in order!
                    nan_mask = np.isnan(orig_inputs[frame_idx])  # shape: (4, N, E)
                    unet_pred_np = denorm_sample(unet_pred[0].cpu().numpy())
                    unet_pred_np[nan_mask] = np.nan
                    unet_baseline.append(unet_pred_np)
                    
                #for seed in range(n_samples):
                #    sample = pipeline(input_sample, seed=seed)
                #    sample_denorm = denorm_sample(sample)
                #    frame_samples.append(sample_denorm)
                #all_samples.append(np.stack(frame_samples))
                pbar.update(1)



    # Save LDM samples
    #da_ldm = xr.DataArray(
    #    all_samples,
    #    dims=["time", "sample", "variable", "y", "x"],
    #    coords={
    #        "time": dates,
    #        "variable": ["precip", "temp", "temp_min", "temp_max"]
    #    },
    #    name="ldm_samples"
    #)
    #da_ldm.to_netcdf("testset_2021_2023_samples_LDM.nc")



    unet_baseline_np = np.array(unet_baseline)  # shape: (time, 4, N, E)
    unet_baseline_np = np.transpose(unet_baseline_np, (0, 2, 3, 1))  # (time, N, E, variable)

    lat2d = ds["lat"].values  # (N, E)
    lon2d = ds["lon"].values  # (N, E)
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


    ds_out.to_netcdf("testset_2021_2023_samples_UNet_baseline.nc", encoding=encoding)
    print(f"UNet baseline saved with shape: {unet_baseline_np.shape}")

    #debug prints
    print(ds_out.dims)
    print(ds_out.coords)
    #print(f"LDM samples saved with shape: {np.array(all_samples).shape}")
