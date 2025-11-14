import sys
sys.path.append("..")
sys.path.append("../..")
import torch
import numpy as np
import config
import json
import xarray as xr
import argparse

from models.components.unet import DownscalingUnetLightning
from models.ae_module import AutoencoderKL
from models.components.ae import SimpleConvEncoder, SimpleConvDecoder
from models.components.ldm.denoiser.ddim import DDIMSampler
from models.ldm_module import LatentDiffusion


from models.components.ldm.denoiser import UNetModel

from DownscalingDataModule import DownscalingDataModule

#Loading dates 
datetime_ref_path = "Training_Chronological_Dataset/RhiresD_target_test_chronological_scaled.nc" #For matching indices to dates
ds = xr.open_dataset(datetime_ref_path)

times = ds["time"].values
dates = np.array([str(np.datetime64(t)) for t in times])



# ckpts
ckpt_unet = "Training_LDM/trained_ckpts/Training_LDM.models.components.unet.DownscalingUnetLightning_checkpoint.ckpt"
ckpt_vae = "Training_LDM/trained_ckpts/Training_LDM.models.ae_module.AutoencoderKL_checkpoint.ckpt"
ckpt_ldm = "Training_LDM/trained_ckpts/LDM_checkpoint.ckpt"

model_UNet = DownscalingUnetLightning(
    in_ch=5, out_ch=4, features=[64, 128, 256, 512],
    channel_names=["precip", "temp", "temp_min", "temp_max"]
)
unet_state_dict = torch.load(ckpt_unet, map_location="cpu")["state_dict"]
model_UNet.load_state_dict(unet_state_dict, strict=False)
model_UNet.eval()



encoder = SimpleConvEncoder(in_dim=4, levels=1, min_ch=64, ch_mult=1)
decoder = SimpleConvDecoder(in_dim=64, levels=1, min_ch=16)
model_VAE = AutoencoderKL.load_from_checkpoint(
    ckpt_vae, encoder=encoder, decoder=decoder, kl_weight=0.01, strict=False
)
model_VAE.eval()



ldm_ckpt = torch.load(ckpt_ldm, map_location="cpu")



#Debug; keys didnt match before 
remapped_ldm_state_dict = {}
for k, v in ldm_ckpt["state_dict"].items():
    if k.startswith("autoencoder.unet_regr.unet."):
        new_key = "autoencoder.unet." + k[len("autoencoder.unet_regr.unet."):]
    elif k.startswith("autoencoder.unet_regr."):
        new_key = "autoencoder.unet." + k[len("autoencoder.unet_regr."):]
    else:
        new_key = k
    remapped_ldm_state_dict[new_key] = v




denoiser = UNetModel(
    in_channels=32, out_channels=32, model_channels=64, num_res_blocks=2,
    attention_resolutions=[1,2,4], context_ch=None, channel_mult=[1,2,4,4],
    conv_resample=True, dims=2, use_fp16=False, num_heads=4
)
model_LDM = LatentDiffusion(denoiser=denoiser, autoencoder=model_VAE)
model_LDM.load_state_dict(remapped_ldm_state_dict, strict=False)
model_LDM.eval()



ddim_num_steps = 129


ddim_eta = 0.0 #Changeable from the .sh script


sampler = DDIMSampler(model_LDM, schedule="linear")
sampler.make_schedule(ddim_num_steps=ddim_num_steps, ddim_eta=ddim_eta, verbose=False)



def ddim_sample_from_t(sampler, model, x_t, t_start, t_end=0, shape=None, **kwargs):
    device = x_t.device
    timesteps = torch.arange(t_start, t_end-1, -1, device=device)
    x = x_t
    for i, t in enumerate(timesteps):
        x, _ = sampler.p_sample_ddim(x, None, t.repeat(x.shape[0]), i, **kwargs)
    return x



def pipeline(input_sample, target_sample=None, seed=None):
    with torch.no_grad():
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        unet_prediction = model_UNet(input_sample)
        if target_sample is not None:
            residuals = target_sample - unet_prediction
        else:
            raise ValueError("target_sample required for posterior refinement.")
        mean, log_var = model_VAE.encode(residuals)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        latent = mean + eps * std
        global ddim_num_steps
        t = torch.tensor([ddim_num_steps-1], device=latent.device).long()
        noise = torch.randn_like(latent)
        noisy_latent = model_LDM.q_sample(latent, t, noise=noise)
        denoised_latent = ddim_sample_from_t(sampler, model_LDM, noisy_latent, t_start=t.item())
        refined_residuals = model_VAE.decode(denoised_latent)
        final_prediction = unet_prediction + refined_residuals
        return final_prediction[0].cpu().numpy()  # shape: (4, H, W)





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
model_VAE.to(device)
model_LDM.to(device)



#Destandardisation
#Number of samples : from .sh script

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
    # sample shape: (4, H, W)
    out = np.empty_like(sample)
    out[0] = denorm_pr(sample[0])  # Precip
    out[1] = denorm_temp(sample[1], temp_params)
    out[2] = denorm_temp(sample[2], temp_min_params)
    out[3] = denorm_temp(sample[3], temp_max_params)
    return out




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=15, help="Number of samples per frame")
    args = parser.parse_args()

    n_samples = args.n_samples
    all_samples = []
    unet_baseline = []

    for batch in test_loader:
        test_inputs, test_targets = batch
        batch_size = test_inputs.shape[0]
        for idx in range(batch_size):
            input_sample = test_inputs[idx].unsqueeze(0).to(device)
            target_sample = test_targets[idx].unsqueeze(0).to(device)
            frame_samples = []
            # baseline pred
            with torch.no_grad():
                unet_pred = model_UNet(input_sample)
                unet_pred_np = denorm_sample(unet_pred[0].cpu().numpy())
                unet_baseline.append(unet_pred_np)
            for seed in range(n_samples):
                sample = pipeline(input_sample, target_sample, seed=seed)
                sample_denorm = denorm_sample(sample)
                frame_samples.append(sample_denorm)
            all_samples.append(np.stack(frame_samples))

    all_samples = np.stack(all_samples)  # shape: (N, n_samples, 4, H, W)
    unet_baseline = np.stack(unet_baseline)  # shape: (N, 4, H, W)



    # LDM samples
    da_ldm = xr.DataArray(
        all_samples,
        dims=["time", "sample", "variable", "y", "x"],
        coords={
            "time": dates,
            "variable": ["precip", "temp", "temp_min", "temp_max"]
        },
        name="ldm_samples"
    )
    da_ldm.to_netcdf("testset_2021_2023_samples_LDM.nc")


#unet pred
    da_unet = xr.DataArray(
        unet_baseline,
        dims=["time", "variable", "y", "x"],
        coords={
            "time": dates,
            "variable": ["precip", "temp", "temp_min", "temp_max"]
        },
        name="unet_baseline"
    )
    da_unet.to_netcdf("testset_2021_2023_samples_UNet_baseline.nc")

    print(f"LDM samples saved with shape: {all_samples.shape}")
    print(f"UNet baseline saved with shape: {unet_baseline.shape}")

