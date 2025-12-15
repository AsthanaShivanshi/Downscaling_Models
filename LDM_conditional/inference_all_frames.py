import sys
sys.path.append("..")
sys.path.append("../..")
import torch
import numpy as np
import json
import xarray as xr
from tqdm import tqdm
import paths

from LDM_conditional.DownscalingDataModule import DownscalingDataModule
from models.unet_module import DownscalingUnetLightning
from models.ae_module import AutoencoderKL
from models.components.ae import SimpleConvEncoder, SimpleConvDecoder
from models.components.ldm.conditioner import AFNOConditionerNetCascade
from models.components.ldm.denoiser.unet import UNetModel
from models.ldm_module import LatentDiffusion
from models.components.ldm.denoiser.ddim import DDIMSampler


S = 250  # DDIM 
N_SAMPLES = 10  # LDM


train_input_paths = {
    'precip': paths.DATASETS_TRAINING_DIR+"/RhiresD_input_train_scaled.nc",
    'temp': paths.DATASETS_TRAINING_DIR+"/TabsD_input_train_scaled.nc",
    'temp_min': paths.DATASETS_TRAINING_DIR+"/TminD_input_train_scaled.nc",
    'temp_max': paths.DATASETS_TRAINING_DIR+"/TmaxD_input_train_scaled.nc"
}
train_target_paths = {
    'precip': paths.DATASETS_TRAINING_DIR+"/RhiresD_target_train_scaled.nc",
    'temp': paths.DATASETS_TRAINING_DIR+"/TabsD_target_train_scaled.nc",
    'temp_min': paths.DATASETS_TRAINING_DIR+"/TminD_target_train_scaled.nc",
    'temp_max': paths.DATASETS_TRAINING_DIR+"/TmaxD_target_train_scaled.nc"
}
val_input_paths = {
    'precip': paths.DATASETS_TRAINING_DIR+"/RhiresD_input_val_scaled.nc",
    'temp': paths.DATASETS_TRAINING_DIR+"/TabsD_input_val_scaled.nc",
    'temp_min': paths.DATASETS_TRAINING_DIR+"/TminD_input_val_scaled.nc",
    'temp_max': paths.DATASETS_TRAINING_DIR+"/TmaxD_input_val_scaled.nc"
}
val_target_paths = {
    'precip': paths.DATASETS_TRAINING_DIR+"/RhiresD_target_val_scaled.nc",
    'temp': paths.DATASETS_TRAINING_DIR+"/TabsD_target_val_scaled.nc",
    'temp_min': paths.DATASETS_TRAINING_DIR+"/TminD_target_val_scaled.nc",
    'temp_max': paths.DATASETS_TRAINING_DIR+"/TmaxD_target_val_scaled.nc"
}
test_input_paths = {
    'precip': paths.DATASETS_TRAINING_DIR+"/RhiresD_input_test_scaled.nc",
    'temp': paths.DATASETS_TRAINING_DIR+"/TabsD_input_test_scaled.nc",
    'temp_min': paths.DATASETS_TRAINING_DIR+"/TminD_input_test_scaled.nc",
    'temp_max': paths.DATASETS_TRAINING_DIR+"/TmaxD_input_test_scaled.nc"
}
test_target_paths = {
    'precip': paths.DATASETS_TRAINING_DIR+"/RhiresD_target_test_scaled.nc",
    'temp': paths.DATASETS_TRAINING_DIR+"/TabsD_target_test_scaled.nc",
    'temp_min': paths.DATASETS_TRAINING_DIR+"/TminD_target_test_scaled.nc",
    'temp_max': paths.DATASETS_TRAINING_DIR+"/TmaxD_target_test_scaled.nc"
}
elevation_path = paths.BASE_DIR+"/sasthana/Downscaling/Downscaling_Models/elevation.tif"

dm = DownscalingDataModule(
    train_input=train_input_paths,
    train_target=train_target_paths,
    val_input=val_input_paths,
    val_target=val_target_paths,
    test_input=test_input_paths,
    test_target=test_target_paths,
    elevation=elevation_path,
    batch_size=1,
    num_workers=0,
    preprocessing={
        'variables': {
            'input': {
                'precip': 'RhiresD',
                'temp': 'TabsD',
                'temp_min': 'TminD',
                'temp_max': 'TmaxD'
            },
            'target': {
                'precip': 'RhiresD',
                'temp': 'TabsD',
                'temp_min': 'TminD',
                'temp_max': 'TmaxD'
            }
        },
        'preprocessing': {
            'nan_to_num': True,
            'nan_value': 0.0
        }
    }
)
dm.setup()
test_loader = dm.test_dataloader()

with open(paths.DATASETS_TRAINING_DIR+"/RhiresD_scaling_params.json", 'r') as f:
    pr_params = json.load(f)
with open(paths.DATASETS_TRAINING_DIR+"/TabsD_scaling_params.json", 'r') as f:
    temp_params = json.load(f)
with open(paths.DATASETS_TRAINING_DIR+"/TminD_scaling_params.json", 'r') as f:
    temp_min_params = json.load(f)
with open(paths.DATASETS_TRAINING_DIR+"/TmaxD_scaling_params.json", 'r') as f:
    temp_max_params = json.load(f)

def denorm_pr(x):
    return np.exp(x * pr_params['std'] + pr_params['mean']) - pr_params['epsilon']

def denorm_temp(x, params):
    return x * params['std'] + params['mean']

def denorm_all(x):
    arr = x.cpu().numpy() if torch.is_tensor(x) else x
    out = np.empty_like(arr)
    for i, (var, params) in enumerate([
        ("precip", pr_params),
        ("temp", temp_params),
        ("temp_min", temp_min_params),
        ("temp_max", temp_max_params)
    ]):
        if var == "precip":
            out[i] = denorm_pr(arr[i])
        else:
            out[i] = denorm_temp(arr[i], params)
    return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ref_ds = xr.open_dataset(test_input_paths['precip'])
dates = ref_ds['time'].values
lat2d = ref_ds["lat"].values
lon2d = ref_ds["lon"].values
ref_ds.close()




model_UNet = DownscalingUnetLightning(
    in_ch=5, out_ch=4, features=[64, 128, 256, 512],
    channel_names=["precip", "temp", "temp_min", "temp_max"]
)
unet_state_dict = torch.load(paths.LDM_DIR+"/trained_ckpts/10km/LDM_conditional.models.unet_module.DownscalingUnetLightning_checkpoint.ckpt", map_location="cpu")["state_dict"]
model_UNet.load_state_dict(unet_state_dict, strict=False)
model_UNet = model_UNet.to(device)
model_UNet.eval()




encoder = SimpleConvEncoder(in_dim=4, levels=2, min_ch=16, ch_mult=4)
decoder = SimpleConvDecoder(in_dim=64, levels=2, min_ch=16, out_dim=4, ch_mult=4)
unet_regr = DownscalingUnetLightning(
    in_ch=5, out_ch=4, features=[64, 128, 256, 512],
    channel_names=["precip", "temp", "temp_min", "temp_max"]
)
unet_regr_ckpt = torch.load(paths.LDM_DIR+"/trained_ckpts/10km/LDM_conditional.models.unet_module.DownscalingUnetLightning_checkpoint.ckpt", map_location="cpu")["state_dict"]
unet_regr.load_state_dict(unet_regr_ckpt, strict=False)
unet_regr.eval()
vae_model = AutoencoderKL(encoder=encoder, decoder=decoder, kl_weight=0.01, ae_flag="residual", unet_regr=unet_regr)
vae_ckpt = torch.load(
    paths.LDM_DIR+"/trained_ckpts/10km/LDM_conditional.models.ae_module.AutoencoderKL_checkpoint.ckpt",
    map_location="cpu"
)["state_dict"]
vae_model.load_state_dict(vae_ckpt, strict=False)
vae_model = vae_model.to(device)
vae_model.eval()


conditioner = AFNOConditionerNetCascade(
    autoencoder=vae_model,
    embed_dim=[64, 128, 256, 256],
    analysis_depth=4,
    cascade_depth=4,
    context_ch=[64, 128, 256, 256]
)


denoiser = UNetModel(
    model_channels=64,
    in_channels=32,
    out_channels=32,
    num_res_blocks=2,
    attention_resolutions=[1, 2, 4],
    context_ch=[64, 128, 256, 256],
    channel_mult=[1, 2, 4, 4],
    conv_resample=True,
    dims=2,
    use_fp16=False,
    num_heads=4
)


ldm = LatentDiffusion(
    denoiser=denoiser,
    autoencoder=vae_model,
    context_encoder=conditioner,
    timesteps=1000,
    parameterization="v",
    loss_type="l2"
)
ldm_ckpt = torch.load(
    paths.LDM_DIR+"/trained_ckpts/10km/LDM_checkpoint_V1_linear_schedule_250_steps.ckpt",
    map_location=device
)
ldm.load_state_dict(ldm_ckpt["state_dict"], strict=False)
ldm = ldm.to(device)
ldm.eval()

sampler = DDIMSampler(ldm, device=device)


var_names = ["precip", "temp", "temp_min", "temp_max"]

unet_preds_list = []
ldm_samples_list = []


with tqdm(total=len(dates), desc="Frame") as pbar:
    for batch_inputs, batch_targets in test_loader:
        batch_inputs = batch_inputs.to(device)
        with torch.no_grad():
            # UNet coarse pred
            unet_pred = model_UNet(batch_inputs)
            unet_pred_denorm = denorm_all(unet_pred[0].cpu())
            unet_preds_list.append(unet_pred_denorm)

            # LDM
            ldm_samples_this_frame = []
            for _ in range(N_SAMPLES):
                # --- PRIOR SAMPLING: sample z ~ N(0, I) ---
                latent_shape = (1, 32, unet_pred.shape[2] // 4, unet_pred.shape[3] // 4)
                z = torch.randn(latent_shape, device=device)

                conditioning = [(unet_pred, None)]
                sampled_latent, _ = sampler.sample(
                    S=S,
                    batch_size=1,
                    shape=latent_shape[1:],
                    conditioning=conditioning,
                    eta=0.1,
                    verbose=False,
                    x_T=z,  # Pass sampled prior as initial latent
                )
                generated_residual = vae_model.decode(sampled_latent)

                if generated_residual.shape != unet_pred.shape:
                    _, _, h1, w1 = unet_pred.shape
                    _, _, h2, w2 = generated_residual.shape
                    crop_h = min(h1, h2)
                    crop_w = min(w1, w2)
                    start_h = (h2 - crop_h) // 2
                    start_w = (w2 - crop_w) // 2
                    generated_residual = generated_residual[:, :, start_h:start_h+crop_h, start_w:start_w+crop_w]
                    if h1 != crop_h or w1 != crop_w:
                        start_h_u = (h1 - crop_h) // 2
                        start_w_u = (w1 - crop_w) // 2
                        unet_pred_crop = unet_pred[:, :, start_h_u:start_h_u+crop_h, start_w_u:start_w_u+crop_w]
                    else:
                        unet_pred_crop = unet_pred
                else:
                    unet_pred_crop = unet_pred
                final_pred = unet_pred_crop + generated_residual
                final_pred_denorm = denorm_all(final_pred[0].cpu())
                ldm_samples_this_frame.append(final_pred_denorm)
            ldm_samples_list.append(np.stack(ldm_samples_this_frame))
        pbar.update(1)

unet_preds_np = np.stack(unet_preds_list)  
ldm_samples_np = np.stack(ldm_samples_list)  


unet_preds_np = np.transpose(unet_preds_np, (0, 2, 3, 1))
ds_unet = xr.Dataset(
    {
        var: (("time", "N", "E"), unet_preds_np[:, :, :, i])
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
ds_unet.to_netcdf(paths.LDM_DIR + "/outputs/test_UNet_baseline.nc", encoding=encoding)
print(f"UNet baseline saved with shape: {unet_preds_np.shape}")

ldm_samples_np = np.transpose(ldm_samples_np, (0, 1, 3, 4, 2))
ds_ldm = xr.Dataset(
    {
        var: (("time", "sample", "N", "E"), ldm_samples_np[:, :, :, :, i])
        for i, var in enumerate(var_names)
    },
    coords={
        "time": dates,
        "sample": np.arange(N_SAMPLES),
        "N": np.arange(lat2d.shape[0]),
        "E": np.arange(lat2d.shape[1]),
        "lat": (("N", "E"), lat2d),
        "lon": (("N", "E"), lon2d),
    }
)
encoding_ldm = {var: {"_FillValue": np.nan} for var in var_names}
ds_ldm.to_netcdf(paths.LDM_DIR + "/outputs/test_LDM_samples_eta_0.1.nc", encoding=encoding_ldm)
print(f"LDM samples saved with shape: {ldm_samples_np.shape}")