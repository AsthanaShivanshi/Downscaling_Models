import torch
import numpy as np
import json
import xarray as xr
from tqdm import tqdm
import paths

from models.unet_module import DownscalingUnetLightning
from models.ae_module import AutoencoderKL
from models.components.ae import SimpleConvEncoder, SimpleConvDecoder
from models.components.ldm.conditioner import AFNOConditionerNetCascade
from models.components.ldm.denoiser.unet import UNetModel as DenoiserUNetModel
from models.ldm_module import LatentDiffusion
from models.components.ldm.denoiser.ddim import DDIMSampler
from DownscalingDataModule import DownscalingDataModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_samples = 10 


with open(paths.DATASETS_TRAINING_DIR + "/RhiresD_scaling_params.json") as f:
    pr_params = json.load(f)
with open(paths.DATASETS_TRAINING_DIR + "/TabsD_scaling_params.json") as f:
    temp_params = json.load(f)
with open(paths.DATASETS_TRAINING_DIR + "/TminD_scaling_params.json") as f:
    temp_min_params = json.load(f)
with open(paths.DATASETS_TRAINING_DIR + "/TmaxD_scaling_params.json") as f:
    temp_max_params = json.load(f)

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

# Define train and val input/target paths (update file paths as needed)
train_input_paths = {
    'precip':  paths.DATASETS_TRAINING_DIR + "/RhiresD_input_train_scaled.nc",
    'temp': paths.DATASETS_TRAINING_DIR + "/TabsD_input_train_scaled.nc",
    'temp_min': paths.DATASETS_TRAINING_DIR + "/TminD_input_train_scaled.nc",
    'temp_max': paths.DATASETS_TRAINING_DIR + "/TmaxD_input_train_scaled.nc"
}
train_target_paths = {
    'precip': paths.DATASETS_TRAINING_DIR + "/RhiresD_target_train_scaled.nc",
    'temp': paths.DATASETS_TRAINING_DIR + "/TabsD_target_train_scaled.nc",
    'temp_min': paths.DATASETS_TRAINING_DIR + "/TminD_target_train_scaled.nc",
    'temp_max': paths.DATASETS_TRAINING_DIR + "/TmaxD_target_train_scaled.nc"
}
val_input_paths = {
    'precip': paths.DATASETS_TRAINING_DIR + "/RhiresD_input_val_scaled.nc",
    'temp': paths.DATASETS_TRAINING_DIR + "/TabsD_input_val_scaled.nc",
    'temp_min': paths.DATASETS_TRAINING_DIR + "/TminD_input_val_scaled.nc",
    'temp_max': paths.DATASETS_TRAINING_DIR + "/TmaxD_input_val_scaled.nc"
}
val_target_paths = {
    'precip': paths.DATASETS_TRAINING_DIR + "/RhiresD_target_val_scaled.nc",
    'temp': paths.DATASETS_TRAINING_DIR + "/TabsD_target_val_scaled.nc",
    'temp_min': paths.DATASETS_TRAINING_DIR + "/TminD_target_val_scaled.nc",
    'temp_max': paths.DATASETS_TRAINING_DIR + "/TmaxD_target_val_scaled.nc"
}

test_input_paths= {
    'precip': paths.DATASETS_TRAINING_DIR + "/RhiresD_input_test_scaled.nc",
    'temp': paths.DATASETS_TRAINING_DIR + "/TabsD_input_test_scaled.nc",
    'temp_min': paths.DATASETS_TRAINING_DIR + "/TminD_input_test_scaled.nc",
    'temp_max': paths.DATASETS_TRAINING_DIR + "/TmaxD_input_test_scaled.nc"
}

test_target_paths = {
    'precip': paths.DATASETS_TRAINING_DIR + "/RhiresD_target_test_scaled.nc",
    'temp': paths.DATASETS_TRAINING_DIR + "/TabsD_target_test_scaled.nc",
    'temp_min': paths.DATASETS_TRAINING_DIR + "/TminD_target_test_scaled.nc",
    'temp_max': paths.DATASETS_TRAINING_DIR + "/TmaxD_target_test_scaled.nc"
}
elevation_path = paths.BASE_DIR + "/sasthana/Downscaling/Downscaling_Models/elevation.tif"



dm = DownscalingDataModule(
    train_input= train_input_paths,  
    train_target= train_target_paths,
    val_input= val_input_paths,
    val_target= val_target_paths,
    test_input= test_input_paths,
    test_target= test_target_paths,
    elevation=elevation_path,
    batch_size=1,
    num_workers=4,
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
dm.setup(stage="test")
test_loader = dm.test_dataloader()

# UNet regr
unet_model = DownscalingUnetLightning(
    in_ch=5,
    out_ch=4,
    features=[64, 128, 256, 512],
    channel_names=["precip", "temp", "temp_min", "temp_max"]
)
unet_ckpt = torch.load(paths.LDM_DIR + "/trained_ckpts/10km/LDM_conditional.models.unet_module.DownscalingUnetLightning_checkpoint.ckpt", map_location=device)
unet_model.load_state_dict(unet_ckpt["state_dict"], strict=False)
unet_model = unet_model.to(device)
unet_model.eval()

# VAE
encoder = SimpleConvEncoder(in_dim=4, levels=2, min_ch=16, ch_mult=4)
decoder = SimpleConvDecoder(in_dim=64, levels=2, min_ch=16, out_dim=4, ch_mult=4)
vae = AutoencoderKL(
    encoder=encoder,
    decoder=decoder,
    kl_weight=0.01,
    ae_flag="residual",
    unet_regr=unet_model
)
vae_ckpt = torch.load(paths.LDM_DIR + "/trained_ckpts/10km/LDM_conditional.models.ae_module.AutoencoderKL_checkpoint.ckpt", map_location=device)
vae.load_state_dict(vae_ckpt["state_dict"], strict=False)
vae = vae.to(device)
vae.eval()

# Conditioner
conditioner = AFNOConditionerNetCascade(
    autoencoder=vae,
    embed_dim=[64, 128, 256, 256],
    analysis_depth=4,
    cascade_depth=4,
    context_ch=[64, 128, 256, 256]
)
conditioner = conditioner.to(device)
conditioner.eval()

# Denoiser 
denoiser = DenoiserUNetModel(
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

# LDM
ldm = LatentDiffusion(
    denoiser=denoiser,
    autoencoder=vae,
    context_encoder=conditioner,
    timesteps=1000,
    parameterization="v",
    loss_type="l2"
)
ldm_ckpt = torch.load(paths.LDM_DIR + "/trained_ckpts/10km/LDM_checkpoint_V1_linear_schedule_250_steps.ckpt", map_location=device)
ldm.load_state_dict(ldm_ckpt["state_dict"], strict=False)
ldm = ldm.to(device)
ldm.eval()

# Sampler
sampler = DDIMSampler(
    model=ldm,
    device=device,
    ddim_num_steps=250,
    ddim_eta=0.0
)

all_samples = []
all_unet_preds = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Frames"):
        x_in, y_true = batch
        x_in = x_in.to(device)
        y_true = y_true.to(device)
        batch_samples = []

        coarse_pred = unet_model(x_in)  # (1, 4, H, W)
        unet_pred_np = coarse_pred[0].cpu().numpy()
        unet_pred_denorm = denorm_sample(unet_pred_np)
        all_unet_preds.append(unet_pred_denorm)

        residual = y_true - coarse_pred  # (1, 4, H, W)
        mu, logvar = vae.encode(residual)
        std = torch.exp(0.5 * logvar)

        for s in range(n_samples):
            eps = torch.randn_like(std)
            z = mu + eps * std  # Sample latent

            context = conditioner([(coarse_pred, None)])

            samples, _ = sampler.sample(
                S=250,
                batch_size=x_in.shape[0],
                shape=z.shape[1:],  # (latent_ch, H, W)
                conditioning=context,
            )
            generated_residual = vae.decode(samples)

            if generated_residual.shape != coarse_pred.shape:
                _, _, h1, w1 = coarse_pred.shape
                _, _, h2, w2 = generated_residual.shape
                crop_h = min(h1, h2)
                crop_w = min(w1, w2)
                start_h = (h2 - crop_h) // 2
                start_w = (w2 - crop_w) // 2
                generated_residual = generated_residual[:, :, start_h:start_h+crop_h, start_w:start_w+crop_w]
                if h1 != crop_h or w1 != crop_w:
                    start_h_u = (h1 - crop_h) // 2
                    start_w_u = (w1 - crop_w) // 2
                    coarse_pred = coarse_pred[:, :, start_h_u:start_h_u+crop_h, start_w_u:start_w_u+crop_w]

            final_pred = coarse_pred + generated_residual
            decoded_np = final_pred[0].cpu().numpy()
            decoded_np = denorm_sample(decoded_np)
            batch_samples.append(decoded_np)

        batch_samples = np.stack(batch_samples, axis=0)  # (n_samples, 4, H, W)
        all_samples.append(batch_samples)

all_samples = np.stack(all_samples, axis=0)  # (time, n_samples, 4, H, W)
all_unet_preds = np.stack(all_unet_preds, axis=0)  # (time, 4, H, W)


ref_ds = xr.open_dataset("Dataset_Setup_I_Chronological_10km/RhiresD_target_test_scaled.nc")
times = ref_ds["time"].values
lat = ref_ds["lat"].values
lon = ref_ds["lon"].values
ref_ds.close()

var_names = ["precip", "temp", "temp_min", "temp_max"]



# preds
ds_unet = xr.DataArray(
    all_unet_preds,
    dims=["time", "variable", "y", "x"],
    coords={
        "time": times[:all_unet_preds.shape[0]],
        "variable": var_names,
        "y": np.arange(lat.shape[0]),
        "x": np.arange(lat.shape[1]),
        "lat": (("y", "x"), lat),
        "lon": (("y", "x"), lon),
    },
    name="unet_preds"
)
ds_unet.to_netcdf(paths.LDM_DIR + "/outputs/unet_testset_preds.nc", encoding={"unet_preds": {"_FillValue": np.nan}})
print("UNet preds shape (time, 4, H, W):", all_unet_preds.shape)



#Samples
ds = xr.DataArray(
    all_samples,
    dims=["time", "sample", "variable", "y", "x"],
    coords={
        "time": times[:all_samples.shape[0]],
        "sample": np.arange(all_samples.shape[1]),
        "variable": var_names,
        "y": np.arange(lat.shape[0]),
        "x": np.arange(lat.shape[1]),
        "lat": (("y", "x"), lat),
        "lon": (("y", "x"), lon),
    },
    name="ldm_samples"
)
ds.to_netcdf(paths.LDM_DIR + "/outputs/ldm_conditional_testset_10_samples.nc", encoding={"ldm_samples": {"_FillValue": np.nan}})
print("All samples shape (time, n_samples, 4, H, W):", all_samples.shape)

