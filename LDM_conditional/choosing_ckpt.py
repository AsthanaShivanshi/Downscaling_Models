import os
import re
from LDM_conditional.models.components.unet import DownscalingUnet
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append("..")
sys.path.append("../..")
import paths
from LDM_conditional.DownscalingDataModule import DownscalingDataModule
from models.components.ae import SimpleConvEncoder, SimpleConvDecoder
from models.ae_module import AutoencoderKL

val_input_paths = {
    'precip': paths.DATASETS_TRAINING_DIR+"/RhiresD_input_val_scaled.nc",
    'temp': paths.DATASETS_TRAINING_DIR+"/TabsD_input_val_scaled.nc",
}
val_target_paths = {
    'precip': paths.DATASETS_TRAINING_DIR+"/RhiresD_target_val_scaled.nc",
    'temp': paths.DATASETS_TRAINING_DIR+"/TabsD_target_val_scaled.nc",
}
elevation_path = paths.BASE_DIR+"/sasthana/Downscaling/Downscaling_Models/elevation.tif"

dm = DownscalingDataModule(
    train_input=None, train_target=None,
    val_input=val_input_paths, val_target=val_target_paths,
    test_input=None, test_target=None,
    elevation=elevation_path,
    batch_size=32,
    num_workers=1,
    preprocessing={
        'variables': {
            'input': {'precip': 'RhiresD', 'temp': 'TabsD'},
            'target': {'precip': 'RhiresD', 'temp': 'TabsD'}
        },
        'preprocessing': {'nan_to_num': True, 'nan_value': 0.0}
    }
)
dm.setup()
val_loader = dm.val_dataloader()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#UNet ckpt
unet_ckpt_path = "trained_ckpts_optimised/12km/LDM_conditional.models.unet_module.DownscalingUnetLightning_logtransform_lr0.001_precip_loss_weight1.0_1.0_crps[0, 1]_factor0.5_pat3.ckpt.ckpt"
unet = DownscalingUnet(in_ch=3, out_ch=2, features=[64,128,256,512])
unet_ckpt = torch.load(unet_ckpt_path, map_location="cpu",weights_only=False)
unet.load_state_dict(unet_ckpt["state_dict"], strict=False)
unet = unet.to(device)
unet.eval()

#For all VAE checkppints::::

ckpt_dir = "trained_ckpts_optimised/12km/VAE_ckpts"
pattern = r"VAE_levels_latentdim_(\d+)_klweight_([0-9.]+)_checkpoint\.ckpt"

results = []

for fname in os.listdir(ckpt_dir):
    match = re.match(pattern, fname)
    if not match:
        continue
    latent_dim = int(match.group(1))
    kl_weight = float(match.group(2))
    ckpt_path = os.path.join(ckpt_dir, fname)
    print(f"Evaluating {fname} ...")

    in_dim = 2
    levels = 2
    min_ch = 16
    ch_mult = 4
    out_dim = 2

    encoder = SimpleConvEncoder(in_dim=in_dim, levels=levels, min_ch=min_ch, ch_mult=ch_mult)
    decoder = SimpleConvDecoder(in_dim=latent_dim, levels=levels, min_ch=min_ch, out_dim=out_dim)
    vae = AutoencoderKL(encoder=encoder, decoder=decoder, latent_dim=latent_dim)
    ckpt = torch.load(ckpt_path, map_location="cpu",weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    vae.load_state_dict(state_dict, strict=False)
    vae = vae.to(device)
    vae.eval()

    total_loss = 0.0
    n_samples = 0


    with torch.no_grad():
        for batch_inputs, batch_targets in tqdm(val_loader, desc=f"Val {latent_dim}-{kl_weight}", leave=False):
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            # regression output and residuals
            regression_output = unet(batch_inputs)
            residuals = batch_targets - regression_output
            # VAE recon res
            recon, *_ = vae(residuals) #First element is recon, unpacked and selected
            loss = torch.nn.functional.l1_loss(recon, residuals, reduction='sum')
            total_loss += loss.item()
            n_samples += residuals.numel()

    avg_loss = total_loss / n_samples
    print(f"latent_dim={latent_dim}, kl_weight={kl_weight}, val_recon_loss={avg_loss:.6f}")
    results.append((latent_dim, kl_weight, avg_loss))


results = np.array(results)
latent_dims = results[:,0] #three

kl_weights = results[:,1]
val_losses = results[:,2]

kl_weights_list= [0.1,0.01,0.001]
colors = {k: c for k, c in zip(sorted(kl_weights_list), ['orange', 'purple', 'cyan'])}
labels = {k: f"KL weight {k}" for k in kl_weights_list}

plt.figure(figsize=(12, 8))

for kl in kl_weights_list:

    mask = kl_weights == kl
    plt.scatter(latent_dims[mask], val_losses[mask], color=colors[kl], label=labels[kl], s=120)

plt.xlabel('latent dim')
plt.ylabel('Validation recon Loss (MAE)')
plt.title('(colored by kl weight): Val loss vs latent dim for 9 VAE ckpts')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/vae_latentdim_vs_valloss_by_klweight.png")
plt.show()