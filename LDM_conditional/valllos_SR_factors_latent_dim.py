import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append("..")
sys.path.append("../..")
import paths
from LDM_conditional.DownscalingDataModule import DownscalingDataModule
from LDM_conditional.models.components.unet import DownscalingUnet
from models.components.ae import SimpleConvEncoder, SimpleConvDecoder
from models.ae_module import AutoencoderKL

sr_factors = ['12km', '24km', '36km', '48km']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt_dirs = {
    '12km': 'trained_ckpts_optimised/12km/VAE_ckpts',
    '24km': 'trained_ckpts_optimised/24km/VAE_ckpts',
    '36km': 'trained_ckpts_optimised/36km/VAE_ckpts',
    '48km': 'trained_ckpts_optimised/48km/VAE_ckpts',
}

pattern = r"VAE_levels_latentdim_(\d+)_klweight_([0-9.]+)_checkpoint\.ckpt"

unet_ckpt_paths = {
    '12km': "trained_ckpts_optimised/12km/UNet_ckpts/LDM_conditional.models.unet_module.DownscalingUnetLightning_logtransform_lr0.001_precip_loss_weight1.0_1.0_crps[0, 1]_factor0.5_pat3.ckpt.ckpt",
    '24km': "trained_ckpts_optimised/24km/LDM_conditional.models.unet_module.DownscalingUnetLightning_24km_logtransform_lr0.001_precip_loss_weight5.0_1.0_crps[0, 1]_factor0.5_pat3.ckpt.ckpt",
    '36km': "trained_ckpts_optimised/36km/LDM_conditional.models.unet_module.DownscalingUnetLightning_36km_logtransform_lr0.001_precip_loss_weight5.0_1.0_crps[0, 1]_factor0.5_pat3.ckpt.ckpt",
    '48km': "trained_ckpts_optimised/48km/LDM_conditional.models.unet_module.DownscalingUnetLightning_48km_logtransform_lr0.001_precip_loss_weight5.0_1.0_crps[0, 1]_factor0.5_pat3.ckpt.ckpt",
}

val_data_dirs = {
    '12km': '../Dataset_Setup_I_Chronological_12km',
    '24km': '../Dataset_Setup_I_Chronological_24km',
    '36km': '../Dataset_Setup_I_Chronological_36km',
    '48km': '../Dataset_Setup_I_Chronological_48km',
}

elevation_path = paths.BASE_DIR+"/sasthana/Downscaling/Downscaling_Models/elevation.tif"

all_latent_dims = []
all_sr_labels = []
all_mae_losses = []

for sr in sr_factors:
    unet_ckpt_path = unet_ckpt_paths[sr]
    unet = DownscalingUnet(in_ch=3, out_ch=2, features=[64,128,256,512])
    unet_ckpt = torch.load(unet_ckpt_path, map_location="cpu",weights_only=False)
    unet.load_state_dict(unet_ckpt["state_dict"], strict=False)
    unet = unet.to(device)
    unet.eval()

    ckpt_dir = ckpt_dirs[sr]
    val_dir = val_data_dirs[sr]

    val_input_paths = {
        'precip': f"{val_dir}/RhiresD_input_val_scaled.nc",
        'temp': f"{val_dir}/TabsD_input_val_scaled.nc",
    }
    val_target_paths = {
        'precip': f"{val_dir}/RhiresD_target_val_scaled.nc",
        'temp': f"{val_dir}/TabsD_target_val_scaled.nc",
    }
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

    for fname in os.listdir(ckpt_dir):
        match = re.match(pattern, fname)
        if not match:
            continue
        latent_dim = int(match.group(1))
        kl_weight = float(match.group(2))
        if kl_weight != 0.01:
            continue
        ckpt_path = os.path.join(ckpt_dir, fname)
        print(f"Evaluating {fname} for {sr} ...")
        # VAE
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
        total_squared_error = 0.0
        n_pixels = 0
        with torch.no_grad():
            for batch_inputs, batch_targets in tqdm(val_loader, desc=f"Val {sr} {latent_dim}-{kl_weight}", leave=False):
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)
                regression_output = unet(batch_inputs)
                residuals = batch_targets - regression_output
                recon, *_ = vae(residuals)
                squared_error = torch.nn.functional.mse_loss(recon, residuals, reduction='sum')
                total_squared_error += squared_error.item()
                n_pixels += residuals.numel()
        norm_rmse = np.sqrt(total_squared_error / n_pixels)

        all_latent_dims.append(latent_dim)
        all_sr_labels.append(sr)
        all_mae_losses.append(norm_rmse)

plt.figure(figsize=(12, 8))
sr_label_indices = {sr: i for i, sr in enumerate(sr_factors)}
y_vals = [sr_label_indices[sr] for sr in all_sr_labels]

color_map = plt.get_cmap('tab10')
colors = [color_map(sr_label_indices[sr]) for sr in all_sr_labels]

sizes = np.array(all_mae_losses)
sizes = 500 * (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-8) + 50 

plt.scatter(all_latent_dims, y_vals, s=sizes, c=colors, edgecolors='black', alpha=0.7)

for x, y, mae in zip(all_latent_dims, y_vals, all_mae_losses):
    plt.text(x, y, f"{mae:.3f}", va='center', ha='center', fontsize=7, color='white')

plt.yticks(list(sr_label_indices.values()), list(sr_label_indices.keys()))
plt.xlabel('Latent Dimension', fontsize=15)
plt.ylabel('SR Factor', fontsize=15)
plt.title('SR Factor vs Latent Dimension\n(Bubble size = Normalised RMSE on Val Set, Color = SR Factor, KL=0.01)', fontsize=17)
plt.grid(True, axis='x')
plt.tight_layout()

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=sr,
           markerfacecolor=color_map(i), markersize=10, markeredgecolor='black')
    for i, sr in enumerate(sr_factors)
]
plt.legend(handles=legend_elements, title='SR Factor')

plt.savefig("outputs/bubble_valloss_vs_SR_factors_colored.png", dpi=1000)
plt.show()