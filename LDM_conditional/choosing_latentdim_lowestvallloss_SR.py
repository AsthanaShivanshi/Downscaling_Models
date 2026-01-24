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
    '12km': "trained_ckpts_optimised/12km/UNet_ckpts/LDM_conditional.models.unet_module.DownscalingUnetLightning_logtransform_lr0.001_precip_loss_weight5.0_1.0_crps[0, 1]_factor0.5_pat3.ckpt.ckpt",
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


latent_dims_best = []
sr_labels = []

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



    best_loss = None
    best_latent_dim = None
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
        #val loss calc
        total_loss = 0.0
        n_samples = 0
        with torch.no_grad():
            for batch_inputs, batch_targets in tqdm(val_loader, desc=f"Val {sr} {latent_dim}-{kl_weight}", leave=False):
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)
                regression_output = unet(batch_inputs)
                residuals = batch_targets - regression_output
                recon, *_ = vae(residuals)
                loss = torch.nn.functional.l1_loss(recon, residuals, reduction='sum')
                total_loss += loss.item()
                n_samples += residuals.numel()
        avg_loss = total_loss / n_samples
        if (best_loss is None) or (avg_loss < best_loss):
            best_loss = avg_loss
            best_latent_dim = latent_dim
    latent_dims_best.append(best_latent_dim)
    sr_labels.append(sr)


plt.figure(figsize=(12, 8))

sr_labels = [str(x) for x in sr_factors] 


plt.scatter(latent_dims_best, sr_labels, s=200, color='purple', edgecolors='black')
for x, y in zip(latent_dims_best, sr_labels):
    plt.text(x, y, f"{x}", va='center', ha='left', fontsize=12)

plt.xlabel('Latent Dimension corresponding to lowest validation MAE', fontsize=15)




plt.ylabel('SR Factor', fontsize=15)
plt.title('SR Factor vs VAE Latent Dimension with lowest validation MAE  (KL=0.01) ', fontsize=17)
plt.grid(True, axis='x')
plt.tight_layout()
plt.savefig("outputs/valloss_vs_SR_factors.png", dpi=1000)
plt.show()