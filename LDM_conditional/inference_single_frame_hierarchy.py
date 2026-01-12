import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import json
import sys

sys.path.append("..")
sys.path.append("../..")

from models.unet_module import DownscalingUnetLightning
from models.ae_module import AutoencoderKL
from models.components.ae import SimpleConvEncoder, SimpleConvDecoder
from models.components.ldm.denoiser import UNetModel
from models.components.ldm.denoiser.ddim import DDIMSampler
from models.components.ldm.conditioner import AFNOConditionerNetCascade
from models.ldm_module import LatentDiffusion
from DownscalingDataModule import DownscalingDataModule

def denorm_pr(x, pr_params):
    return np.exp(x * pr_params['std'] + pr_params['mean']) - pr_params['epsilon']

def denorm_temp(x, params):
    return x * params['std'] + params['mean']



#Running from downscaling models due to sourcing env diffscaler.sh


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def main(idx):
    with open("Dataset_Setup_I_Chronological_12km/RhiresD_scaling_params.json", 'r') as f:
        pr_params = json.load(f)
    with open("Dataset_Setup_I_Chronological_12km/TabsD_scaling_params.json", 'r') as f:
        temp_params = json.load(f)

    # Data paths
    train_input_paths = {
        'precip': "Dataset_Setup_I_Chronological_12km/RhiresD_input_train_scaled.nc",
        'temp': "Dataset_Setup_I_Chronological_12km/TabsD_input_train_scaled.nc",
    }
    train_target_paths = {
        'precip': "Dataset_Setup_I_Chronological_12km/RhiresD_target_train_scaled.nc",
        'temp':  "Dataset_Setup_I_Chronological_12km/TabsD_target_train_scaled.nc",
    }
    val_input_paths = {
        'precip': "Dataset_Setup_I_Chronological_12km/RhiresD_input_val_scaled.nc",
        'temp': "Dataset_Setup_I_Chronological_12km/TabsD_input_val_scaled.nc",
    }
    val_target_paths = {
        'precip': "Dataset_Setup_I_Chronological_12km/RhiresD_target_val_scaled.nc",
        'temp': "Dataset_Setup_I_Chronological_12km/TabsD_target_val_scaled.nc",
    }
    test_input_paths = {
        'precip': "Dataset_Setup_I_Chronological_12km/RhiresD_input_test_scaled.nc",
        'temp': "Dataset_Setup_I_Chronological_12km/TabsD_input_test_scaled.nc",
    }
    test_target_paths = {
        'precip': "Dataset_Setup_I_Chronological_12km/RhiresD_target_test_scaled.nc",
        'temp': "Dataset_Setup_I_Chronological_12km/TabsD_target_test_scaled.nc",
    }
    elevation_path = 'elevation.tif'




    # DM
    dm = DownscalingDataModule(
        train_input=train_input_paths,
        train_target=train_target_paths,
        val_input=val_input_paths,
        val_target=val_target_paths,
        test_input=test_input_paths,
        test_target=test_target_paths,
        elevation=elevation_path,
        batch_size=32,
        num_workers=4,
        preprocessing={
            'variables': {
                'input': {'precip': 'RhiresD', 'temp': 'TabsD'},
                'target': {'precip': 'RhiresD', 'temp': 'TabsD'}
            },
            'preprocessing': {'nan_to_num': True, 'nan_value': 0.0}
        }
    )


    dm.setup()
    test_loader = dm.test_dataloader()
    test_inputs, test_targets = next(iter(test_loader))



    # UNet
    unet_regr = DownscalingUnetLightning(
        in_ch=3,
        out_ch=2,
        features=[64, 128, 256, 512],
        channel_names=["precip", "temp"],
        precip_scaling_json="Dataset_Setup_I_Chronological_12km/RhiresD_scaling_params.json",
    )


    unet_regr_ckpt = torch.load(
        "LDM_conditional/trained_ckpts_optimised/12km/LDM_conditional.models.unet_module.DownscalingUnetLightning_logtransform_lr0.01_precip_loss_weight5.0_1.0_crps[0, 1]_factor0.5_pat3.ckpt.ckpt",
        map_location="cpu", weights_only=False
    )["state_dict"]
    unet_regr.load_state_dict(unet_regr_ckpt, strict=False)
    unet_regr = unet_regr.to(device)
    unet_regr.eval()



    # VAE
    encoder = SimpleConvEncoder(in_dim=2, levels=2, min_ch=16, ch_mult=4)
    decoder = SimpleConvDecoder(in_dim=16, levels=2, min_ch=16, out_dim=2, ch_mult=4)
    vae_model = AutoencoderKL(
        encoder=encoder,
        decoder=decoder,
        kl_weight=0.001,
        ae_flag="residual",
        unet_regr=unet_regr,
        latent_dim=16
    )
    vae_ckpt = torch.load(
        "LDM_conditional/trained_ckpts_optimised/12km/VAE_ckpts/VAE_levels_latentdim_16_klweight_0.001_checkpoint.ckpt",
        map_location="cpu"
    )["state_dict"]
    vae_model.load_state_dict(vae_ckpt, strict=False)
    vae_model = vae_model.to(device)

    vae_model.eval()


    denoiser = UNetModel(
        model_channels=64,
        in_channels=16,
        out_channels=16,
        num_res_blocks=2,
        attention_resolutions=[1, 2, 4],
        context_ch=[64, 128, 256, 256],
        channel_mult=[1, 2, 4, 4],
        conv_resample=True,
        dims=2,
        use_fp16=False,
        num_heads=4
    )



    conditioner = AFNOConditionerNetCascade(
        autoencoder=vae_model,
        embed_dim=[64, 128, 256, 256],
        analysis_depth=4,
        cascade_depth=4,
        context_ch=[64, 128, 256, 256]
    )



    ldm = LatentDiffusion(
        denoiser=denoiser,
        autoencoder=vae_model,
        context_encoder=conditioner,
        timesteps=1000,
        parameterization="v",
        loss_type="l2",
        beta_schedule="quadratic",   
        linear_start=1e-4,       
        linear_end=2e-2,            
        cosine_s=8e-3               
    )



    ldm_ckpt = torch.load(
        "LDM_conditional/trained_ckpts_optimised/12km/LDM_ckpts/LDM_ckpt_model.parameterization=0_model.timesteps=0_model.noise_schedule=0_lr0.0001_latent_dim16_checkpoint.ckpt",
        map_location=device
    )



    ldm.load_state_dict(ldm_ckpt["state_dict"], strict=False)
    ldm = ldm.to(device)
    ldm.eval()
    sampler = DDIMSampler(ldm, device=device)


    # inf
    with torch.no_grad():
        input_sample = test_inputs[idx].unsqueeze(0).to(device)  # (1, C_in, H, W)
        unet_pred = unet_regr(input_sample)                   

        mu, logvar = vae_model.encode(unet_pred)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_vae = mu + eps * std



        latent_shape = (1, 16, unet_pred.shape[2] // 4, unet_pred.shape[3] // 4)
        z = torch.randn(latent_shape, device=device)


        sampled_latent, _ = sampler.sample(
            S=1000,
            batch_size=1,
            shape=latent_shape[1:],
            conditioning=[(unet_pred, None)],
            eta=0.0,
            verbose=True,
            x_T=z,
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
                unet_pred = unet_pred[:, :, start_h_u:start_h_u+crop_h, start_w_u:start_w_u+crop_w]
        final_pred = unet_pred + generated_residual




        final_pred_np = final_pred[0].cpu().numpy()
        unet_pred_np = unet_pred[0].cpu().numpy()
        input_np = input_sample[0, :unet_pred_np.shape[0]].cpu().numpy()
        target_np = test_targets[idx][:unet_pred_np.shape[0]].cpu().numpy()

        channel_names = ["Precip", "Temp"]
        params_list = [pr_params, temp_params]

        input_denorm = np.empty_like(input_np)
        for i, params in enumerate(params_list):
            input_denorm[i] = denorm_pr(input_np[i], pr_params) if i == 0 else denorm_temp(input_np[i], params)
        unet_pred_denorm = np.empty_like(unet_pred_np)
        for i, params in enumerate(params_list):
            unet_pred_denorm[i] = denorm_pr(unet_pred_np[i], pr_params) if i == 0 else denorm_temp(unet_pred_np[i], params)
        ldm_pred_denorm = np.empty_like(final_pred_np)
        for i, params in enumerate(params_list):
            ldm_pred_denorm[i] = denorm_pr(final_pred_np[i], pr_params) if i == 0 else denorm_temp(final_pred_np[i], params)
        target_denorm = np.empty_like(target_np)
        for i, params in enumerate(params_list):
            target_denorm[i] = denorm_pr(target_np[i], pr_params) if i == 0 else denorm_temp(target_np[i], params)

        vmins = [min(input_denorm[j].min(), unet_pred_denorm[j].min(), ldm_pred_denorm[j].min(), target_denorm[j].min()) for j in range(len(params_list))]
        vmaxs = [max(input_denorm[j].max(), unet_pred_denorm[j].max(), ldm_pred_denorm[j].max(), target_denorm[j].max()) for j in range(len(params_list))]

        fig, axes = plt.subplots(4, len(params_list), figsize=(5*len(params_list), 12), dpi=150)
        if len(params_list) == 1:
            axes = axes[:, np.newaxis]
        for j in range(len(params_list)):
            axes[0, j].imshow(np.flipud(input_denorm[j]), cmap='coolwarm', vmin=vmins[j], vmax=vmaxs[j])
            axes[0, j].set_title(f"Input (denorm) {channel_names[j]}")
            axes[0, j].axis('off')
            axes[1, j].imshow(np.flipud(unet_pred_denorm[j]), cmap='coolwarm', vmin=vmins[j], vmax=vmaxs[j])
            axes[1, j].set_title(f"UNet Output (denorm) {channel_names[j]}")
            axes[1, j].axis('off')
            axes[2, j].imshow(np.flipud(ldm_pred_denorm[j]), cmap='coolwarm', vmin=vmins[j], vmax=vmaxs[j])
            axes[2, j].set_title(f"LDM Output (denorm) {channel_names[j]}")
            axes[2, j].axis('off')
            axes[3, j].imshow(np.flipud(target_denorm[j]), cmap='coolwarm', vmin=vmins[j], vmax=vmaxs[j])
            axes[3, j].set_title(f"Ground Truth (denorm) {channel_names[j]}")
            axes[3, j].axis('off')
            cbar = fig.colorbar(axes[0, j].images[0], ax=axes[:, j], fraction=0.02, pad=0.01)
            cbar.ax.set_ylabel(channel_names[j])

        fig.savefig(f"LDM_conditional/outputs/debug_output_{idx}_ldm_hierarchy.png")
        print(f"Plot saved as debug_output_{idx}_ldm_hierarchy.png")



        #Printing stats : debug
        print("unet_pred shape:", unet_pred.shape)
        print("sampled_latent shape:", sampled_latent.shape)
        print("unet_pred stats:", unet_pred.min().item(), unet_pred.max().item(), torch.isnan(unet_pred).any().item())
        print("generated_residual stats:", generated_residual.min().item(), generated_residual.max().item(), torch.isnan(generated_residual).any().item())
        print("final_pred stats:", final_pred.min().item(), final_pred.max().item(), torch.isnan(final_pred).any().item())
        print("sampled_latent stats:", sampled_latent.min().item(), sampled_latent.max().item(), torch.isnan(sampled_latent).any().item())
        print("z stats:", z.min().item(), z.max().item(), torch.isnan(z).any().item())
        print("z_vae stats:", z_vae.min().item(), z_vae.max().item(), torch.isnan(z_vae).any().item())
        print("mu stats:", mu.min().item(), mu.max().item(), torch.isnan(mu).any().item())
        print("std stats:", std.min().item(), std.max().item(), torch.isnan(std).any().item())



        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=26, help="index to plot")
    args = parser.parse_args()
    main(args.idx)