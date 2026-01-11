import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import json
import sys

sys.path.append("..")
sys.path.append("../..")

from models.unet_module import DownscalingUnetLightning
from DownscalingDataModule import DownscalingDataModule
from models.components.diff.denoiser.unet import UNetModel
from models.components.diff.denoiser.ddim import DDIMSampler
from models.components.diff.conditioner import AFNOConditionerNetCascade
from models.diff_module import DDIMResidualContextual

def denorm_pr(x, pr_params):
    return np.exp(x * pr_params['std'] + pr_params['mean']) - pr_params['epsilon']

def denorm_temp(x, params):
    return x * params['std'] + params['mean']



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(idx):




    with open("Dataset_Setup_I_Chronological_12km/RhiresD_scaling_params.json", 'r') as f:
        pr_params = json.load(f)
    with open("Dataset_Setup_I_Chronological_12km/TabsD_scaling_params.json", 'r') as f:
        temp_params = json.load(f)








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
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()




    train_inputs, train_targets = next(iter(train_loader))
    val_inputs, val_targets = next(iter(val_loader))
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

    

    # DDIM
    denoiser = UNetModel(
        model_channels=32,
        in_channels=2,
        out_channels=2,
        num_res_blocks=2,
        attention_resolutions=[1, 2, 4],
        context_ch=[32, 64, 128],
        channel_mult=[1, 2, 4],
        conv_resample=True,
        dims=2,
        use_fp16=False,
        num_heads=2
    )
    conditioner = AFNOConditionerNetCascade(
        autoencoder=None,
        input_channels=[2],
        embed_dim=[32, 64, 128],
        analysis_depth=3,
        cascade_depth=3,
        context_ch=[32, 64, 128]
    )
    ddim = DDIMResidualContextual(
        denoiser=denoiser,
        context_encoder=conditioner,
        timesteps=1000,
        parameterization="v",
        loss_type="l2"
    )
    ddim_ckpt = torch.load(
        "DDIM_conditional_derived/trained_ckpts/12km/DDIM_checkpoint_model.parameterization=0_model.timesteps=0_model.beta_schedule=0-v1.ckpt",
        map_location=device
    )
    ddim.load_state_dict(ddim_ckpt["state_dict"], strict=False)
    ddim = ddim.to(device)
    ddim.eval()
    sampler = DDIMSampler(ddim, device=device)

    # inf
    with torch.no_grad():
        input_sample = test_inputs[idx].unsqueeze(0).to(device)  # (1, C_in, H, W)
        unet_pred = unet_regr(input_sample)                      # (1, C_out, H, W)
        context = [(unet_pred, None)]
        sample_shape = unet_pred.shape[1:]                       # (C_out, H, W)
        z = torch.randn((1, *sample_shape), device=device)       # (1, C_out, H, W)

        residual, _ = sampler.sample(
            S=1000,
            batch_size=1,
            shape=sample_shape,
            conditioning=context,
            eta=0.0,
            verbose=False,
            x_T=z,
        )
        final_pred = unet_pred + residual

        final_pred_np = final_pred[0].cpu().numpy()              # (C_out, H, W)
        unet_pred_np = unet_pred[0].cpu().numpy()                # (C_out, H, W)
        input_np = input_sample[0, :unet_pred_np.shape[0]].cpu().numpy()  # (C_out, H, W)
        target_np = test_targets[idx][:unet_pred_np.shape[0]].cpu().numpy()  # (C_out, H, W)

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

        print("input_denorm", np.nanmin(input_denorm), np.nanmax(input_denorm))
        print("unet_pred_denorm", np.nanmin(unet_pred_denorm), np.nanmax(unet_pred_denorm))
        print("ddim_pred_denorm", np.nanmin(ldm_pred_denorm), np.nanmax(ldm_pred_denorm))
        print("target_denorm", np.nanmin(target_denorm), np.nanmax(target_denorm))

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
            axes[2, j].set_title(f"DDIM stochastic generation (eta=0.0)Output (denorm) {channel_names[j]}")
            axes[2, j].axis('off')
            axes[3, j].imshow(np.flipud(target_denorm[j]), cmap='coolwarm', vmin=vmins[j], vmax=vmaxs[j])
            axes[3, j].set_title(f"Ground Truth (denorm) {channel_names[j]}")
            axes[3, j].axis('off')
            cbar = fig.colorbar(axes[0, j].images[0], ax=axes[:, j], fraction=0.02, pad=0.01)
            cbar.ax.set_ylabel(channel_names[j])

        fig.savefig(f"DDIM_conditional_derived/outputs/debug_output_{idx}.png")
        print(f"Plot saved as debug_output_{idx}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=26, help="index to plot")
    args = parser.parse_args()
    main(args.idx)

    #default index 26,, can be changed from 1. to batch size,,, from the slurm script