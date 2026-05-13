import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import json
import sys
from tqdm import tqdm
import xarray as xr

sys.path.append("..")
sys.path.append("../..")

from models.unet_module import DownscalingUnetLightning
from DownscalingDataModule import DownscalingDataModule
from models.components.diff.denoiser.unet import UNetModel
from models.components.diff.denoiser.ddim import DDIMSampler
from models.components.diff.conditioner import AFNOConditionerNetCascade
from models.diff_module import DDIMResidualContextual

def denorm_pr(x, pr_params):
    arr = x * pr_params['std'] + pr_params['mean']
    if np.isnan(arr).any():
        print("NaNs before exp in denorm_pr!")
    arr = np.exp(arr) - pr_params['epsilon']
    if np.isnan(arr).any():
        print("NaNs after exp in denorm_pr!")
    return arr

def denorm_temp(x, params):
    arr = x * params['std'] + params['mean']
    if np.isnan(arr).any():
        print("NaNs in denorm_temp!")
    return arr

with xr.open_dataset("Dataset_Setup_I_Chronological_12km/RhiresD_target_train_scaled.nc") as ds:
    precip_mask_full = ~np.isnan(ds["RhiresD"].values[0])  # shape [H_full, W_full]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(idx, sampling_steps_list=[250, 500, 750, 999]):
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
        "LDM_conditional/trained_ckpts_optimised/12km/UNet_ckpts/LDM_conditional.models.unet_module.DownscalingUnetLightning_logtransform_lr0.01_precip_loss_weight5.0_1.0_crps[0, 1]_factor0.5_pat3.ckpt.ckpt",
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
        loss_type="l1",
        beta_schedule="cosine",
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        use_ema=True,
        ema_decay=0.9999,
        lr=1e-4
    )
    ddim_ckpt = torch.load(
        "DDIM_conditional_derived/trained_ckpts/12km/DDIM_checkpoint_L1_cosine_schedule_loss_parameterisation_v.ckpt",
        map_location=device
    )
    ddim.load_state_dict(ddim_ckpt["state_dict"], strict=False)
    ddim = ddim.to(device)
    ddim.eval()
    sampler = DDIMSampler(ddim, device=device)

    # Check for parameter corruption
    for name, param in unet_regr.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN detected in UNet parameter: {name}")
        if torch.isinf(param).any():
            print(f"Inf detected in UNet parameter: {name}")
    for name, param in ddim.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN detected in DDIM parameter: {name}")
        if torch.isinf(param).any():
            print(f"Inf detected in DDIM parameter: {name}")


    maes = {step: {} for step in sampling_steps_list}
    maes['Coarse'] = {}
    channel_names = ["Precip", "Temp"]
    params_list = [pr_params, temp_params]

    with torch.no_grad():
        input_sample = test_inputs[idx].unsqueeze(0).to(device)
        unet_pred = unet_regr(input_sample)
        context = [(unet_pred, None)]
        sample_shape = unet_pred.shape[1:]
        target_np = test_targets[idx][:unet_pred.shape[1]].cpu().numpy()

        # Denormalize target for MAE computation
        target_denorm = np.empty_like(target_np)
        for i, params in enumerate(params_list):
            target_denorm[i] = denorm_pr(target_np[i], pr_params) if i == 0 else denorm_temp(target_np[i], params)

        # Denormalize coarse input for plotting and MAE
        coarse_input_np = input_sample[0, :2].cpu().numpy()  # assuming first 2 channels are precip, temp
        coarse_denorm = np.empty_like(coarse_input_np)
        for i, params in enumerate(params_list):
            coarse_denorm[i] = denorm_pr(coarse_input_np[i], pr_params) if i == 0 else denorm_temp(coarse_input_np[i], params)

        arr_h, arr_w = target_denorm[0].shape
        mask_h, mask_w = precip_mask_full.shape
        if mask_h != arr_h or mask_w != arr_w:
            start_h = (mask_h - arr_h) // 2
            start_w = (mask_w - arr_w) // 2
            precip_mask = precip_mask_full[start_h:start_h+arr_h, start_w:start_w+arr_w]
        else:
            precip_mask = precip_mask_full

        # Coarse MAE
        for i, name in enumerate(channel_names):
            mask = precip_mask if name.lower().startswith("precip") else np.ones_like(target_denorm[i], dtype=bool)
            coarse_mae = np.nanmean(np.abs(coarse_denorm[i][mask] - target_denorm[i][mask]))
            maes['Coarse'][name] = coarse_mae

        # UNet MAE (deterministic)
        unet_pred_np = unet_pred[0].cpu().numpy()
        unet_pred_denorm = np.empty_like(unet_pred_np)
        for i, params in enumerate(params_list):
            unet_pred_denorm[i] = denorm_pr(unet_pred_np[i], pr_params) if i == 0 else denorm_temp(unet_pred_np[i], params)
        for i, name in enumerate(channel_names):
            mask = precip_mask if name.lower().startswith("precip") else np.ones_like(target_denorm[i], dtype=bool)
            unet_mae = np.nanmean(np.abs(unet_pred_denorm[i][mask] - target_denorm[i][mask]))
            maes['UNet'] = maes.get('UNet', {})
            maes['UNet'][name] = unet_mae

        # DDIM MAE for each sampling step
        for step in sampling_steps_list:
            z = torch.randn((1, *sample_shape), device=device)
            residual, _ = sampler.sample(
                S=step,
                batch_size=1,
                shape=sample_shape,
                conditioning=context,
                eta=0.0,
                verbose=False,
                x_T=z,
                schedule="cosine"
            )
            final_pred = unet_pred + residual
            final_pred_np = final_pred[0].cpu().numpy()
            ddim_pred_denorm = np.empty_like(final_pred_np)
            for i, params in enumerate(params_list):
                ddim_pred_denorm[i] = denorm_pr(final_pred_np[i], pr_params) if i == 0 else denorm_temp(final_pred_np[i], params)
            for i, name in enumerate(channel_names):
                mask = precip_mask if name.lower().startswith("precip") else np.ones_like(target_denorm[i], dtype=bool)
                mae = np.nanmean(np.abs(ddim_pred_denorm[i][mask] - target_denorm[i][mask]))
                maes[step][name] = mae

        # Print MAE table
        print("\nMAE Table for Frame idx =", idx)
        print(f"{'Steps':>8} | {'Precip':>10} | {'Temp':>10}")
        print("-" * 34)
        print(f"{'Coarse':>8} | {maes['Coarse']['Precip']:10.4f} | {maes['Coarse']['Temp']:10.4f}")
        print(f"{'UNet':>8} | {maes['UNet']['Precip']:10.4f} | {maes['UNet']['Temp']:10.4f}")
        for step in sampling_steps_list:
            print(f"{step:8d} | {maes[step]['Precip']:10.4f} | {maes[step]['Temp']:10.4f}")

        # Plotting (now with coarse input in first row)
        vmins = [target_denorm[j][precip_mask].min() if channel_names[j].lower().startswith("precip")
                else target_denorm[j].min() for j in range(len(params_list))]
        vmaxs = [target_denorm[j][precip_mask].max() if channel_names[j].lower().startswith("precip")
                else target_denorm[j].max() for j in range(len(params_list))]
        fig, axes = plt.subplots(4, len(params_list), figsize=(5*len(params_list), 12), dpi=150)
        if len(params_list) == 1:
            axes = axes[:, np.newaxis]
        for j in range(len(params_list)):
            arrs = [coarse_denorm[j], unet_pred_denorm[j], ddim_pred_denorm[j], target_denorm[j]]
            titles = ["Coarse Input", "UNet Output", "DDIM Output", "Ground Truth"]
            for i, arr in enumerate(arrs):
                arr_to_plot = arr.copy()
                if channel_names[j].lower().startswith("precip"):
                    mask_h, mask_w = precip_mask.shape
                    arr_h, arr_w = arr_to_plot.shape
                    if mask_h != arr_h or mask_w != arr_w:
                        start_h = (mask_h - arr_h) // 2
                        start_w = (mask_w - arr_w) // 2
                        mask_cropped = precip_mask[start_h:start_h+arr_h, start_w:start_w+arr_w]
                    else:
                        mask_cropped = precip_mask
                    arr_to_plot[~mask_cropped] = np.nan
                axes[i, j].imshow(np.flipud(arr_to_plot), cmap='coolwarm', vmin=vmins[j], vmax=vmaxs[j])
                axes[i, j].set_title(f"{titles[i]} (denorm) {channel_names[j]}")
                axes[i, j].axis('off')
            cbar = fig.colorbar(axes[0, j].images[0], ax=axes[:, j], fraction=0.02, pad=0.01)
            cbar.ax.set_ylabel(channel_names[j])
        fig.savefig(f"DDIM_conditional_derived/outputs/debug_output_{idx}_model_l1_model_1.png")
        print(f"Plot saved as debug_output_{idx}_model_l1_model_1.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=None, help="index to plot (if None, run all)")
    parser.add_argument("--sampling_steps", type=int, nargs='+', default=[250, 500, 750, 999], help="List of DDIM sampling steps")
    args = parser.parse_args()
    if args.idx is not None:
        main(args.idx, args.sampling_steps)
    else:
        with xr.open_dataset("Dataset_Setup_I_Chronological_12km/RhiresD_input_test_scaled.nc") as ds:
            N = ds.dims["time"]  # Total frames in test
        print(f"Total frames to downscale: {N}")
        for idx in tqdm(range(N), desc="Downscaling frames"):
            main(idx, args.sampling_steps)