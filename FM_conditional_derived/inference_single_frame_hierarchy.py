import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import json
import sys
from tqdm import tqdm
import xarray as xr
import os

sys.path.append("..")
sys.path.append("../..")

from models.unet_module import DownscalingUnetLightning
from DownscalingDataModule import DownscalingDataModule
from models.components.diff.denoiser.unet import UNetModel
from models.components.diff.denoiser.sample import flow_matching_sample
from models.components.diff.conditioner import AFNOConditionerNetCascade
from models.diff_module import DDIMResidualContextual

base_seed = 124
num_steps= 10
num_samples = 1


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

def main(idx, num_steps=num_steps, num_samples=num_samples):
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
        "LDM_conditional/trained_ckpts_optimised/12km/UNet_ckpts/LDM_conditional.models.unet_module.DownscalingUnetLightning_12km_logtransform_lr0.001_precip_loss_weight1.0_1.0_crps[]_factor0.5_pat3.ckpt.ckpt",
        map_location="cpu", weights_only=False
    )["state_dict"]
    unet_regr.load_state_dict(unet_regr_ckpt, strict=False)
    unet_regr = unet_regr.to(device)
    unet_regr.eval()

    # FM Model
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
    fm_model = DDIMResidualContextual(
        denoiser=denoiser,
        context_encoder=conditioner,
        loss_type="l2",
        use_ema=True,
        ema_decay=0.9999,
        lr=1e-4
    )
    fm_ckpt = torch.load(
        f"FM_conditional_derived/trained_ckpts/12km/FM_L2=0.ckpt",
        map_location=device
    )


    fm_model.load_state_dict(fm_ckpt["state_dict"], strict=False)
    fm_model = fm_model.to(device)
    fm_model.eval()
    if fm_model.context_encoder is not None:
        fm_model.context_encoder.eval()

    maes = {step: {} for step in num_steps}
    maes['Coarse'] = {}
    channel_names = ["Precip", "Temp"]
    params_list = [pr_params, temp_params]

    with torch.no_grad():


        input_sample = test_inputs[idx].unsqueeze(0).to(device)
        unet_pred = unet_regr(input_sample)
        context = [(unet_pred, None)]

        sample_shape = unet_pred.shape[1:]
        target_np = test_targets[idx][:unet_pred.shape[1]].cpu().numpy()



  
        target_denorm = np.empty_like(target_np)
        for i, params in enumerate(params_list):
            target_denorm[i] = denorm_pr(target_np[i], pr_params) if i == 0 else denorm_temp(target_np[i], params)

        coarse_input_np = input_sample[0, :2].cpu().numpy() #precip, temp, that order . 
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

        for i, name in enumerate(channel_names):
            mask = precip_mask if name.lower().startswith("precip") else np.ones_like(target_denorm[i], dtype=bool)
            coarse_mae = np.nanmean(np.abs(coarse_denorm[i][mask] - target_denorm[i][mask]))
            maes['Coarse'][name] = coarse_mae

        unet_pred_np = unet_pred[0].cpu().numpy()
        unet_pred_denorm = np.empty_like(unet_pred_np)
        for i, params in enumerate(params_list):
            unet_pred_denorm[i] = denorm_pr(unet_pred_np[i], pr_params) if i == 0 else denorm_temp(unet_pred_np[i], params)
        for i, name in enumerate(channel_names):
            mask = precip_mask if name.lower().startswith("precip") else np.ones_like(target_denorm[i], dtype=bool)
            unet_mae = np.nanmean(np.abs(unet_pred_denorm[i][mask] - target_denorm[i][mask]))
            maes['UNet'] = maes.get('UNet', {})
            maes['UNet'][name] = unet_mae

        for step in num_steps:
            fm_pred_denorm_list = []
            for j in range(num_samples):
                torch.manual_seed(base_seed + j)


                np.random.seed(base_seed + j)
                z = torch.randn((1, *sample_shape), device=device)
                residual = flow_matching_sample(
                    fm_model,
                    conditioning=context,
                    shape=(1, *sample_shape),
                    steps=step,
                    device=device,
                    x_T=z,
                    integration="euler",
                    verbose=False
                )


                final_pred = unet_pred + residual
                final_pred_np = final_pred[0].cpu().numpy()



                fm_pred_denorm = np.empty_like(final_pred_np)
                for i, params in enumerate(params_list):
                    fm_pred_denorm[i] = denorm_pr(final_pred_np[i], pr_params) if i == 0 else denorm_temp(final_pred_np[i], params)
                fm_pred_denorm_list.append(fm_pred_denorm)
            fm_pred_denorm_mean = np.mean(fm_pred_denorm_list, axis=0)
            for i, name in enumerate(channel_names):
                mask = precip_mask if name.lower().startswith("precip") else np.ones_like(target_denorm[i], dtype=bool)
                mae = np.nanmean(np.abs(fm_pred_denorm_mean[i][mask] - target_denorm[i][mask]))
                maes[step][name] = mae

        print("\nMAE Table for Frame idx =", idx)
        print(f"{'Steps':>8} | {'Precip':>10} | {'Temp':>10}")
        print("-" * 34)
        print(f"{'Coarse':>8} | {maes['Coarse']['Precip']:10.4f} | {maes['Coarse']['Temp']:10.4f}")
        print(f"{'UNet':>8} | {maes['UNet']['Precip']:10.4f} | {maes['UNet']['Temp']:10.4f}")

        
        for idx_step, step in enumerate(num_steps):
            row_label = "FM Output" if idx_step == 0 else str(step)
            print(f"{row_label:>8} | {maes[step]['Precip']:10.4f} | {maes[step]['Temp']:10.4f}")


        vmins = [target_denorm[j][precip_mask].min() if channel_names[j].lower().startswith("precip")
                else target_denorm[j].min() for j in range(len(params_list))]
        vmaxs = [target_denorm[j][precip_mask].max() if channel_names[j].lower().startswith("precip")
                else target_denorm[j].max() for j in range(len(params_list))]
        fig, axes = plt.subplots(4, len(params_list), figsize=(5*len(params_list), 12), dpi=1000)
        if len(params_list) == 1:
            axes = axes[:, np.newaxis]
        for j in range(len(params_list)):
            arrs = [coarse_denorm[j], unet_pred_denorm[j], fm_pred_denorm_mean[j], target_denorm[j]]
            titles = ["Coarse Input", "UNet Output", "FM Output", "Ground Truth"]
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
        os.makedirs("FM_conditional_derived/outputs_inference", exist_ok=True)
        fig.savefig(f"FM_conditional_derived/outputs_inference/debug_output_{idx}_FM_model.png")
        print(f"Plot saved as FM_conditional_derived/outputs_inference/debug_output_{idx}_FM_model.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=None, help="index to plot (if None, run all)")
    parser.add_argument("--num_steps", type=int, nargs='+', default=num_steps, help="List of FM sampling steps")
    parser.add_argument("--num_samples", type=int, default=num_samples, help="Number of FM samples to average")
    args = parser.parse_args()
    if args.idx is not None:
        main(args.idx, args.num_steps, args.num_samples)
    else:
        with xr.open_dataset("Dataset_Setup_I_Chronological_12km/RhiresD_input_test_scaled.nc") as ds:
            N = ds.dims["time"]  # Total frames in test
        print(f"Total frames to downscale: {N}")
        for idx in tqdm(range(N), desc="Downscaling frames"):
            main(idx, args.num_steps, args.num_samples)