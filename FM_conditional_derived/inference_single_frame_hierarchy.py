import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from tqdm import tqdm

sys.path.append("..")
sys.path.append("../..")

from models.unet_module import DownscalingUnetLightning
from DownscalingDataModule import DownscalingDataModule
from models.components.diff.denoiser.unet import UNetModel
from models.components.diff.conditioner import AFNOConditionerNetCascade
from models.diff_module import FMContextual


# ------------------------------------------------------------------------------------#

base_seed = 124
DEFAULT_NUM_STEPS = [10]
DEFAULT_NUM_SAMPLES = 1


# ------------------------------------------------------------------------------------#

def denorm_pr(x, pr_params):
    arr = x * pr_params["std"] + pr_params["mean"]
    if np.isnan(arr).any():
        print("NaNs before exp in denorm_pr!")
    arr = np.exp(arr) - pr_params["epsilon"]
    if np.isnan(arr).any():
        print("NaNs after exp in denorm_pr!")
    return arr


def denorm_temp(x, params):
    arr = x * params["std"] + params["mean"]
    if np.isnan(arr).any():
        print("NaNs in denorm_temp!")
    return arr


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_precip_mask(mask_full, target_shape):
    arr_h, arr_w = target_shape
    mask_h, mask_w = mask_full.shape

    if mask_h != arr_h or mask_w != arr_w:
        start_h = (mask_h - arr_h) // 2
        start_w = (mask_w - arr_w) // 2
        return mask_full[start_h:start_h + arr_h, start_w:start_w + arr_w]

    return mask_full


def get_test_sample(test_loader, idx, device):
    running_idx = 0

    for batch_inputs, batch_targets in test_loader:
        batch_size = batch_inputs.shape[0]

        if running_idx <= idx < running_idx + batch_size:
            local_idx = idx - running_idx
            input_sample = batch_inputs[local_idx:local_idx + 1].to(device)
            target_sample = batch_targets[local_idx:local_idx + 1].to(device)
            return input_sample, target_sample

        running_idx += batch_size

    raise IndexError(f"idx={idx} is out of range for the test set")


# ------------------------------------------------------------------------------------#

with xr.open_dataset("Dataset_Setup_I_Chronological_12km/RhiresD_target_train_scaled.nc") as ds:
    precip_mask_full = ~np.isnan(ds["RhiresD"].values[0])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------------------#

def main(idx, num_steps=None, num_samples=DEFAULT_NUM_SAMPLES):
    if num_steps is None:
        num_steps = DEFAULT_NUM_STEPS.copy()
    elif isinstance(num_steps, int):
        num_steps = [num_steps]

    with open("Dataset_Setup_I_Chronological_12km/RhiresD_scaling_params.json", "r") as f:
        pr_params = json.load(f)
    with open("Dataset_Setup_I_Chronological_12km/TabsD_scaling_params.json", "r") as f:
        temp_params = json.load(f)

    train_input_paths = {
        "precip": "Dataset_Setup_I_Chronological_12km/RhiresD_input_train_scaled.nc",
        "temp": "Dataset_Setup_I_Chronological_12km/TabsD_input_train_scaled.nc",
    }
    train_target_paths = {
        "precip": "Dataset_Setup_I_Chronological_12km/RhiresD_target_train_scaled.nc",
        "temp": "Dataset_Setup_I_Chronological_12km/TabsD_target_train_scaled.nc",
    }
    val_input_paths = {
        "precip": "Dataset_Setup_I_Chronological_12km/RhiresD_input_val_scaled.nc",
        "temp": "Dataset_Setup_I_Chronological_12km/TabsD_input_val_scaled.nc",
    }
    val_target_paths = {
        "precip": "Dataset_Setup_I_Chronological_12km/RhiresD_target_val_scaled.nc",
        "temp": "Dataset_Setup_I_Chronological_12km/TabsD_target_val_scaled.nc",
    }
    test_input_paths = {
        "precip": "Dataset_Setup_I_Chronological_12km/RhiresD_input_test_scaled.nc",
        "temp": "Dataset_Setup_I_Chronological_12km/TabsD_input_test_scaled.nc",
    }
    test_target_paths = {
        "precip": "Dataset_Setup_I_Chronological_12km/RhiresD_target_test_scaled.nc",
        "temp": "Dataset_Setup_I_Chronological_12km/TabsD_target_test_scaled.nc",
    }
    elevation_path = "elevation.tif"

    # ------------------------------------------------------------------------------------#

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
            "variables": {
                "input": {"precip": "RhiresD", "temp": "TabsD"},
                "target": {"precip": "RhiresD", "temp": "TabsD"},
            },
            "preprocessing": {"nan_to_num": True, "nan_value": 0.0},
        },
    )

    dm.setup()
    test_loader = dm.test_dataloader()
    input_sample, target_sample = get_test_sample(test_loader, idx, device)

    # ------------------------------------------------------------------------------------#

    unet_regr = DownscalingUnetLightning(
        in_ch=3,
        out_ch=2,
        features=[64, 128, 256, 512],
        channel_names=["precip", "temp"],
        precip_scaling_json="Dataset_Setup_I_Chronological_12km/RhiresD_scaling_params.json",
    )
    unet_regr_ckpt = torch.load(
        "LDM_conditional/trained_ckpts_optimised/12km/UNet_ckpts/LDM_conditional.models.unet_module.DownscalingUnetLightning_12km_logtransform_lr0.001_precip_loss_weight1.0_1.0_crps[]_factor0.5_pat3.ckpt.ckpt",
        map_location="cpu",
        weights_only=False,
    )["state_dict"]
    unet_regr.load_state_dict(unet_regr_ckpt, strict=False)
    unet_regr = unet_regr.to(device)
    unet_regr.eval()

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
        num_heads=2,
    )
    conditioner = AFNOConditionerNetCascade(
        autoencoder=None,
        input_channels=[2],
        embed_dim=[32, 64, 128],
        analysis_depth=3,
        cascade_depth=3,
        context_ch=[32, 64, 128],
    )
    fm_model = FMContextual(
        denoiser=denoiser,
        context_encoder=conditioner,
        loss_type="l2",
        use_ema=True,
        ema_decay=0.9999,
        lr=1e-4,
    )
    fm_ckpt = torch.load(
        "FM_conditional_derived/trained_ckpts/12km/FM_L2=0.ckpt",
        map_location=device,
    )
    fm_model.load_state_dict(fm_ckpt["state_dict"], strict=False)
    fm_model = fm_model.to(device)
    fm_model.eval()

    if fm_model.context_encoder is not None:
        fm_model.context_encoder.eval()

    # ------------------------------------------------------------------------------------#

    with torch.no_grad():
        unet_pred = unet_regr(input_sample)

    maes = {step: {} for step in num_steps}
    maes["Coarse"] = {}
    maes["UNet"] = {}

    channel_names = ["Precip", "Temp"]
    params_list = [pr_params, temp_params]

    target_np = target_sample[0].detach().cpu().numpy()
    target_denorm = np.empty_like(target_np)

    for i, params in enumerate(params_list):
        if i == 0:
            target_denorm[i] = denorm_pr(target_np[i], pr_params)
        else:
            target_denorm[i] = denorm_temp(target_np[i], params)

    coarse_input_np = input_sample[0, :2].detach().cpu().numpy()
    coarse_denorm = np.empty_like(coarse_input_np)

    for i, params in enumerate(params_list):
        if i == 0:
            coarse_denorm[i] = denorm_pr(coarse_input_np[i], pr_params)
        else:
            coarse_denorm[i] = denorm_temp(coarse_input_np[i], params)

    precip_mask = get_precip_mask(precip_mask_full, target_denorm[0].shape)

    for i, name in enumerate(channel_names):
        mask = precip_mask if name.lower().startswith("precip") else np.ones_like(target_denorm[i], dtype=bool)
        coarse_mae = np.nanmean(np.abs(coarse_denorm[i][mask] - target_denorm[i][mask]))
        maes["Coarse"][name] = coarse_mae

    unet_pred_np = unet_pred[0].detach().cpu().numpy()
    unet_pred_denorm = np.empty_like(unet_pred_np)

    for i, params in enumerate(params_list):
        if i == 0:
            unet_pred_denorm[i] = denorm_pr(unet_pred_np[i], pr_params)
        else:
            unet_pred_denorm[i] = denorm_temp(unet_pred_np[i], params)

    for i, name in enumerate(channel_names):
        mask = precip_mask if name.lower().startswith("precip") else np.ones_like(target_denorm[i], dtype=bool)
        unet_mae = np.nanmean(np.abs(unet_pred_denorm[i][mask] - target_denorm[i][mask]))
        maes["UNet"][name] = unet_mae

    fm_pred_denorm_mean = None

    for step in num_steps:
        fm_pred_denorm_list = []

        for j in range(num_samples):
            set_seed(base_seed + j)

            with torch.no_grad():
                unet_pred = unet_regr(input_sample)
                fm_pred = fm_model.sample(
                    x=input_sample,
                    num_steps=step,
                    use_ema=True,
                    coarse_pred=unet_pred,
                )

            final_pred_np = fm_pred[0].detach().cpu().numpy()
            fm_pred_denorm = np.empty_like(final_pred_np)

            for i, params in enumerate(params_list):
                if i == 0:
                    fm_pred_denorm[i] = denorm_pr(final_pred_np[i], pr_params)
                else:
                    fm_pred_denorm[i] = denorm_temp(final_pred_np[i], params)

            fm_pred_denorm_list.append(fm_pred_denorm)

        fm_pred_denorm_mean = np.mean(fm_pred_denorm_list, axis=0)

        for i, name in enumerate(channel_names):
            mask = precip_mask if name.lower().startswith("precip") else np.ones_like(target_denorm[i], dtype=bool)
            mae = np.nanmean(np.abs(fm_pred_denorm_mean[i][mask] - target_denorm[i][mask]))
            maes[step][name] = mae

    print(f"\nMAE Table for Frame idx = {idx}")
    print(f"{'Steps':>8} | {'Precip':>10} | {'Temp':>10}")
    print("-" * 34)
    print(f"{'Coarse':>8} | {maes['Coarse']['Precip']:10.4f} | {maes['Coarse']['Temp']:10.4f}")
    print(f"{'UNet':>8} | {maes['UNet']['Precip']:10.4f} | {maes['UNet']['Temp']:10.4f}")

    for step in num_steps:
        print(f"{str(step):>8} | {maes[step]['Precip']:10.4f} | {maes[step]['Temp']:10.4f}")

    vmins = [
        target_denorm[j][precip_mask].min() if channel_names[j].lower().startswith("precip") else target_denorm[j].min()
        for j in range(len(params_list))
    ]
    vmaxs = [
        target_denorm[j][precip_mask].max() if channel_names[j].lower().startswith("precip") else target_denorm[j].max()
        for j in range(len(params_list))
    ]

    fig, axes = plt.subplots(4, len(params_list), figsize=(5 * len(params_list), 12), dpi=300)
    if len(params_list) == 1:
        axes = axes[:, np.newaxis]

    for j in range(len(params_list)):
        arrs = [coarse_denorm[j], unet_pred_denorm[j], fm_pred_denorm_mean[j], target_denorm[j]]
        titles = ["Coarse Input", "UNet Output", "FM Output", "Ground Truth"]

        for i, arr in enumerate(arrs):
            arr_to_plot = arr.copy()

            if channel_names[j].lower().startswith("precip"):
                arr_to_plot[~precip_mask] = np.nan

            axes[i, j].imshow(np.flipud(arr_to_plot), cmap="coolwarm", vmin=vmins[j], vmax=vmaxs[j])
            axes[i, j].set_title(f"{titles[i]} (denorm) {channel_names[j]}")
            axes[i, j].axis("off")

        cbar = fig.colorbar(axes[0, j].images[0], ax=axes[:, j], fraction=0.02, pad=0.01)
        cbar.ax.set_ylabel(channel_names[j])

    os.makedirs("FM_conditional_derived/outputs_inference", exist_ok=True)
    save_path = f"FM_conditional_derived/outputs_inference/debug_output_{idx}_FM_model.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved as {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=None, help="index to plot (if None, run all)")
    parser.add_argument(
        "--num_steps",
        type=int,
        nargs="+",
        default=DEFAULT_NUM_STEPS,
        help="List of FM sampling steps",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="Number of FM samples to average",
    )
    args = parser.parse_args()

    if args.idx is not None:
        main(args.idx, args.num_steps, args.num_samples)
    else:
        with xr.open_dataset("Dataset_Setup_I_Chronological_12km/RhiresD_input_test_scaled.nc") as ds:
            n_frames = ds.sizes["time"]

        print(f"Total frames to downscale: {n_frames}")
        for idx in tqdm(range(n_frames), desc="Downscaling frames"):
            main(idx, args.num_steps, args.num_samples)