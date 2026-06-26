import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))
sys.path.append(str(SCRIPT_DIR.parent.parent))

from models.unet_module import DownscalingUnetLightning
from DownscalingDataModule import DownscalingDataModule
from models.components.diff.denoiser.unet import UNetModel
from models.components.diff.denoiser.ddim import DDIMSampler
from models.components.diff.conditioner import AFNOConditionerNetCascade
from models.diff_module import DDIMResidualContextual


DATA_DIR = "Dataset_Setup_I_Chronological_12km"
OUT_DIR = SCRIPT_DIR / "DDIM_conditional_derived" / "output_inference"

UNET_CKPT = SCRIPT_DIR / "LDM_conditional" / "trained_ckpts" / "12km" / "LDM_conditional.models.unet_module.DownscalingUnetLightning_bs32_lr0.001_delta1.0_factor0.5_pat3.ckpt.ckpt"
DDIM_CKPT = SCRIPT_DIR / "DDIM_conditional_derived" / "trained_ckpts" / "12km" / "DDIM_checkpoint_L1_quadratic_schedule_loss_parameterisation_v.ckpt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHANNEL_NAMES = ["Precip", "Temp"]


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def denorm_pr(x, pr_params):
    arr = x * pr_params["std"] + pr_params["mean"]
    arr = np.exp(arr) - pr_params["epsilon"]
    return arr


def denorm_temp(x, params):
    return x * params["std"] + params["mean"]


def denorm_channels(arr, pr_params, temp_params):
    out = np.empty_like(arr)
    out[0] = denorm_pr(arr[0], pr_params)
    out[1] = denorm_temp(arr[1], temp_params)
    return out


def crop_mask(mask, arr2d):
    mh, mw = mask.shape
    ah, aw = arr2d.shape
    if (mh, mw) == (ah, aw):
        return mask
    sh = (mh - ah) // 2
    sw = (mw - aw) // 2
    return mask[sh:sh + ah, sw:sw + aw]


def masked_mae(pred, target, mask):
    return np.nanmean(np.abs(pred[mask] - target[mask]))


def print_load_info(name, result):
    print(f"\n{name} load:")
    print(f"  missing keys   : {len(result.missing_keys)}")
    print(f"  unexpected keys: {len(result.unexpected_keys)}")
    if result.missing_keys:
        print(f"  first missing  : {result.missing_keys[:10]}")
    if result.unexpected_keys:
        print(f"  first unexpected: {result.unexpected_keys[:10]}")


def build_datamodule():
    dm = DownscalingDataModule(
        train_input={
            "precip": str(DATA_DIR / "RhiresD_input_train_scaled.nc"),
            "temp": str(DATA_DIR / "TabsD_input_train_scaled.nc"),
        },
        train_target={
            "precip": str(DATA_DIR / "RhiresD_target_train_scaled.nc"),
            "temp": str(DATA_DIR / "TabsD_target_train_scaled.nc"),
        },
        val_input={
            "precip": str(DATA_DIR / "RhiresD_input_val_scaled.nc"),
            "temp": str(DATA_DIR / "TabsD_input_val_scaled.nc"),
        },
        val_target={
            "precip": str(DATA_DIR / "RhiresD_target_val_scaled.nc"),
            "temp": str(DATA_DIR / "TabsD_target_val_scaled.nc"),
        },
        test_input={
            "precip": str(DATA_DIR / "RhiresD_input_test_scaled.nc"),
            "temp": str(DATA_DIR / "TabsD_input_test_scaled.nc"),
        },
        test_target={
            "precip": str(DATA_DIR / "RhiresD_target_test_scaled.nc"),
            "temp": str(DATA_DIR / "TabsD_target_test_scaled.nc"),
        },
        elevation=str(SCRIPT_DIR / "elevation.tif"),
        batch_size=1,
        num_workers=2,
        preprocessing={
            "variables": {
                "input": {"precip": "RhiresD", "temp": "TabsD"},
                "target": {"precip": "RhiresD", "temp": "TabsD"},
            },
            "preprocessing": {"nan_to_num": True, "nan_value": 0.0},
        },
    )
    dm.setup()
    return dm


def build_unet():
    model = DownscalingUnetLightning(
        in_ch=3,
        out_ch=2,
        features=[64, 128, 256, 512],
        channel_names=["precip", "temp"],
        precip_scaling_json=str(DATA_DIR / "RhiresD_scaling_params.json"),
    )
    ckpt = torch.load(UNET_CKPT, map_location="cpu", weights_only=False)
    result = model.load_state_dict(ckpt["state_dict"], strict=False)
    print_load_info("UNet", result)
    return model.to(DEVICE).eval()


def build_ddim():
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

    model = DDIMResidualContextual(
        denoiser=denoiser,
        context_encoder=conditioner,
        timesteps=1000,
        parameterization="v",
        loss_type="l1",
        beta_schedule="quadratic",
        linear_start=1e-4,
        linear_end=2e-2,
        use_ema=True,
        ema_decay=0.9999,
        lr=1e-4,
    )

    ckpt = torch.load(DDIM_CKPT, map_location="cpu", weights_only=False)
    result = model.load_state_dict(ckpt["state_dict"], strict=False)
    print_load_info("DDIM", result)

    model = model.to(DEVICE).eval()
    sampler = DDIMSampler(model, device=DEVICE)
    return model, sampler


def save_plot(sample_idx, step, coarse_denorm, unet_denorm, ddim_denorm, target_denorm, precip_mask):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 2, figsize=(10, 12), dpi=150)
    titles = ["Coarse Input", "UNet Output", f"DDIM Output S={step}", "Ground Truth"]

    for j, name in enumerate(CHANNEL_NAMES):
        vmin = target_denorm[j][precip_mask].min() if name.lower().startswith("precip") else target_denorm[j].min()
        vmax = target_denorm[j][precip_mask].max() if name.lower().startswith("precip") else target_denorm[j].max()

        arrays = [coarse_denorm[j], unet_denorm[j], ddim_denorm[j], target_denorm[j]]
        for i, arr in enumerate(arrays):
            arr_plot = arr.copy()
            if name.lower().startswith("precip"):
                arr_plot[~precip_mask] = np.nan
            axes[i, j].imshow(np.flipud(arr_plot), cmap="coolwarm", vmin=vmin, vmax=vmax)
            axes[i, j].set_title(f"{titles[i]} - {name}")
            axes[i, j].axis("off")

    out_file = OUT_DIR / f"debug_output_{sample_idx}_S{step}.png"
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved: {out_file}")


def run_sample(sample_idx, input_batch, target_batch, unet_regr, ddim, sampler, pr_params, temp_params, precip_mask_full, sampling_steps):
    input_sample = input_batch.to(DEVICE)
    target_np = target_batch[0, :2].cpu().numpy()

    target_denorm = denorm_channels(target_np, pr_params, temp_params)
    precip_mask = crop_mask(precip_mask_full, target_denorm[0])

    with torch.no_grad():
        unet_pred = unet_regr(input_sample)
        context = [(unet_pred, None)]
        sample_shape = tuple(unet_pred.shape[1:])

        coarse_np = input_sample[0, :2].cpu().numpy()
        coarse_denorm = denorm_channels(coarse_np, pr_params, temp_params)

        unet_np = unet_pred[0].cpu().numpy()
        unet_denorm = denorm_channels(unet_np, pr_params, temp_params)

        maes = {"Coarse": {}, "UNet": {}}

        for i, name in enumerate(CHANNEL_NAMES):
            mask = precip_mask if name.lower().startswith("precip") else np.ones_like(target_denorm[i], dtype=bool)
            maes["Coarse"][name] = masked_mae(coarse_denorm[i], target_denorm[i], mask)
            maes["UNet"][name] = masked_mae(unet_denorm[i], target_denorm[i], mask)

        fixed_z = torch.randn((1, *sample_shape), device=DEVICE)

        with ddim.ema_scope("inference"):
            for step in sampling_steps:
                residual, _ = sampler.sample(
                    S=step,
                    batch_size=1,
                    shape=sample_shape,
                    conditioning=context,
                    eta=0.0,
                    verbose=False,
                    x_T=fixed_z.clone(),
                    schedule="quadratic",
                )

                final_pred = unet_pred + residual
                final_np = final_pred[0].cpu().numpy()
                ddim_denorm = denorm_channels(final_np, pr_params, temp_params)

                maes[step] = {}
                for i, name in enumerate(CHANNEL_NAMES):
                    mask = precip_mask if name.lower().startswith("precip") else np.ones_like(target_denorm[i], dtype=bool)
                    maes[step][name] = masked_mae(ddim_denorm[i], target_denorm[i], mask)

                save_plot(sample_idx, step, coarse_denorm, unet_denorm, ddim_denorm, target_denorm, precip_mask)

    print(f"\nMAE Table for Frame idx = {sample_idx}")
    print(f"{'Steps':>8} | {'Precip':>10} | {'Temp':>10}")
    print("-" * 34)
    print(f"{'Coarse':>8} | {maes['Coarse']['Precip']:10.4f} | {maes['Coarse']['Temp']:10.4f}")
    print(f"{'UNet':>8} | {maes['UNet']['Precip']:10.4f} | {maes['UNet']['Temp']:10.4f}")
    for step in sampling_steps:
        print(f"{step:8d} | {maes[step]['Precip']:10.4f} | {maes[step]['Temp']:10.4f}")


def main(idx=None, sampling_steps=None):
    if sampling_steps is None:
        sampling_steps = [250, 500, 750, 999]

    pr_params = load_json(DATA_DIR / "RhiresD_scaling_params.json")
    temp_params = load_json(DATA_DIR / "TabsD_scaling_params.json")

    with xr.open_dataset(DATA_DIR / "RhiresD_target_train_scaled.nc") as ds:
        precip_mask_full = ~np.isnan(ds["RhiresD"].values[0])

    dm = build_datamodule()
    test_loader = dm.test_dataloader()

    unet_regr = build_unet()
    ddim, sampler = build_ddim()

    if idx is not None:
        for sample_idx, (input_batch, target_batch) in enumerate(test_loader):
            if sample_idx == idx:
                run_sample(sample_idx, input_batch, target_batch, unet_regr, ddim, sampler, pr_params, temp_params, precip_mask_full, sampling_steps)
                return
        raise IndexError(f"idx={idx} out of range")

    for sample_idx, (input_batch, target_batch) in enumerate(tqdm(test_loader, desc="Downscaling frames")):
        run_sample(sample_idx, input_batch, target_batch, unet_regr, ddim, sampler, pr_params, temp_params, precip_mask_full, sampling_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=None, help="single sample index")
    parser.add_argument("--sampling_steps", type=int, nargs="+", default=[250, 500, 750, 999])
    args = parser.parse_args()
    main(args.idx, args.sampling_steps)