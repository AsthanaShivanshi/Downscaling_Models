import hydra
from LDM_conditional.models.unet_module import DownscalingUnet
from LDM_conditional.models.ae_module import AutoencoderKL
from LDM_conditional.models.ldm_module import LatentDiffusion
from LDM_conditional.models.components.ldm.denoiser import DDIMSampler
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="configs", config_name="LDM_config")
def main(cfg: DictConfig):
    import torch
    import numpy as np
    import json
    import xarray as xr
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_samples = cfg.get("n_samples", 10)

    # Load normalization parameters
    with open("Dataset_Setup_I_Chronological_10km/RhiresD_scaling_params.json") as f:
        pr_params = json.load(f)
    with open("Dataset_Setup_I_Chronological_10km/TabsD_scaling_params.json") as f:
        temp_params = json.load(f)
    with open("Dataset_Setup_I_Chronological_10km/TminD_scaling_params.json") as f:
        temp_min_params = json.load(f)
    with open("Dataset_Setup_I_Chronological_10km/TmaxD_scaling_params.json") as f:
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

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    unet_ckpt = torch.load(cfg.model.unet_regr, map_location=device)
    vae_ckpt = torch.load(cfg.model.ae_load_state_file, map_location=device)
    ldm_ckpt_path = cfg.callbacks.model_checkpoint.dirpath + "/" + cfg.callbacks.model_checkpoint.filename + ".ckpt"
    ldm_ckpt = torch.load(ldm_ckpt_path, map_location=device)

    # Instantiate models
    unet_model = DownscalingUnet(
        in_ch=cfg.model.get("in_ch", 5),
        out_ch=cfg.model.get("out_ch", 4),
        features=cfg.model.get("features", [64,128,256,512])
    )
    state_dict = unet_ckpt["state_dict"]
    new_state_dict = {k[len("unet."):]: v for k, v in state_dict.items() if k.startswith("unet.")}
    unet_model.load_state_dict(new_state_dict)
    unet_model = unet_model.to(device)
    unet_model.eval()

    vae = AutoencoderKL(
        encoder=hydra.utils.instantiate(cfg.encoder),
        decoder=hydra.utils.instantiate(cfg.decoder)
    )
    vae.load_state_dict(vae_ckpt["state_dict"], strict=False)
    vae = vae.to(device)
    vae.eval()

    conditioner = hydra.utils.instantiate(cfg.conditioner, autoencoder=vae)
    conditioner = conditioner.to(device)
    conditioner.eval()
    
    ldm = LatentDiffusion(
        denoiser=hydra.utils.instantiate(cfg.denoiser),
        autoencoder=vae,
        unet_regr=unet_model,
        context_encoder=conditioner,
        parameterization=cfg.model.get("parameterization", "v"),
        loss_type=cfg.model.get("loss_type", "l2"),
        timesteps=cfg.model.get("timesteps", 1000)
    )
    ldm.load_state_dict(ldm_ckpt["state_dict"], strict=False)
    ldm = ldm.to(device)
    ldm.eval()

    sampler = DDIMSampler(
        model=ldm,
        schedule=cfg.sampler.get("schedule", "linear"),
        device=device,
        ddim_num_steps=cfg.sampler.get("ddim_num_steps", 250),
        ddim_eta=cfg.sampler.get("ddim_eta", 0.0)
    )

    all_samples = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Frames"):
            x_in, _ = batch
            x_in = x_in.to(device)
            batch_samples = []
            # UNet pred
            coarse_pred = unet_model(x_in)
            coarse_pred_encoded = vae.encode(coarse_pred)[0]
            context = conditioner([(coarse_pred_encoded, None)])
            # Latent 
            z_shape = (ldm.denoiser.in_channels, coarse_pred.shape[-2], coarse_pred.shape[-1])

            for s in range(n_samples):
                z_start = torch.randn((x_in.shape[0],) + z_shape, device=device)
                samples, _ = sampler.sample(
                    cfg.sampler.ddim_num_steps,
                    x_in.shape[0],
                    shape=z_shape,
                    conditioning=context,
                )

                decoded = vae.decode(samples)
                decoded_np = decoded.cpu().numpy()

                decoded_np = np.stack([denorm_sample(d) for d in decoded_np])
                batch_samples.append(decoded_np)
            # Stacking samples for this batch: [batch, n_samples, 4, H, W]
            batch_samples = np.stack(batch_samples, axis=1)
            all_samples.append(batch_samples)

    all_samples = np.concatenate(all_samples, axis=0)  # [num_frames, n_samples, 4, H, W]


    ref_ds = xr.open_dataset("Dataset_Setup_I_Chronological_10km/RhiresD_target_test_scaled.nc")
    times = ref_ds["time"].values
    lat = ref_ds["lat"].values
    lon = ref_ds["lon"].values
    ref_ds.close()

    var_names = ["precip", "temp", "temp_min", "temp_max"]

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

    ds.to_netcdf("ldm_conditional_testset_10_samples.nc", encoding={"ldm_samples": {"_FillValue": np.nan}})
    print("Saved NetCDF: ldm_conditional_testset_samples.nc")
    print("All samples shape (time, n_samples, 4, H, W):", all_samples.shape)

if __name__ == "__main__":
    main()