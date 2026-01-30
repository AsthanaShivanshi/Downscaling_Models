import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
import os
import hydra
import torch
from omegaconf import DictConfig
from lightning import Trainer, LightningModule, LightningDataModule, Callback
from lightning.pytorch.loggers import WandbLogger
from models.components.unet import DownscalingUnet
import wandb

def train(cfg: DictConfig):
    # Set seed if specified
    if cfg.get("seed"):
        torch.manual_seed(cfg.seed)

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)


    unet_model = None
    if cfg.model.get("unet_regr"):
        unet_model = DownscalingUnet(
            in_ch=cfg.model.unet_regr.get("in_ch", 3),
            out_ch=cfg.model.unet_regr.get("out_ch", 2),
            features=cfg.model.unet_regr.get("features", [64,128,256,512])
        )
        checkpoint = torch.load(cfg.model.unet_regr.checkpoint, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("unet."):
                new_state_dict[k[len("unet."):]] = v
            else:
                new_state_dict[k] = v
        unet_model.load_state_dict(new_state_dict, strict=False)
        print("Loaded UNet mean regression model from:", cfg.model.unet_regr.checkpoint)

    # Instantiate DDIM 
    model: LightningModule = hydra.utils.instantiate(cfg.model, unet_regr=unet_model)

    # Debugging prints for val 
    print("DataModule val files", datamodule.val_input, datamodule.val_target)
    if isinstance(datamodule.val_input, dict):
        for k, v in datamodule.val_input.items():
            if not os.path.exists(v):
                raise FileNotFoundError(f"Val input file for {k} not found: {v}")
    if isinstance(datamodule.val_target, dict):
        for k, v in datamodule.val_target.items():
            if not os.path.exists(v):
                raise FileNotFoundError(f"Val target file for {k} not found: {v}")

 
    run_name = (
    f"DDIM_{cfg.model.get('parameterization', 'none')}_"
    f"{cfg.model.get('beta_schedule', 'none')}_"
    f"bs{cfg.experiment.get('batch_size', 'NA')}_"
    f"seed{cfg.get('seed', 'NA')}_"
    f"{os.environ.get('HYDRA_JOB_NUM', '0')}_{os.getpid()}"
)

    logger = WandbLogger(project="DDIM_residual_run_12km_bivariate", log_model=True, name=run_name)
    logger.experiment.config.update({
    # Model
    "timesteps": cfg.model.get("timesteps"),
    "beta_schedule": cfg.model.get("beta_schedule"),
    "linear_start": cfg.model.get("linear_start"),
    "linear_end": cfg.model.get("linear_end"),
    "learning_rate": cfg.model.get("lr"),
    "loss_type": cfg.model.get("loss_type"),
    "parameterization": cfg.model.get("parameterization"),
    "use_ema": cfg.model.get("use_ema"),
    "ema_decay": cfg.model.get("ema_decay"),
    # Denoiser
    "denoiser_in_channels": cfg.model.denoiser.get("in_channels"),
    "denoiser_out_channels": cfg.model.denoiser.get("out_channels"),
    "denoiser_model_channels": cfg.model.denoiser.get("model_channels"),
    "denoiser_num_res_blocks": cfg.model.denoiser.get("num_res_blocks"),
    "denoiser_attention_resolutions": cfg.model.denoiser.get("attention_resolutions"),
    "denoiser_context_ch": cfg.model.denoiser.get("context_ch"),
    "denoiser_channel_mult": cfg.model.denoiser.get("channel_mult"),
    "denoiser_num_heads": cfg.model.denoiser.get("num_heads"),
    "denoiser_dims": cfg.model.denoiser.get("dims"),
    "denoiser_use_fp16": cfg.model.denoiser.get("use_fp16"),
    # UNet regression
    "unet_regr_ckpt": cfg.model.unet_regr.get("checkpoint") if cfg.model.get("unet_regr") else None,
    # Sampler
    "sampler_schedule": cfg.model.sampler_cfg.get("schedule"),
    "sampler_device": cfg.model.sampler_cfg.get("device"),
    "sampler_ddim_num_steps": cfg.model.sampler_cfg.get("ddim_num_steps"),
    "sampler_ddim_eta": cfg.model.sampler_cfg.get("ddim_eta"),
    
    # Experiment
    "batch_size": cfg.experiment.get("batch_size"),
    "num_workers": cfg.experiment.get("num_workers"),
    
    # Callbacks
    "early_stopping_patience": cfg.callbacks.early_stopping.get("patience"),
    "early_stopping_monitor": cfg.callbacks.early_stopping.get("monitor"),
    "model_checkpoint_monitor": cfg.callbacks.model_checkpoint.get("monitor"),
    "model_checkpoint_dir": cfg.callbacks.model_checkpoint.get("dirpath"),
    "model_checkpoint_filename": cfg.callbacks.model_checkpoint.get("filename"),
    # Trainer
    "trainer_accelerator": cfg.get("trainer", {}).get("accelerator", "cuda"),
    "trainer_devices": cfg.get("trainer", {}).get("devices", 1),
    # Seed
    "seed": cfg.get("seed"),
})
    print("Wandb run name:", run_name)
    print("Wandb run id:", logger.experiment.id)

    # ckpt callback from config
    callbacks = []
    if cfg.get("callbacks"):
        for cb_cfg in cfg.callbacks.values():
            callbacks.append(hydra.utils.instantiate(cb_cfg))

    trainer = Trainer(
        callbacks=callbacks,
        logger=logger,
        max_epochs=100,
        accelerator=cfg.get("trainer", {}).get("accelerator", "cuda"),
        devices=cfg.get("trainer", {}).get("devices", 1),
    )

    # Train
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    # Test best checkpoint
    ckpt_path = trainer.checkpoint_callback.best_model_path if hasattr(trainer, "checkpoint_callback") else None
    trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


    ckpt_path = None
    if hasattr(trainer, "checkpoint_callback"):
        ckpt_path = trainer.checkpoint_callback.best_model_path
        print("Best checkpoint path:", ckpt_path)
        if logger is not None:
            logger.experiment.summary["best_checkpoint_path"] = ckpt_path

    wandb.finish() 
@hydra.main(version_base="1.3", config_path="configs", config_name="DDIM_bivariate_config.yaml")
def main(cfg: DictConfig):
    train(cfg)

if __name__ == "__main__":
    main()