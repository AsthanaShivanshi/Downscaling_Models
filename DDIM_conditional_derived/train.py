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
            in_ch=cfg.model.denoiser.get("in_channels", 2),
            out_ch=cfg.model.denoiser.get("out_channels", 2),
            features=cfg.model.denoiser.get("features", [64,128,256,512])
        )
        checkpoint = torch.load(cfg.model.unet_regr.checkpoint, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("unet."):
                new_state_dict[k[len("unet."):]] = v
            else:
                new_state_dict[k] = v
        unet_model.load_state_dict(new_state_dict)
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

    wandb.finish() 

    run_name = f"run_{os.environ.get('HYDRA_JOB_NUM', '0')}_{os.getpid()}"
    logger = WandbLogger(project="DDIM_residual_run_12km_bivariate", log_model=True, name=run_name)
    logger.experiment.config.update({
        "timesteps": cfg.model.get("timesteps"),
        "beta_schedule": cfg.model.get("beta_schedule"),
        "learning_rate": cfg.model.get("lr"),
        "batch_size": cfg.experiment.get("batch_size"),
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
        max_epochs=200,
        accelerator=cfg.get("trainer", {}).get("accelerator", "cuda"),
        devices=cfg.get("trainer", {}).get("devices", 1),
    )

    # Train
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    # Test best checkpoint
    ckpt_path = trainer.checkpoint_callback.best_model_path if hasattr(trainer, "checkpoint_callback") else None
    trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

@hydra.main(version_base="1.3", config_path="configs", config_name="DDIM_bivariate_config.yaml")
def main(cfg: DictConfig):
    train(cfg)

if __name__ == "__main__":
    main()