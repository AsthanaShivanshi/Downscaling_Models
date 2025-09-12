import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
import numpy as np
import hydra
import torch
from omegaconf import DictConfig
from lightning import Trainer, LightningModule, LightningDataModule, Callback
from lightning.pytorch.loggers import WandbLogger
from models.components.unet import DownscalingUnet

def train(cfg: DictConfig):
    # Set seed if specified
    if cfg.get("seed"):
        torch.manual_seed(cfg.seed)

    # Instantiate datamodule
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    # Load UNet regr model if needed
    if cfg.model.get("unet_regr"):
        unet_model = DownscalingUnet(
            in_ch=cfg.model.get("in_ch", 5),
            out_ch=cfg.model.get("out_ch", 4),
            features=cfg.model.get("features", [64,128,256,512])
        )
        checkpoint = torch.load(cfg.model.unet_regr, map_location="cpu")
        unet_model.load_state_dict(checkpoint["state_dict"])
        cfg.model.unet_regr = unet_model

    # Instantiate model
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # WandB logger
    logger = WandbLogger(project="downscaling", log_model=True)

    # ckpt callback from config
    callbacks = []
    if cfg.get("callbacks"):
        for cb_cfg in cfg.callbacks.values():
            callbacks.append(hydra.utils.instantiate(cb_cfg))

    # Trainer
    trainer = Trainer(
        callbacks=callbacks,
        logger=logger,
        max_epochs=cfg.get("trainer", {}).get("max_epochs", 100),
        accelerator=cfg.get("trainer", {}).get("accelerator", "gpu"),
        devices=cfg.get("trainer", {}).get("devices", 1),
    )

    # Train
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    # Testing best ckpt
    ckpt_path = trainer.checkpoint_callback.best_model_path if hasattr(trainer, "checkpoint_callback") else None
    trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

@hydra.main(version_base="1.3", config_path="configs", config_name="UNet_config.yaml")
def main(cfg: DictConfig):
    train(cfg)

if __name__ == "__main__":
    main()