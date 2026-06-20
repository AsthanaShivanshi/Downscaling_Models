import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import os
import hydra
import torch
from omegaconf import DictConfig
from lightning import Trainer, LightningModule, LightningDataModule
from lightning.pytorch.loggers import WandbLogger
from models.components.unet import DownscalingUnet
import wandb


def train(cfg: DictConfig):
    if cfg.get("seed"):
        torch.manual_seed(cfg.seed)

    torch.set_float32_matmul_precision("high")

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    unet_model = None
    if cfg.model.get("unet_regr"):
        unet_model = DownscalingUnet(
            in_ch=cfg.model.get("in_ch", 3),
            out_ch=cfg.model.get("out_ch", 2),
            features=cfg.model.get("features", [64, 128, 256, 512]),
        )

        checkpoint = torch.load(cfg.model.unet_regr, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("unet."):
                new_state_dict[k[len("unet."):]] = v

        unet_model.load_state_dict(new_state_dict)
        print("Loaded UNet mean regression model from:", cfg.model.unet_regr)

    model: LightningModule = hydra.utils.instantiate(cfg.model, unet_regr=unet_model)

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

    run_name = f"run_{cfg.get('hydra', {}).get('job', {}).get('id', os.getpid())}"
    logger = WandbLogger(project=cfg.wandb.project, log_model=True, name=run_name)

    wandb_config = {
        "experiment.batch_size": cfg.experiment.batch_size,
        "experiment.num_workers": cfg.experiment.num_workers,
        "variables.input": dict(cfg.variables.input),
        "variables.target": dict(cfg.variables.target),
        "preprocessing.nan_to_num": cfg.preprocessing.nan_to_num,
        "preprocessing.nan_value": cfg.preprocessing.nan_value,
        "unet_regr_ckpt": cfg.model.get("unet_regr"),
        "wandb_project": cfg.wandb.project,
        "model": dict(cfg.model),
    }

    if cfg.get("callbacks") and cfg.callbacks.get("model_checkpoint"):
        if cfg.callbacks.model_checkpoint.get("dirpath") is not None:
            wandb_config["model_checkpoint_dir"] = cfg.callbacks.model_checkpoint.dirpath
        if cfg.callbacks.model_checkpoint.get("filename") is not None:
            wandb_config["model_checkpoint_filename"] = cfg.callbacks.model_checkpoint.filename

    logger.experiment.config.update(wandb_config)

    print("Wandb run name:", run_name)
    print("Wandb run id:", logger.experiment.id)

    callbacks = []
    if cfg.get("callbacks"):
        for cb_cfg in cfg.callbacks.values():
            callbacks.append(hydra.utils.instantiate(cb_cfg))

    trainer_kwargs = {
        "callbacks": callbacks,
        "logger": logger,
        "max_epochs": 200,
        "accelerator": "cuda",
        "devices": 1,
    }

    if cfg.get("trainer"):
        trainer_kwargs.update(dict(cfg.trainer))

    trainer = Trainer(**trainer_kwargs)

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    ckpt_path = trainer.checkpoint_callback.best_model_path if hasattr(trainer, "checkpoint_callback") else None

    if ckpt_path:
        logger.experiment.config.update({"best_checkpoint": ckpt_path})
        logger.experiment.log({"best_checkpoint": ckpt_path})

    trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


@hydra.main(version_base="1.3", config_path="configs", config_name="UNet_bivariate_config_12km")
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()