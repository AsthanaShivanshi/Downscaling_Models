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

    # Instantiate datamodule
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    # Load UNet regr model if needed,,,
    #AsthanaSh_: fixed state_dict names with "unet." prefix by removing it,,was giving an error while loading
    if cfg.model.get("unet_regr"):
        unet_model = DownscalingUnet(
            in_ch=cfg.model.get("in_ch", 3),
            out_ch=cfg.model.get("out_ch", 2),
            features=cfg.model.get("features", [64,128,256,512])
        )

        checkpoint = torch.load(cfg.model.unet_regr, map_location="cpu",weights_only=False)
        state_dict = checkpoint["state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("unet."):
                new_state_dict[k[len("unet."):]] = v
        unet_model.load_state_dict(new_state_dict)
        print("Loaded UNet mean regression model from:", cfg.model.unet_regr)

    # Instantiate model: for UNet and VAE,,,,for Unet, do NOT forget to pass the RhiresD params for forward pass changes (see source code): TO SELF!!!!!!!
    model: LightningModule = hydra.utils.instantiate(cfg.model, unet_regr=unet_model if cfg.model.get("unet_regr") else None)

    # val files were not being read: AsthanaSh
    print("DataModule val files", datamodule.val_input, datamodule.val_target)
    if isinstance(datamodule.val_input, dict):
        for k, v in datamodule.val_input.items():
            if not os.path.exists(v):
                raise FileNotFoundError(f"Val input file for {k} not found: {v}")
    if isinstance(datamodule.val_target, dict):
        for k, v in datamodule.val_target.items():
            if not os.path.exists(v):
                raise FileNotFoundError(f"Val target file for {k} not found: {v}")
#For LDM, pass
    #autoencoder_cfg = cfg.model.autoencoder
    #autoencoder = hydra.utils.instantiate(autoencoder_cfg, unet_regr=unet_model if cfg.model.get("unet_regr") else None)
    #context_encoder = None
    #if cfg.model.get("context_encoder"):
    #    context_encoder = hydra.utils.instantiate(cfg.model.context_encoder, autoencoder=autoencoder)
    #model = hydra.utils.instantiate(cfg.model, autoencoder=autoencoder, context_encoder=context_encoder, unet_regr=unet_model if cfg.model.get("unet_regr") else None)
    

    wandb.finish()  

    run_name = f"run_{cfg.get('hydra', {}).get('job', {}).get('id', os.getpid())}" #unique for each run,,, appends job id
    logger = WandbLogger(project=cfg.wandb.project, log_model=True, name=run_name)



    wandb_config = {
        "vae.latent_dim": cfg.vae.latent_dim,
        "vae.kl_weight": cfg.vae.kl_weight,
        "vae.ae_flag": cfg.vae.ae_flag,
        "vae.beta_anneal_steps": cfg.vae.beta_anneal_steps,
        "vae.encoder_levels": cfg.vae.encoder_levels,
        "vae.encoder_min_ch": cfg.vae.encoder_min_ch,
        "vae.encoder_ch_mult": cfg.vae.encoder_ch_mult,
        "vae.decoder_levels": cfg.vae.decoder_levels,
        "vae.decoder_min_ch": cfg.vae.decoder_min_ch,
        "vae.lr": cfg.vae.lr,
        "vae.batch_size": cfg.vae.batch_size,
        "vae.num_workers": cfg.vae.num_workers,
        "vae.scheduler_factor": cfg.vae.scheduler_factor,
        "vae.scheduler_patience": cfg.vae.scheduler_patience,
        "experiment.batch_size": cfg.experiment.batch_size,
        "experiment.num_workers": cfg.experiment.num_workers,
        "variables.input": dict(cfg.variables.input),
        "variables.target": dict(cfg.variables.target),
        "preprocessing.nan_to_num": cfg.preprocessing.nan_to_num,
        "preprocessing.nan_value": cfg.preprocessing.nan_value,
        "encoder": dict(cfg.encoder),
        "decoder": dict(cfg.decoder),
        "model_checkpoint_dir": cfg.callbacks.model_checkpoint.dirpath,
        "model_checkpoint_filename": cfg.callbacks.model_checkpoint.filename,
        "unet_regr_ckpt": cfg.model.unet_regr,
        "lr_scheduler": dict(cfg.lr_scheduler),
        "wandb_project": cfg.wandb.project,
    }
    logger.experiment.config.update(wandb_config)

    print("Wandb run name:", run_name)
    print("Wandb run id:", logger.experiment.id)

    # ckpt callback from config
    callbacks = []
    if cfg.get("callbacks"):
        for cb_cfg in cfg.callbacks.values():
            callbacks.append(hydra.utils.instantiate(cb_cfg))

    # Trainer : no min epochs, just early stopping after 10 epochs of no improvement in val loss
    trainer = Trainer(
        callbacks=callbacks,
        logger=logger,
        max_epochs=200,
        
        #accelerator="cpu",  # Forcing CPU for debugging on interactive partition : AsthanaSh
        #devices=1,

        #For submitting jobs to cluster, GPU , training, uncomment
        accelerator=cfg.get("trainer", {}).get("accelerator", "cuda"),
        devices=cfg.get("trainer", {}).get("devices", 1),
    )

    # Train
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    ckpt_path = trainer.checkpoint_callback.best_model_path if hasattr(trainer, "checkpoint_callback") else None

    if ckpt_path:
        logger.experiment.config.update({"best_checkpoint": ckpt_path})
        logger.experiment.log({"best_checkpoint": ckpt_path})

    trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

@hydra.main(version_base="1.3", config_path="configs", config_name="VAE_bivariate_config.yaml")
def main(cfg: DictConfig):
    train(cfg)

if __name__ == "__main__":
    main()