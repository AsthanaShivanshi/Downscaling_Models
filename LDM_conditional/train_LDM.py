from ast import For
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

    # Instantiate model: for UNet and VAE
    #model: LightningModule = hydra.utils.instantiate(cfg.model, unet_regr=unet_model if cfg.model.get("unet_regr") else None)

    #Checking that files exist /debugging prints cz val files were not being read: AsthanaSh
    #print("DataModule val files", datamodule.val_input, datamodule.val_target)
    #if isinstance(datamodule.val_input, dict):
    #    for k, v in datamodule.val_input.items():
    #        if not os.path.exists(v):
    #            raise FileNotFoundError(f"Val input file for {k} not found: {v}")
    #if isinstance(datamodule.val_target, dict):
    #    for k, v in datamodule.val_target.items():
    #        if not os.path.exists(v):
    #            raise FileNotFoundError(f"Val target file for {k} not found: {v}")
#For LDM, pass
    autoencoder_cfg = cfg.model.autoencoder
    autoencoder = hydra.utils.instantiate(autoencoder_cfg, unet_regr=unet_model if cfg.model.get("unet_regr") else None)
    context_encoder = None
    if cfg.model.get("context_encoder"):
       context_encoder = hydra.utils.instantiate(cfg.model.context_encoder, autoencoder=autoencoder)
    model = hydra.utils.instantiate(cfg.model, autoencoder=autoencoder, context_encoder=context_encoder, unet_regr=unet_model if cfg.model.get("unet_regr") else None)
    
    wandb.finish()

    run_name = f"run_{os.environ.get('HYDRA_JOB_NUM', '0')}_{os.getpid()}"
    logger = WandbLogger(project="LDM_optim_run_12km_bivariate", log_model=True, name=run_name)

    logger.experiment.config.update({
        "learning_rate": cfg.model.get("lr"),
        "noise_schedule": cfg.sampler.get("schedule"),
        "parameterization": cfg.model.get("parameterization"),
        "loss_type": cfg.model.get("loss_type"),
        "timesteps": cfg.model.get("timesteps"),
        "latent_dim": cfg.get("latent_dim"),
        "vae_kl_weight": cfg.model.get("kl_weight"),
        "vae_beta_anneal_steps": cfg.model.get("beta_anneal_steps"),
        "batch_size": cfg.experiment.get("batch_size"),
        "num_workers": cfg.experiment.get("num_workers"),
        "denoiser_channels": cfg.denoiser.get("model_channels"),
        "denoiser_num_heads": cfg.denoiser.get("num_heads"),
        "denoiser_attention_resolutions": cfg.denoiser.get("attention_resolutions"),
        "denoiser_channel_mult": cfg.denoiser.get("channel_mult"),
        "conditioner_context_ch": cfg.conditioner.get("context_ch"),
        "conditioner_embed_dim": cfg.conditioner.get("embed_dim"),
        "ae_ckpt": cfg.model.get("ae_load_state_file"),
        "unet_regr_ckpt": cfg.model.get("unet_regr"),
        "train_input_paths": cfg.data.train.get("input"),
        "train_target_paths": cfg.data.train.get("target"),
        "val_input_paths": cfg.data.val.get("input") if cfg.data.get("val") else None,
        "val_target_paths": cfg.data.val.get("target") if cfg.data.get("val") else None,
        "test_input_paths": cfg.data.test.get("input") if cfg.data.get("test") else None,
        "test_target_paths": cfg.data.test.get("target") if cfg.data.get("test") else None,
    })

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

    # Testing ckpt
    ckpt_path = trainer.checkpoint_callback.best_model_path if hasattr(trainer, "checkpoint_callback") else None
    trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

@hydra.main(version_base="1.3", config_path="configs", config_name="LDM_bivariate_config.yaml")
def main(cfg: DictConfig):
    train(cfg)

if __name__ == "__main__":
    main()