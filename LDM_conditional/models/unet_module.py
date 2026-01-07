from typing import Any

import torch
from lightning import LightningModule
import json
import os

"""LightningModule for the downscaling setup use case.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """



from .components.unet import DownscalingUnet


def crps_loss(pred, target):
    # MAE: CRPS det
    return torch.mean(torch.abs(pred - target))



class DownscalingUnetLightning(LightningModule):

    def __init__(
        self, 
        in_ch=1, 
        out_ch=1, 
        features=[64,128,256,512], 
        channel_names=None, 
        unet_regr=None, 
        precip_channel_idx=0, 
        lr=1e-3,
        huber_delta=1.0,
        precip_loss_weight=1.0, #Making it a tunable hyperparameter
        use_crps_channels=None, # list of channels using crps.- 
        precip_scaling_json=None, #File needs to be loaded at initialisation time in train.py.:; AsthanaSh
    ):
        super().__init__()
        self.save_hyperparameters()
        self.unet = DownscalingUnet(in_ch, out_ch, features)

        self.precip_channel_idx = precip_channel_idx
        self.loss_fn_precip = torch.nn.HuberLoss(delta=self.hparams.huber_delta, reduction='mean')
        self.loss_fn_temp = torch.nn.MSELoss(reduction='mean')
        self.channel_names = channel_names if channel_names is not None else [f"channel_{i}" for i in range(out_ch)]
        self.unet_regr = unet_regr
        self.register_buffer("loss_weights", torch.ones(out_ch))
        self.loss_weights[precip_channel_idx] = 1

        self.loss_weights[precip_channel_idx] = precip_loss_weight


        self.use_crps_channels = use_crps_channels if use_crps_channels is not None else []



        #Note: these json parameters have to be saved in a file during preprocessing :::: RhiresD 
        if precip_scaling_json is not None and os.path.isfile(precip_scaling_json):
            with open(precip_scaling_json, 'r') as f:
                scaling=json.load(f)
                self.precip_epsilon= scaling["epsilon"]
                self.precip_mean= scaling["mean"]
                self.precip_std= scaling["std"]

    def forward(self, x):
            out = self.unet(x)
            # 
            # log(eps) = mean + std * z ------> z = (log(eps) - mu) / std 
            precip_idx = self.precip_channel_idx
            epsilon = self.precip_epsilon
            mean = self.precip_mean
            std = self.precip_std

            min_log = torch.log(torch.tensor(epsilon, device=out.device, dtype=out.dtype))
            min_z = (min_log - mean) / std

            out_precip = out[:, precip_idx:precip_idx+1, ...]

            out_precip = torch.maximum(out_precip, min_z)
            out = out.clone()


            out[:, precip_idx:precip_idx+1, ...] = out_precip
            return out


    def weighted_loss(self, y_hat, y):
        per_channel_losses = []
        for i in range(y_hat.shape[1]):
            if i in self.use_crps_channels:
                loss = crps_loss(y_hat[:, i, ...], y[:, i, ...])
            elif i == self.precip_channel_idx:
                loss = self.loss_fn_precip(y_hat[:, i, ...], y[:, i, ...])
            else:
                loss = self.loss_fn_temp(y_hat[:, i, ...], y[:, i, ...])
            weighted_loss = self.loss_weights[i] * loss
            per_channel_losses.append(weighted_loss)
        total_loss = torch.mean(torch.stack(per_channel_losses))
        return total_loss




    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.weighted_loss(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        for i, name in enumerate(self.channel_names):
            if i in self.use_crps_channels:
                crps = crps_loss(y_hat[:, i, ...], y[:, i, ...])
                self.log(f"train/{name}_crps", crps, on_epoch=True)
            elif i == self.precip_channel_idx:
                huber = self.loss_fn_precip(y_hat[:, i, ...], y[:, i, ...])
                self.log(f"train/{name}_huber", huber, on_epoch=True)
            else:
                mse = self.loss_fn_temp(y_hat[:, i, ...], y[:, i, ...])
                self.log(f"train/{name}_mse", mse, on_epoch=True)
        return loss




    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.weighted_loss(y_hat, y)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)



        for i, name in enumerate(self.channel_names):
            if i in self.use_crps_channels:
                crps = crps_loss(y_hat[:, i, ...], y[:, i, ...])
                self.log(f"val/{name}_crps", crps, on_epoch=True)
            elif i == self.precip_channel_idx:
                huber = self.loss_fn_precip(y_hat[:, i, ...], y[:, i, ...])
                self.log(f"val/{name}_huber", huber, on_epoch=True)
            else:
                mse = self.loss_fn_temp(y_hat[:, i, ...], y[:, i, ...])
                self.log(f"val/{name}_mse", mse, on_epoch=True)
        return loss



    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.weighted_loss(y_hat, y)
        self.log("test/loss", loss, on_epoch=True, prog_bar=True)
        return loss



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=getattr(self.hparams, "lr_scheduler", {}).get("factor", 0.5),
            patience=getattr(self.hparams, "lr_scheduler", {}).get("patience", 3),
            min_lr=getattr(self.hparams, "lr_scheduler", {}).get("min_lr", 1e-6),
        )


        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1
            }
        }

    def last_layer(self):
        return self.unet.outputs