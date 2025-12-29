from typing import Any

import torch
from lightning import LightningModule

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
        huber_delta=1.0  #only for precip,,, smaller delta might be better for low precip values. 
    ):
        super().__init__()
        self.save_hyperparameters()
        self.unet = DownscalingUnet(in_ch, out_ch, features)

        # Loss: Huber, MSE : precip, temp
        self.precip_channel_idx = precip_channel_idx
        self.loss_fn_precip = torch.nn.HuberLoss(delta=self.hparams.huber_delta, reduction='mean') #precipitation



        self.loss_fn_temp = torch.nn.MSELoss(reduction='mean') #temp
        self.channel_names = channel_names if channel_names is not None else [f"channel_{i}" for i in range(out_ch)]
        self.unet_regr = unet_regr
        self.register_buffer("loss_weights", torch.ones(out_ch))
        self.loss_weights[precip_channel_idx] = 1


    def forward(self, x):
        return self.unet(x)

    def weighted_loss(self, y_hat, y):
        per_channel_losses = []
        for i in range(y_hat.shape[1]):
            if i == self.precip_channel_idx:
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

        # Per-channel metrics (add this block)
        for i, name in enumerate(self.channel_names):
            if i == self.precip_channel_idx:
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

        # Per-channel metrics
        for i, name in enumerate(self.channel_names):
            if i == self.precip_channel_idx:
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