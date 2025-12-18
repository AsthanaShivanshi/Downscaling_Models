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

    def __init__(self, in_ch=1, out_ch=1, features=[64,128,256,512], channel_names=None, unet_regr=None, precip_channel_idx=0, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.unet = DownscalingUnet(in_ch, out_ch, features)

        self.loss_fn = torch.nn.MSELoss()
        self.channel_names = channel_names if channel_names is not None else [f"channel_{i}" for i in range(out_ch)]
        self.unet_regr = unet_regr
        self.register_buffer("loss_weights", torch.ones(out_ch))
        self.loss_weights[precip_channel_idx] = 10  #Customisable weight for precip channel depending on requirement. Gives more wt to loss of precip in that case 



    def forward(self, x):
        return self.unet(x)



    def weighted_loss(self, y_hat, y):
        mse = ((y_hat - y) ** 2).mean(dim=(2, 3))
        per_channel_loss = mse.mean(dim=0) * self.loss_weights
        return per_channel_loss



    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        per_channel_loss = self.weighted_loss(y_hat, y)
        for i, loss in enumerate(per_channel_loss):
            self.log(f"train_loss_{self.channel_names[i]}", loss, on_step=False, on_epoch=True, prog_bar=True)
        total_loss = per_channel_loss.sum()
        self.log("train_loss", total_loss, on_epoch=True, prog_bar=True)
        return total_loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        per_channel_loss = self.weighted_loss(y_hat, y)
        for i, loss in enumerate(per_channel_loss):
            self.log(f"val_loss_{self.channel_names[i]}", loss, on_step=False, on_epoch=True, prog_bar=True)
        total_loss = per_channel_loss.sum()
        self.log("val/loss", total_loss, on_epoch=True, prog_bar=True)
        return total_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        per_channel_loss = self.weighted_loss(y_hat, y)
        for i, loss in enumerate(per_channel_loss):
            self.log(f"test_loss_{self.channel_names[i]}", loss, on_step=False, on_epoch=True, prog_bar=True)
        total_loss = per_channel_loss.sum()
        self.log("test/loss", total_loss, on_epoch=True, prog_bar=True)
        return total_loss

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