import json
import os

import torch
from lightning import LightningModule

from .components.unet import DownscalingUnet


class DownscalingUnetLightning(LightningModule):
    def __init__(
        self,
        in_ch=1,
        out_ch=1,
        features=None,
        channel_names=None,
        unet_regr=None,
        precip_channel_idx=0,
        lr=1e-3,
        huber_delta=1.0,
        precip_loss_weight=1.0,
        use_crps_channels=None,
        precip_scaling_json=None,
    ):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.save_hyperparameters()
        self.unet = DownscalingUnet(in_ch, out_ch, features)

        self.precip_channel_idx = precip_channel_idx
        self.loss_fn_precip = torch.nn.HuberLoss(delta=huber_delta, reduction="none")
        self.loss_fn_temp = torch.nn.L1Loss(reduction="none")

        self.channel_names = channel_names if channel_names is not None else [f"channel_{i}" for i in range(out_ch)]
        self.unet_regr = unet_regr
        self.use_crps_channels = use_crps_channels if use_crps_channels is not None else []

        self.register_buffer("loss_weights", torch.ones(out_ch))
        self.loss_weights[precip_channel_idx] = precip_loss_weight

        self.precip_epsilon = None
        self.precip_mean = None
        self.precip_std = None

        if precip_scaling_json is not None and os.path.isfile(precip_scaling_json):
            with open(precip_scaling_json, "r") as f:
                scaling = json.load(f)
            self.precip_epsilon = scaling["epsilon"]
            self.precip_mean = scaling["mean"]
            self.precip_std = scaling["std"]

    def forward(self, x):
        out = self.unet(x)

        if self.precip_epsilon is None or self.precip_mean is None or self.precip_std is None:
            return out

        precip_idx = self.precip_channel_idx
        epsilon = out.new_tensor(self.precip_epsilon)
        mean = out.new_tensor(self.precip_mean)
        std = out.new_tensor(self.precip_std)

        min_z = (torch.log(epsilon) - mean) / std

        out = out.clone()
        out[:, precip_idx:precip_idx + 1, ...] = torch.maximum(
            out[:, precip_idx:precip_idx + 1, ...],
            min_z,
        )
        return out

    def weighted_loss(self, y_hat, y):
        per_channel_losses = []

        for i in range(y_hat.shape[1]):
            if i == self.precip_channel_idx:
                loss = self.loss_fn_precip(y_hat[:, i, ...], y[:, i, ...]).mean()
            else:
                loss = self.loss_fn_temp(y_hat[:, i, ...], y[:, i, ...]).mean()
            per_channel_losses.append(self.loss_weights[i] * loss)

        return torch.stack(per_channel_losses).mean()
    




    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        total_loss = self.weighted_loss(y_hat, y)
        self.log("train_loss", total_loss, on_epoch=True, prog_bar=True)

        for i, name in enumerate(self.channel_names):
            if i == self.precip_channel_idx:
                channel_loss = self.loss_fn_precip(y_hat[:, i, ...], y[:, i, ...]).mean()
                self.log(f"train/{name}_huber", channel_loss, on_epoch=True)
            else:
                channel_loss = self.loss_fn_temp(y_hat[:, i, ...], y[:, i, ...]).mean()
                self.log(f"train/{name}_loss", channel_loss, on_epoch=True)

        return total_loss
    



    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        total_loss = self.weighted_loss(y_hat, y)
        self.log("val/loss", total_loss, on_epoch=True, prog_bar=True)

        for i, name in enumerate(self.channel_names):
            if i == self.precip_channel_idx:
                channel_loss = self.loss_fn_precip(y_hat[:, i, ...], y[:, i, ...]).mean()
                self.log(f"val/{name}_huber", channel_loss, on_epoch=True)
            else:
                channel_loss = self.loss_fn_temp(y_hat[:, i, ...], y[:, i, ...]).mean()
                self.log(f"val/{name}_loss", channel_loss, on_epoch=True)

        return total_loss




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
            mode="min",
            factor=self.hparams.lr_scheduler_factor,
            patience=self.hparams.lr_scheduler_patience,
            min_lr=self.hparams.lr_scheduler_min_lr,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }



    def last_layer(self):
        return self.unet.outputs