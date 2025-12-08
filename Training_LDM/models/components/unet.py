import torch
import torch.nn as nn
from lightning import LightningModule
import torch.nn.functional as F

#Additional citation : https://github.com/soof-golan/spacecutter-lightning/tree/14febad15c47ba1861c7c3c3397a1929fd47e65d/README.md

#AsthanaSh : This is the reference Unet upon which the GAN is based.
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(DoubleConv, self).__init__()
        # note: bias=false for the batchnorm step!
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
    
    def forward(self, inputs):
        x = self.conv(inputs)
        return x


class Encoder_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class Decoder_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)
    def forward(self, inputs, skip):
        x = self.up(inputs)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class DownscalingUnet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64,128,256,512]):
        super().__init__()
        self.Encoder1 = Encoder_Block(in_ch, features[0])
        self.Encoder2 = Encoder_Block(features[0], features[1])
        self.Encoder3 = Encoder_Block(features[1], features[2])
        self.Encoder4 = Encoder_Block(features[2], features[3])
        self.bottleneck = DoubleConv(features[3], features[3]*2)
        self.Decoder1 = Decoder_Block(features[3]*2, features[3])
        self.Decoder2 = Decoder_Block(features[3], features[2])
        self.Decoder3 = Decoder_Block(features[2], features[1])
        self.Decoder4 = Decoder_Block(features[1], features[0])
        self.outputs = nn.Conv2d(features[0], out_ch, kernel_size=1)

        # He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


    def forward(self, x):
        original_height = x.shape[2]
        original_width = x.shape[3]
        pad_height = (16 - original_height % 16) % 16
        pad_width = (16 - original_width % 16) % 16
        x_padded = F.pad(x, (0, pad_width, 0, pad_height))

        s1, p1 = self.Encoder1(x_padded)
        s2, p2 = self.Encoder2(p1)
        s3, p3 = self.Encoder3(p2)
        s4, p4 = self.Encoder4(p3)
        b = self.bottleneck(p4)
        d1 = self.Decoder1(b, s4)
        d2 = self.Decoder2(d1, s3)
        d3 = self.Decoder3(d2, s2)
        d4 = self.Decoder4(d3, s1)
        out = self.outputs(d4)
        out_cropped = out[:, :, :original_height, :original_width]
        x_cropped = x[:, :, :original_height, :original_width]
        final_out = out_cropped + x_cropped[:, :out_cropped.shape[1], :, :]
        final_out[:, 0:1, :, :] = F.relu(final_out[:, 0:1, :, :]) # Only for precip channel
        return final_out



class DownscalingUnetLightning(LightningModule):
    def __init__(self, in_ch=1, out_ch=1, features=[64,128,256,512], channel_names=None, unet_regr=None, precip_channel_idx=0):
        super().__init__()
        self.unet = DownscalingUnet(in_ch, out_ch, features)
        self.loss_fn = nn.MSELoss()
        # Pass channel names from config
        self.channel_names = channel_names if channel_names is not None else [f"channel_{i}" for i in range(out_ch)]
        self.unet_regr= unet_regr
        self.register_buffer("loss_weights", torch.ones(out_ch))
        self.loss_weights[precip_channel_idx]=1 #equal weight to all channels after transform


    def forward(self, x):
        return self.unet(x)
    
    #weighted loss needed: due to channel loss not going down for precip
    def compute_weighted_loss(self, y_hat, y):

        mse = ((y_hat - y) ** 2).mean(dim=(2, 3))  
      
        per_channel_loss = mse.mean(dim=0) * self.loss_weights
        return per_channel_loss


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        per_channel_loss = self.compute_weighted_loss(y_hat, y)


        for i, loss in enumerate(per_channel_loss):
            self.log(f"train_loss_{self.channel_names[i]}", loss, on_step=False, on_epoch=True, prog_bar=True)
        total_loss = per_channel_loss.sum() #Changed from mean to sum for weighted loss, because precip loss was not going down. 
        self.log("train_loss", total_loss, on_epoch=True, prog_bar=True)
        return total_loss
    
    

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        per_channel_loss = self.compute_weighted_loss(y_hat, y)
        for i, loss in enumerate(per_channel_loss):
            self.log(f"val_loss_{self.channel_names[i]}", loss, on_step=False, on_epoch=True, prog_bar=True)
        total_loss = per_channel_loss.sum()
        self.log("val/loss", total_loss, on_epoch=True, prog_bar=True)
        return total_loss
    
    #testing step to get of the error
    def test_step(self, batch, batch_idx):


        x, y = batch
        y_hat = self(x)
        per_channel_loss = self.compute_weighted_loss(y_hat, y)
        for i, loss in enumerate(per_channel_loss):
            self.log(f"test_loss_{self.channel_names[i]}", loss, on_step=False, on_epoch=True, prog_bar=True)
        total_loss = per_channel_loss.sum()
        self.log("test/loss", total_loss, on_epoch=True, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.8,
            patience=3,  
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