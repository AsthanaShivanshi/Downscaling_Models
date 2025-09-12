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


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = DoubleConv(in_c, out_c)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = DoubleConv(out_c*2, out_c)
        
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


class DownscalingUnet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64,128,256,512]):
        super().__init__()
        self.e1 = EncoderBlock(in_ch, features[0])
        self.e2 = EncoderBlock(features[0], features[1])
        self.e3 = EncoderBlock(features[1], features[2])
        self.e4 = EncoderBlock(features[2], features[3])         
        self.b = DoubleConv(features[3], features[3]*2)         
        self.d1 = DecoderBlock(features[3]*2, features[3])
        self.d2 = DecoderBlock(features[3], features[2])
        self.d3 = DecoderBlock(features[2], features[1])
        self.d4 = DecoderBlock(features[1], features[0])         
        self.outputs = nn.Conv2d(features[0], out_ch, kernel_size=1, padding=0)

    def forward(self, x):

        #og size before padding
        original_height=x.shape[2] #OG dims : batchsize, channel, height and width
        original_width=x.shape[3]
        #padding to make it multiple of 16 for feeding into UNet
        pad_height = (16 - original_height % 16) % 16
        pad_width = (16 - original_width % 16) % 16
        x_padded = F.pad(x, (0, pad_width, 0, pad_height))

        # Encoder
        s1, p1 = self.e1(x_padded)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        # Bottleneck

        b = self.b(p4)

        # Decoder
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        # Output
        out = self.outputs(d4)
        out_cropped= out[:,:,:original_height, :original_width]
        return out_cropped



class DownscalingUnetLightning(LightningModule):
    def __init__(self, in_ch=1, out_ch=1, features=[64,128,256,512], channel_names=None):
        super().__init__()
        self.unet = DownscalingUnet(in_ch, out_ch, features)
        self.loss_fn = nn.MSELoss()
        # Pass channel names from config
        self.channel_names = channel_names if channel_names is not None else [f"channel_{i}" for i in range(out_ch)]

    def forward(self, x):
        return self.unet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        per_channel_loss = ((y_hat - y) ** 2).mean(dim=(0, 2, 3))
        for i, loss in enumerate(per_channel_loss):
            self.log(f"train_loss_{self.channel_names[i]}", loss, on_epoch=True, prog_bar=True)
        total_loss = per_channel_loss.mean()
        self.log("train_loss", total_loss, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        per_channel_loss = ((y_hat - y) ** 2).mean(dim=(0, 2, 3))
        for i, loss in enumerate(per_channel_loss):
            self.log(f"val_loss_{self.channel_names[i]}", loss, on_epoch=True, prog_bar=True)
        total_loss = per_channel_loss.mean()
        self.log("val/loss", total_loss, on_epoch=True, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def last_layer(self):
        return self.unet.outputs