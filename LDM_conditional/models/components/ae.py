import numpy as np
import torch.nn as nn

from .resnet import ResBlock2D


class SimpleConvEncoder(nn.Module):
    def __init__(self, in_dim=1, levels=2, min_ch=16, ch_mult: int = 4):
        super().__init__()
        self.in_dim = in_dim
        self.levels = levels
        sequence = []
        channels = np.hstack([
            in_dim, 
            (in_dim*ch_mult**np.arange(1,levels+1)).clip(min=min_ch)
        ])
        
        for i in range(levels):
            in_channels = int(channels[i])
            out_channels = int(channels[i+1])
            res_block = ResBlock2D(
                in_channels, out_channels,
                kernel_size=(3,3),
                norm_kwargs={"num_groups": 1}  # <--- revert to original
            )
            sequence.append(res_block)
            downsample = nn.Conv2d(out_channels, out_channels,
                kernel_size=(2,2), stride=(2,2))
            sequence.append(downsample)
            in_channels = out_channels
        self.net = nn.Sequential(*sequence)

    def forward(self, x):
        return self.net(x)


class SimpleConvDecoder(nn.Module):
    def __init__(self, in_dim=1, levels=2, min_ch=16, out_dim=4, ch_mult: int = 4):
        super().__init__()
        self.in_dim = in_dim
        self.levels = levels
        self.out_dim = out_dim 
        encoder_channels = np.hstack([
            in_dim, 
            (in_dim*ch_mult**np.arange(1,levels+1)).clip(min=min_ch)
        ])
        decoder_channels = list(encoder_channels[::-1])
        decoder_channels[0] = in_dim
        decoder_channels[-1] = out_dim

        sequence = []
        for i in range(levels):
            in_channels = int(decoder_channels[i])
            out_channels = int(decoder_channels[i+1])
            upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2,2), stride=(2,2))
            sequence.append(upsample)
            res_block = ResBlock2D(
                out_channels, out_channels,
                kernel_size=(3,3),
                norm_kwargs={"num_groups": 1}
            )
            sequence.append(res_block)
        self.net = nn.Sequential(*sequence)

    def forward(self, x):
        return self.net(x)
    
    def last_layer(self):
        return self.net[-1].sequence[-1]
