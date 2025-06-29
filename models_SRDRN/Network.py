#!/usr/bin/env python
#title           :Network.py
#description     :Architecture file for multivariate bias correction
#author          :Fang Wang,Di Tian
#date            :2023/2/22
#usage           :from Network import Generator
#python_version  :3.10

#Translated from original tensorflow to pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, stride=1, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(channels, momentum=0.5)
        self.prelu = nn.PReLU(num_parameters=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, stride=1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(channels, momentum=0.5)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return out

class Generator(nn.Module):
    def __init__(self, in_channels=5, out_channels=4, num_res_blocks=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.prelu1 = nn.PReLU(num_parameters=64)
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualBlock(64, 3) for _ in range(num_res_blocks)])
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64, momentum=0.5)
        # Upsampling blocks
        self.up1 = nn.Conv2d(64, 512, kernel_size=3, stride=1, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.prelu2 = nn.PReLU(num_parameters=512)
        self.up2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.prelu3 = nn.PReLU(num_parameters=512)
        self.dropout = nn.Dropout2d(p=0.5)
        self.conv3 = nn.Conv2d(512, out_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.prelu1(x1)
        x2 = self.res_blocks(x1)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x = x1 + x2
        x = self.up1(x)
        x = self.upsample1(x)
        x = self.prelu2(x)
        x = self.up2(x)
        x = self.upsample2(x)
        x = self.prelu3(x)
        x = self.dropout(x)
        x = self.conv3(x)
        return x