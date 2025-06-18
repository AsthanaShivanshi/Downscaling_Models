import torch
import torch.nn as nn #Neural network module based on 
#https://github.com/DSIP-FBK/DiffScaler/blob/main/src/models/components/unet.py hierarchy of models. First one is a UNet I have customised it to my requirements here. 

#Please note that to make use of this architecture, you need to pad the input first so that the layers can be used as is (multiples of 16) 
#At the end, we have to not forget to crop the output back to the original size (265x370).


import torch.nn.functional as F

#################################Unet building blocks#####################################
class DoubleConv(nn.Module): #2 convolutional layers, batchnorm and ReLU
    def __init__(self, in_channels, out_channels): #Two convolutional layers stacked together 
        super(DoubleConv, self).__init__()

        self.conv= nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(inplace=True)) #Two layers
        
    def forward(self,inputs):
        return self.conv(inputs) #Forward pass through the above two convolutional layers
    

class Encoder_Block(nn.Module): #Encoder block for downsampling spatial dimensions and for extracting features
    def __init__(self, in_channels, out_channels):
        super(Encoder_Block, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels) #Extracts features at the current resolution
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) #Halves the spatial size

        #Forward pass through the encoder block defined aboce

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p #Output of the encoder block is the output of the conv layer and the pooled output
    

class Decoder_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder_Block, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        if x.shape[2:] != skip.shape[2:]:
            x = torch.nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

#xxxxxxxxxxxxxxxxxxxxxxxxUNet architecturexxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class UNet(nn.Module):

    def __init__(self, in_channels=2, out_channels=2, features=[64, 128, 256, 512]): #number of filters at each level
        super(UNet, self).__init__()
        self.Encoder1 = Encoder_Block(in_channels, features[0])
        self.Encoder2 = Encoder_Block(features[0], features[1])
        self.Encoder3 = Encoder_Block(features[1], features[2])
        self.Encoder4 = Encoder_Block(features[2], features[3])

        #Bottleneck is the deepest part of the network, 
        self.bottleneck = DoubleConv(features[3], features[3]*2)

        #Decoder blocks
        self.Decoder1 = Decoder_Block(features[3]*2, features[3])   
        self.Decoder2 = Decoder_Block(features[3], features[2])
        self.Decoder3 = Decoder_Block(features[2], features[1])
        self.Decoder4 = Decoder_Block(features[1], features[0]) #Each step doubles the spatial resolution

        #Final convolutional layer

        self.outputs= nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self,inputs, targets=None):
        #Original size has to be stored
        original_height=inputs.shape[2]
        original_width=inputs.shape[3]

        #Padding to make sure the input is a multiple of 16
        padding_height= (16 - original_height % 16) % 16
        padding_width= (16 - original_width % 16) % 16

        inputs= F.pad(inputs, (0, padding_width, 0, padding_height))




        ###Encoder###
        s1,p1= self.Encoder1(inputs)
        s2,p2= self.Encoder2(p1)
        s3,p3= self.Encoder3(p2)
        s4,p4= self.Encoder4(p3)

        #Bottleneck
        b= self.bottleneck(p4)

        ###Decoder###

        d1= self.Decoder1(b, s4)
        d2= self.Decoder2(d1, s3)
        d3= self.Decoder3(d2, s2)
        d4= self.Decoder4(d3, s1)

        outputs= self.outputs(d4) 

        #This output has to be cropped back to the original size due to the padding introduced earlier

        outputs= outputs[ :, :, :original_height, :original_width] 
    
        #Adding residuals back the output to produce the final output
        inputs_cropped = inputs[:, :, :original_height, :original_width]

        final_outputs = outputs + inputs_cropped

        return final_outputs




