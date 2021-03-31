import torch
import torch.nn as nn

# local
from models.CSPDarknet53 import CSPDarknet53

# ------
# Unit of Convolution Layer
# Conv2d + BatchNormalization + LeakyReLU
# ------
class ConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ConvUnit, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=kernel_size//2, 
            bias=False
        )
        self.batchnrom = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnrom(x)
        x = self.activation(x)
        return x

# ------
# Spatial Pyramid Pooling (SPP)
# 5x5, 9x9, 13x13 maxpooling
# return a feature map with same size 
# ------
class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5,9,13]):
        super(SpatialPyramidPooling, self).__init__()
        self.maxpools = nn.ModuleList(
            [nn.MaxPool2d(ps,1,ps//2) for ps in pool_sizes]
        )

    def forward(self, x):
        # feature maps of 5x5, 9x9, 13x13
        fms = [maxpool(x) for maxpool in self.maxpools[::-1]]
        # I don't why to reverse the sequence of the maxpools, but I need their pre-trained model.
        x = torch.cat(fms + [x], dim=1)
        return x

# ------
# Upsampling
# integrates channels then upsample with a factor of 2
# ------
class ConvUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvUpsample, self).__init__()
        self.upsample = nn.Sequential(
            ConvUnit(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x):
        x = self.upsample(x)
        return x

# ------
# YOLO head
# ------
class YOLOHead(nn.Module):
    def __init__(self, channels:list, in_channels):
        super(YOLOHead, self).__init__()
        self.head = nn.Sequential(
            ConvUnit(in_channels, channels[0], 3),
            nn.Conv2d(channels[0], channels[1], 1)
        )
    
    def forward(self, x):
        return self.head(x)


def make_three_conv(channels:list, in_channels):
    convs = nn.Sequential(
        ConvUnit(in_channels, channels[0], 1),
        ConvUnit(channels[0], channels[1], 3),
        ConvUnit(channels[1], channels[0], 1)
    )
    return convs


def make_five_conv(channels:list, in_channels):
    convs = nn.Sequential(
        ConvUnit(in_channels, channels[0], 1),
        ConvUnit(channels[0], channels[1], 3),
        ConvUnit(channels[1], channels[0], 1),
        ConvUnit(channels[0], channels[1], 3),
        ConvUnit(channels[1], channels[0], 1)
    )
    return convs


# ------
# YOLOv4!!!
# ------
class YOLOv4(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super(YOLOv4, self).__init__()
        self.backbone = CSPDarknet53()
        # one anchor has object:bool, [cx, cy, w, h], and its class
        head_channels = (num_anchors//3) * (5 + num_classes)

        # necks
        # pyramid 5 to SPP to head 1
        self.conv_p5_1 = make_three_conv([512,1024], 1024)
        self.SPP = SpatialPyramidPooling()
        self.conv_p5_2 = make_three_conv([512,1024], 2048)
        self.upsample_p5 = ConvUpsample(512, 256)  # upsampling
        self.conv_p5_3 = make_five_conv([512, 1024], 1024)
        self.head_1 = YOLOHead([1024, head_channels], 512)  # head 1
        
        # pyramid 4 to head 2
        self.conv_p4_1 = ConvUnit(512, 256, 1)
        self.conv_p4_2 = make_five_conv([256, 512], 512)
        self.upsample_p4 = ConvUpsample(256, 128)  # upsampling
        self.downsample_p4 = ConvUnit(256, 512, 3, stride=2)  # donwsampling
        self.conv_p4_3 = make_five_conv([256, 512], 512)
        self.head_2 = YOLOHead([512, head_channels], 256)  # head 2
        
        # pyramid 3 to head 3
        self.conv_p3_1 = ConvUnit(256, 128, 1)
        self.conv_p3_2 = make_five_conv([128,256], 256)
        self.downsample_p3 = ConvUnit(128, 256, 3, stride=2)  # downsampling
        self.head_3 = YOLOHead([256, head_channels], 128)  # head 3


    def forward(self, x):
        # extract p3, p4, p5 from backbone
        p3, p4, p5 = self.backbone(x)

        # p5 pipeline (head)
        p5 = self.conv_p5_1(p5)
        p5 = self.SPP(p5)
        p5 = self.conv_p5_2(p5)
        p5_up = self.upsample_p5(p5)

        # p4 pipeline (head)
        p4 = self.conv_p4_1(p4)
        p4 = torch.cat([p4,p5_up], dim=1)
        p4 = self.conv_p4_2(p4)
        p4_up = self.upsample_p4(p4)

        # p3 pipeline
        p3 = self.conv_p3_1(p3)
        p3 = torch.cat([p3,p4_up], dim=1)
        p3 = self.conv_p3_2(p3)
        p3_down = self.downsample_p3(p3)

        # p4 pipeline (tail)
        p4 = torch.cat([p3_down, p4], dim=1)
        p4 = self.conv_p4_3(p4)
        p4_down = self.downsample_p4(p4)

        # p5 pipline (tail)
        p5 = torch.cat([p4_down,p5], dim=1)
        p5 = self.conv_p5_3(p5)

        # heads
        h1 = self.head_1(p5)
        h2 = self.head_2(p4)
        h3 = self.head_3(p3)

        return h1, h2, h3