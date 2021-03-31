import torch.nn as nn
import torch.nn.functional as fun
import torch

# ------
# Mish activation
# ------
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(fun.softplus(x))

# ------
# Unit of Convolution Layer
# Conv2d + BatchNormalization + Mish
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
        self.activation = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnrom(x)
        x = self.activation(x)
        return x

# ------
# Residual Block
# ------
class ResBlock(nn.Module):
    def __init__(self, channels, hidden_channels=None):
        super(ResBlock, self).__init__()

        if hidden_channels is None:
            hidden_channels = channels

        self.block = nn.Sequential(
            ConvUnit(channels, hidden_channels, 1),
            ConvUnit(hidden_channels, channels, 3)
        )

    def forward(self, x):
        return x + self.block(x)

# ------
# CSP Residual block
# Downsampling by a factor of 2
# ------
class CSPResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, is_head):
        super(CSPResBlock, self).__init__()
        self.downsample_conv = ConvUnit(in_channels, out_channels, 3, stride=2)
        # head will double the channels
        if is_head:
            self.part1 = ConvUnit(out_channels, out_channels, 1)
            self.part2 = ConvUnit(out_channels, out_channels, 1)
            self.block = nn.Sequential(
                ResBlock(channels=out_channels, hidden_channels=out_channels//2),
                ConvUnit(out_channels, out_channels, 1)
            )
            self.conv_mix = ConvUnit(out_channels*2, out_channels, 1)
        else:
            self.part1 = ConvUnit(out_channels, out_channels//2, 1)
            self.part2 = ConvUnit(out_channels, out_channels//2, 1)
            self.block = nn.Sequential(
                *[ResBlock(out_channels//2) for _ in range(num_layers)],
                ConvUnit(out_channels//2, out_channels//2, 1)
            )
            self.conv_mix = ConvUnit(out_channels, out_channels, 1)

    def forward(self, x):
        # halve size
        x = self.downsample_conv(x)
        # part1 to shortcut
        x1 = self.part1(x)
        # part2 to resblocks
        x2 = self.part2(x)
        x2 = self.block(x2)
        # stack then mix
        x = torch.cat([x1,x2], dim=1)
        x = self.conv_mix(x)
        return x

# ------
# CSP Darknet
# returns 3 feature maps
# ------
class CSPDarknet53(nn.Module):
    def __init__(self):
        super(CSPDarknet53, self).__init__()
        self.conv1 = ConvUnit(3, 32, kernel_size=3, stride=1)
        self.stages = nn.ModuleList([
            CSPResBlock(32, 64, 1, is_head=True),
            CSPResBlock(64, 128, 2, is_head=False),
            CSPResBlock(128, 256, 8, is_head=False),
            CSPResBlock(256, 512, 8, is_head=False),
            CSPResBlock(512, 1024, 4, is_head=False)
        ])
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.stages[0](x)
        x = self.stages[1](x)
        p3 = self.stages[2](x)  # pyramid 3
        p4 = self.stages[3](p3)  # pyramid 4
        p5 = self.stages[4](p4)  # pyramid 5
        return p3, p4, p5

