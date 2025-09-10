import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# from .conv import Conv
from lib.models.model_zoo.conv import Conv
from lib.models.model_zoo.modules import ASPP, DACBlock, DACBlock2, AIM

try:
    from .UAFM import*
except ImportError:
    from UAFM import*

# from .modules import ASPP, DACBlock, DACBlock2, AIM
# from conv import Conv
# from modules import ASPP, DACBlock, DACBlock2, AIM

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))
    
# ---------------- Encoder: Pretrained ResNet-50 ----------------
class ResNet50Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Use layers up to the final conv
        self.resnet = resnet
        self.initial = nn.Sequential(
            resnet.conv1,  # 7x7 conv
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        # self.layer0 = nn.Sequential(
        #     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),  # Change stride to 1
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True)
        # )  # 64 channels
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels

    def forward(self, x):
        # print(self.resnet)
        
        l0 = self.initial(x)
        # l0 = self.layer0(x)
        l1 = self.layer1(l0)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        return l0, l1, l2, l3, l4  # return final output and skip connections

# ---------------- Decoder ----------------
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

# ---------------- Segmentation Model ----------------
class ResNet50UNet(nn.Module):
    def __init__(self, num_classes=4, pretrained=True, name="ResNet50UNet_DAC"):
        super().__init__()
        self.name = name
        self.encoder = ResNet50Encoder(pretrained=pretrained)
        self.sppf = SPPF(2048, 2048, k=3)
        self.aspp = ASPP(2048, 2048)
        
        # self.aim = AIM(iC_list=(64, 256, 512, 1024, 2048), oC_list=(64, 256, 512, 1024, 2048))
        self.dac1 = DACBlock2(256)
        self.dac2 = DACBlock2(512)
        self.dac3 = DACBlock2(1024)
        self.dac4 = DACBlock2(2048)

        self.decoder4 = DecoderBlock(6144, 1024, 512) # 2048 from encoder + 1024 skip
        self.decoder3 = DecoderBlock(512, 512, 256)
        self.decoder2 = DecoderBlock(256, 256, 128)
        # self.decoder1 = DecoderBlock(128, 64, 64)

        self.upconv = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(128, num_classes, kernel_size=1)
        # self.convc = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        p0, p1, p2, p3, p4 = self.encoder(x)
     
        # p0 = F.interpolate(p0, scale_factor=2, mode="bilinear", align_corners=False)
        # l0, l1, l2, l3, l4 = self.aim(p0, p1, p2, p3, p4)

        l1 = self.dac1(p1)
        l2 = self.dac2(p2)
        l3 = self.dac3(p3)
        l4 = self.dac4(p4)

        x1 = self.sppf(p4)
        x2 = l4 #self.aspp(p4)
        x = torch.cat([p4, x1, x2], dim=1)

        x = self.decoder4(x, l3)
        x = self.decoder3(x, l2)
        x = self.decoder2(x, l1)

        # Optional: skip connection from initial layer (if you want)
        # x = self.decoder1(x, skips_from_initial)
        x = self.upconv(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # upsample to original size
        # print("After decoder1 shape:", x.shape)
        x = self.final_conv(x)
        return x


# =======================U-Net with DAC + RMP =======================
# --- Encoder Block ---
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

# --- RMP Block (Residual Multi-kernel Pooling) ---
class RMPBlock(nn.Module):
    def __init__(self, in_channels, pool_sizes=[2, 3, 5, 6]):
        super(RMPBlock, self).__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=ps),
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            ) for ps in pool_sizes
        ])
        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels * (len(pool_sizes) + 1), in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.size()[2:]
        out = [x]
        for stage in self.stages:
            y = stage(x)
            y = nn.functional.interpolate(y, size=size, mode="bilinear", align_corners=True)
            out.append(y)
        out = torch.cat(out, dim=1)
        return self.conv_out(out)

# --- U-Net with DAC + RMP bottleneck ---
class UNetDACRMP(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNetDACRMP, self).__init__()
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck: DAC + RMP
        self.dac = DACBlock(512)
        self.rmp = RMPBlock(512)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)
        
        # Output
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        # Bottleneck with DAC + RMP
        b = self.dac(self.pool4(e4))
        b = self.rmp(b)
        
        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        return self.out_conv(d1)

# ===========================U-Net with DAC =======================

# --- Encoder Block (Conv + BN + ReLU twice) ---


# --- U-Net with DAC Bottleneck ---
class UNetDAC(nn.Module):
    def __init__(self, out_channels=4, pretrained=True):
        super(UNetDAC, self).__init__()
        
        # Encoder
        resent = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) if pretrained else models.resnet50(weights=None)
        self.enc1 = nn.Sequential(resent.conv1, resent.bn1, resent.relu)  # 64 channels
        self.enc2 = nn.Sequential(resent.maxpool, resent.layer1) # 128 channels
        self.enc3 = resent.layer2 # 256 channels
        self.enc4 = resent.layer3 # 512 channels

        # Bottleneck with DAC
        self.bottleneck = DACBlock(512)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)
        
        # Output
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool4(e4))
        
        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        return self.out_conv(d1)





if __name__ == "__main__":
    model = ResNet50UNet(num_classes=4, pretrained=True)
    # model = UNetDAC(in_channels=3, out_channels=2) 
    # model = UNetDACRMP(in_channels=3, out_channels=2) # for 2-class segmentation
    x = torch.randn(2, 3, 512, 512)  # batch=2, RGB 512x512
    y = model(x)
    print("Input shape:", x.shape)    # (batch, 3, H, W)
    print("Output shape:", y.shape)  # (batch, num_classes, H, W)
