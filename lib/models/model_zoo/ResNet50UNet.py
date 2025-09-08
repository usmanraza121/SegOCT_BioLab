import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from .conv import Conv

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
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels

    def forward(self, x):
        # print(self.resnet)
        skips = {}
        x = self.initial(x)
        skips["layer1"] = self.layer1(x)
        skips["layer2"] = self.layer2(skips["layer1"])
        skips["layer3"] = self.layer3(skips["layer2"])
        x = self.layer4(skips["layer3"])
        return x, skips

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
    def __init__(self, num_classes=4, pretrained=True, name="ResNet50UNet"):
        super().__init__()
        self.name = name
        self.encoder = ResNet50Encoder(pretrained=pretrained)

        self.decoder4 = DecoderBlock(2048, 1024, 512)
        self.decoder3 = DecoderBlock(512, 512, 256)
        self.decoder2 = DecoderBlock(256, 256, 128)
        self.decoder1 = DecoderBlock(128, 64, 64)
        self.sppf = SPPF(2048, 2048, k=3)
        self.upconv = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        x, skips = self.encoder(x)
        # print("Encoder output shape:", x.shape)
        # print("Skip connection shapes:", {k: v.shape for k, v in skips.items()})
        x = self.sppf(x)
        x = self.decoder4(x, skips["layer3"])
        # print("After decoder4 shape:", x.shape)
        x = self.decoder3(x, skips["layer2"])
        # print("After decoder3 shape:", x.shape)
        x = self.decoder2(x, skips["layer1"])
        # print("After decoder2 shape:", x.shape)
        # Optional: skip connection from initial layer (if you want)
        # x = self.decoder1(x, skips_from_initial)
        x = self.upconv(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # upsample to original size
        # print("After decoder1 shape:", x.shape)
        x = self.final_conv(x)
        return x

# ---------------- Example ----------------
if __name__ == "__main__":
    model = ResNet50UNet(num_classes=4, pretrained=True)
    x = torch.randn(2, 3, 512, 512)  # batch=2, RGB 512x512
    y = model(x)
    print("Output shape:", y.shape)  # (batch, num_classes, H, W)
