import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import math
import numpy as np

class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class EfficientDilatedConv(nn.Module):
    """More efficient dilated convolution block with depthwise separable convolutions"""
    def __init__(self, inplanes, planes, dilation_rates=[1, 6, 12, 18]):
        super(EfficientDilatedConv, self).__init__()
        self.dilated_convs = nn.ModuleList()
        
        for rate in dilation_rates:
            # Depthwise separable convolution
            conv_block = nn.Sequential(
                # Depthwise convolution
                nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, 
                         padding=rate, dilation=rate, groups=inplanes, bias=False),
                nn.BatchNorm2d(inplanes),
                nn.ReLU(inplace=True),
                # Pointwise convolution
                nn.Conv2d(inplanes, planes//len(dilation_rates), kernel_size=1, bias=False),
                nn.BatchNorm2d(planes//len(dilation_rates)),
                nn.ReLU(inplace=True)
            )
            self.dilated_convs.append(conv_block)
        
        # Global average pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inplanes, planes//len(dilation_rates), 1, bias=False),
            nn.BatchNorm2d(planes//len(dilation_rates)),
            nn.ReLU(inplace=True)
        )
        
        # Final fusion
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(planes + planes//len(dilation_rates), planes, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        results = []
        for conv in self.dilated_convs:
            results.append(conv(x))
        
        # Global pooling branch
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
        results.append(global_feat)
        
        # Concatenate and fuse
        out = torch.cat(results, dim=1)
        out = self.fuse_conv(out)
        
        return out

class ImprovedConv2n(nn.Module):
    """Improved two-scale convolution with residual connections and attention"""
    def __init__(self, in_hc=64, in_lc=256, out_c=64, main=0):
        super(ImprovedConv2n, self).__init__()
        self.main = main
        mid_c = min(in_hc, in_lc)
        
        # Improved activation and normalization
        self.act = nn.GELU()
        self.h2l_pool = nn.AvgPool2d(2, stride=2)
        self.l2h_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # Stage 0 with residual connections
        self.h2h_0 = nn.Sequential(
            nn.Conv2d(in_hc, mid_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid_c),
            self.act
        )
        self.l2l_0 = nn.Sequential(
            nn.Conv2d(in_lc, mid_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid_c),
            self.act
        )

        # Enhanced stage 1 with separable convolutions
        self.h2h_1 = self._make_separable_conv(mid_c, mid_c, dilation=2)
        self.h2l_1 = self._make_separable_conv(mid_c, mid_c, dilation=4)
        self.l2h_1 = self._make_separable_conv(mid_c, mid_c, dilation=6)
        self.l2l_1 = self._make_separable_conv(mid_c, mid_c, dilation=8)

        # Attention modules
        self.cbam_h = CBAM(mid_c)
        self.cbam_l = CBAM(mid_c)

        if self.main == 0:
            self.h2h_2 = self._make_separable_conv(mid_c, mid_c, dilation=2)
            self.l2h_2 = self._make_separable_conv(mid_c, mid_c, dilation=2)
            self.h2h_3 = nn.Sequential(
                nn.Conv2d(mid_c, out_c, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_c)
            )
            self.identity = nn.Conv2d(in_hc, out_c, 1)
        elif self.main == 1:
            self.h2l_2 = self._make_separable_conv(mid_c, mid_c, dilation=2)
            self.l2l_2 = self._make_separable_conv(mid_c, mid_c, dilation=2)
            self.l2l_3 = nn.Sequential(
                nn.Conv2d(mid_c, out_c, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_c)
            )
            self.identity = nn.Conv2d(in_lc, out_c, 1)

    def _make_separable_conv(self, in_c, out_c, dilation=1):
        return nn.Sequential(
            # Depthwise
            nn.Conv2d(in_c, in_c, 3, 1, dilation, dilation=dilation, groups=in_c, bias=False),
            nn.BatchNorm2d(in_c),
            self.act,
            # Pointwise
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, in_h, in_l):
        h = self.h2h_0(in_h)
        l = self.l2l_0(in_l)

        # Stage 1 with cross-connections
        h2h = self.h2h_1(h)
        h2l = self.h2l_1(self.h2l_pool(h))
        l2l = self.l2l_1(l)
        l2h = self.l2h_1(self.l2h_up(l))
        
        h = self.act(h2h + l2h)
        l = self.act(l2l + h2l)

        # Apply attention
        h = self.cbam_h(h)
        l = self.cbam_l(l)

        if self.main == 0:
            h2h = self.h2h_2(h)
            l2h = self.l2h_2(self.l2h_up(l))
            h_fuse = self.act(h2h + l2h)
            out = self.act(self.h2h_3(h_fuse) + self.identity(in_h))
        elif self.main == 1:
            h2l = self.h2l_2(self.h2l_pool(h))
            l2l = self.l2l_2(l)
            l_fuse = self.act(h2l + l2l)
            out = self.act(self.l2l_3(l_fuse) + self.identity(in_l))

        return out

class ImprovedDecoderBlock(nn.Module):
    """Enhanced decoder block with attention and residual connections"""
    def __init__(self, in_channels, skip_channels, out_channels, use_attention=True):
        super().__init__()
        
        # Main convolution path
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        # Residual connection
        self.residual = nn.Conv2d(in_channels + skip_channels, out_channels, 1) if in_channels + skip_channels != out_channels else nn.Identity()
        
        # Attention mechanism
        self.use_attention = use_attention
        if use_attention:
            self.attention = CBAM(out_channels)
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x, skip):
        # Upsample x to match skip connection size
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        
        # Concatenate skip connection
        x_cat = torch.cat([x, skip], dim=1)
        
        # Main path
        out = self.conv1(x_cat)
        out = self.dropout(out)
        out = self.conv2(out)
        
        # Residual connection
        residual = self.residual(x_cat)
        out = out + residual
        
        # Apply attention
        if self.use_attention:
            out = self.attention(out)
            
        return out

class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network for multi-scale feature fusion"""
    def __init__(self, feature_dims):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for dim in feature_dims:
            self.lateral_convs.append(
                nn.Conv2d(dim, 256, 1, bias=False)
            )
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(256, 256, 3, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, features):
        # features: list of tensors from low to high resolution
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        
        # Top-down pathway
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=laterals[i].shape[2:], mode='bilinear', align_corners=False
            )
        
        # Apply final convolutions
        outputs = [conv(lateral) for conv, lateral in zip(self.fpn_convs, laterals)]
        return outputs

class UOCNet(nn.Module):
    """Enhanced ResNet50UNet with modern architectural improvements"""
    def __init__(self, name = 'UOCNet', num_classes=4, pretrained=True, dropout_rate=0.1):
        super().__init__()
        self.name = name
        # Encoder
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        self.initial = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1  # 256
        self.layer2 = resnet.layer2  # 512
        self.layer3 = resnet.layer3  # 1024
        self.layer4 = resnet.layer4  # 2048
        
        # Feature enhancement modules
        self.efficient_aspp = EfficientDilatedConv(2048, 512)
        
        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork([64, 256, 512, 1024, 512])
        
        # Improved decoder with attention
        # self.decoder4 = ImprovedDecoderBlock(512, 256, 256)  # FPN output + skip
        # self.decoder3 = ImprovedDecoderBlock(256, 256, 128)
        # self.decoder2 = ImprovedDecoderBlock(128, 256, 64)
        # self.decoder1 = ImprovedDecoderBlock(64, 256, 32)

        self.decoder4 = ImprovedDecoderBlock(512, 0, 256)  # FPN output + skip
        self.decoder3 = ImprovedDecoderBlock(512, 0, 128)
        self.decoder2 = ImprovedDecoderBlock(384, 0, 64)
        self.decoder1 = ImprovedDecoderBlock(320, 0, 32)
        
        # Final layers
        self.final_upconv = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(16, num_classes, 1)
        )
        
        # Deep supervision (optional)
        self.deep_supervision = True
        if self.deep_supervision:
            self.aux_heads = nn.ModuleList([
                nn.Conv2d(256, num_classes, 1),
                nn.Conv2d(128, num_classes, 1),
                nn.Conv2d(64, num_classes, 1)
            ])

    def forward(self, x):
        input_size = x.shape[2:]
        
        # Encoder
        x0 = self.initial(x)    # 64, H/4, W/4
        x1 = self.layer1(x0)    # 256, H/4, W/4
        x2 = self.layer2(x1)    # 512, H/8, W/8
        x3 = self.layer3(x2)    # 1024, H/16, W/16
        x4 = self.layer4(x3)    # 2048, H/32, W/32
        # print(f'x0 = {x0.shape}, x1 = {x1.shape}, x2 = {x2.shape}, x3 = {x3.shape}, x4 = {x4.shape}')
        # Enhanced feature processing
        x4_enhanced = self.efficient_aspp(x4)
        # print(f'x_enhanced = {x4_enhanced.shape}')
        # Feature Pyramid Network
        fpn_features = self.fpn([x0, x1, x2, x3, x4_enhanced])
      
        # for i, f in enumerate(fpn_features):
        #     print(f'FPN{i}= ',f.shape)
        # Decoder with FPN features
        d4 = self.decoder4(fpn_features[4], fpn_features[3])
        d3 = self.decoder3(d4, fpn_features[2])
        d2 = self.decoder2(d3, fpn_features[1])
        d1 = self.decoder1(d2, fpn_features[0])
        
        # Final upsampling and prediction
        output = self.final_upconv(d1)
        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)
        output = self.final_conv(output)
        # print(f'output= {output.shape}')
        # Deep supervision during training
        # if self.training and self.deep_supervision:
        #     aux_outputs = []
        #     for i, aux_head in enumerate(self.aux_heads):
        #         if i == 0:  # d4
        #             aux_out = aux_head(d4)
        #         elif i == 1:  # d3
        #             aux_out = aux_head(d3)
        #         else:  # d2
        #             aux_out = aux_head(d2)
        #         aux_out = F.interpolate(aux_out, size=input_size, mode='bilinear', align_corners=False)
        #         aux_outputs.append(aux_out)
            
        #     return output, aux_outputs
        
        return output

# Loss function for deep supervision
class DeepSupervisionLoss(nn.Module):
    def __init__(self, weights=[1.0, 0.8, 0.6, 0.4]):
        super().__init__()
        self.weights = weights
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, outputs, target):
        if isinstance(outputs, tuple):
            main_output, aux_outputs = outputs
            loss = self.weights[0] * self.criterion(main_output, target)
            for i, aux_output in enumerate(aux_outputs):
                loss += self.weights[i+1] * self.criterion(aux_output, target)
            return loss
        else:
            return self.criterion(outputs, target)

if __name__ == "__main__":
    model = UOCNet(num_classes=4, pretrained=True)
  
    x = torch.randn(2, 3, 512, 512)  # batch=2, RGB 512x512
    y = model(x)
    print("Input shape:", x.shape)    # (batch, 3, H, W)
    print("Output shape:", y.shape)  # (batch, num_classes, H, W)