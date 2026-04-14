import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class ConvBNReLU(nn.Module):
    """3x3 (or kxk) Conv → BN → ReLU.  padding auto-calculated."""
    def __init__(self, c1, c2, k=3, d=1, act=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c1, c2, k, padding=autopad(k, d=d), dilation=d, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True) if act else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class Conv(nn.Module):
    """YOLOv8-style Conv used by MSFR."""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d),
                              groups=g, dilation=d, bias=False)
        self.bn   = nn.BatchNorm2d(c2)
        self.act  = self.default_act if act is True \
                    else act if isinstance(act, nn.Module) \
                    else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ============================================================
#   Pretrained ResNet-50
# ============================================================

class ResNet50Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        self.initial = nn.Sequential(
            resnet.conv1,   
            resnet.bn1,
            resnet.relu,
            resnet.maxpool  
        )                   
        self.layer1 = resnet.layer1   # out: (B, 256,  64,  64)
        self.layer2 = resnet.layer2   # out: (B, 512,  32,  32)
        self.layer3 = resnet.layer3   # out: (B,1024,  16,  16)
        self.layer4 = resnet.layer4   # out: (B,2048,   8,   8)

    def forward(self, x):
        p0 = self.initial(x)   # 64
        p1 = self.layer1(p0)   # 256
        p2 = self.layer2(p1)   # 512
        p3 = self.layer3(p2)   # 1024
        p4 = self.layer4(p3)   # 2048
        return p0, p1, p2, p3, p4


# ============================================================
#  MSFR: Multi-Scale Feature Refinement (MSFR) module at the bottleneck
# ============================================================

class SpatialPool(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m   = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        y0 = self.cv1(x)
        y1 = self.m(y0)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([y0, y1, y2, y3], dim=1))


class MSFR(nn.Module):
    def __init__(self, in_dim, out_dim=None, rates=(1, 2, 4)):
        super().__init__()
        if out_dim is None:
            out_dim = in_dim
        mid = in_dim // 2

        self.branches = nn.ModuleList()
        # 1×1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_dim, mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True)
        ))
        # dilated 3×3 branches
        for r in rates:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_dim, mid, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(mid), nn.ReLU(inplace=True)
            ))
        # global context branch
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_dim, mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True)
        )
        # fuse: 1×1 + 3 dilated + 1 pool = (2 + len(rates)) × mid channels
        self.fuse = nn.Sequential(
            nn.Conv2d(mid * (2 + len(rates)), out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True)
        )
        self.pool = SpatialPool(mid, out_dim)

    def forward(self, x):
        res    = [b(x) for b in self.branches]
        pooled = self.pool(res[0])
        img    = self.image_pool(x)
        img    = F.interpolate(img, size=x.shape[2:],
                               mode='bilinear', align_corners=False)
        res.append(img)
        fuse = self.fuse(torch.cat(res, dim=1))
        return fuse + pooled


# ============================================================
#  AIM: Attention-based Integration Module
# ============================================================

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.mlp(self.gap(x))
        return x * w.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size,
                                 padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class AIM(nn.Module):
    def __init__(self, enc_ch, dec_ch, branch_ch=64, reduction=16):
        super().__init__()
        fused_ch = enc_ch + dec_ch
        multi_ch = branch_ch * 3       # three dilated branches

        self.input_proj    = ConvBNReLU(fused_ch, branch_ch, k=1)
        # three parallel dilated convs
        self.d1 = ConvBNReLU(branch_ch, branch_ch, k=3, d=1)
        self.d2 = ConvBNReLU(branch_ch, branch_ch, k=3, d=2)
        self.d4 = ConvBNReLU(branch_ch, branch_ch, k=3, d=4)

        self.channel_attn  = ChannelAttention(multi_ch, reduction)
        self.spatial_attn  = SpatialAttention(kernel_size=7)
        self.output_conv   = nn.Conv2d(multi_ch, branch_ch, 1, bias=False)

    def forward(self, enc, dec):
        x       = torch.cat([enc, dec], dim=1)
        x       = self.input_proj(x)
        f_multi = torch.cat([self.d1(x), self.d2(x), self.d4(x)], dim=1)
        f_ca    = self.channel_attn(f_multi)
        f_sa    = self.spatial_attn(f_ca)
        return self.output_conv(f_sa)


# ============================================================
#  Decoder block 
# ============================================================

class DecoderBlock(nn.Module):
    def __init__(self, dec_ch, enc_ch, aim_ch, scale_factor=2):
        super().__init__()
        self.scale  = scale_factor
        self.proj   = ConvBNReLU(dec_ch, aim_ch, k=1)
        self.aim    = AIM(enc_ch=enc_ch, dec_ch=aim_ch, branch_ch=aim_ch)

    def forward(self, dec, enc_skip):
        dec = F.interpolate(dec, scale_factor=self.scale,
                            mode='bilinear', align_corners=False)
        dec = self.proj(dec)
        
        if dec.shape[2:] != enc_skip.shape[2:]:
            dec = F.interpolate(dec, size=enc_skip.shape[2:],
                                mode='bilinear', align_corners=False)
        return self.aim(enc_skip, dec)


# ============================================================
#  UNetOCT Model
# ============================================================

class UNetOCT(nn.Module):
    def __init__(self, in_ch=3, num_classes=4, pretrained=True):
        super().__init__()

        # ---- Encoder ----
        self.encoder = ResNet50Encoder(pretrained=pretrained)

        if in_ch != 3:
            old_conv = self.encoder.initial[0]     # original conv1
            new_conv = nn.Conv2d(in_ch, 64, kernel_size=7,
                                 stride=2, padding=3, bias=False)
            with torch.no_grad():
                new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True) \
                                     if in_ch == 1 \
                                     else old_conv.weight[:, :in_ch]
            self.encoder.initial[0] = new_conv

        self.msfr = MSFR(in_dim=2048, out_dim=256)

        # ---- Decoder ----
        self.dec4 = DecoderBlock(dec_ch=256,  enc_ch=1024, aim_ch=256)
        self.dec3 = DecoderBlock(dec_ch=256,  enc_ch=512,  aim_ch=128)
        self.dec2 = DecoderBlock(dec_ch=128,  enc_ch=256,  aim_ch=64)
        self.dec1 = DecoderBlock(dec_ch=64,   enc_ch=64,   aim_ch=32)

        self.head = nn.Sequential(
            ConvBNReLU(32, 32, k=3),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x):
        # ---------- Encoder ----------
        p0, p1, p2, p3, p4 = self.encoder(x)
        # p0: (B,   64, 128, 128)
        # p1: (B,  256,  64,  64)
        # p2: (B,  512,  32,  32)
        # p3: (B, 1024,  16,  16)
        # p4: (B, 2048,   8,   8)

        b = self.msfr(p4)      # (B, 256,  8,  8)

        # ---------- Decoder ----------
        d4 = self.dec4(b,  p3) # (B, 256, 16, 16)
        d3 = self.dec3(d4, p2) # (B, 128, 32, 32)
        d2 = self.dec2(d3, p1) # (B,  64, 64, 64)
        d1 = self.dec1(d2, p0) # (B,  32,128,128)

        out = self.head(d1)   
        out = F.interpolate(out, scale_factor=4,
                            mode='bilinear', align_corners=False)
        # print(f"output shape: {out.shape}")
        return out             # (B, num_classes, 512, 512)



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetOCT(in_ch=3, num_classes=4, pretrained=False).to(device)

    x   = torch.randn(2, 3, 512, 512, device=device)
    out = model(x)
    print(f"Input  : {x.shape}")
    print(f"Output : {out.shape}")    # expected: (2, 4, 512, 512)

    # n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Trainable parameters: {n_params / 1e6:.2f} M")