import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ---------------- Utility blocks ----------------
def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Conv -> BN -> SiLU (like many modern nets)."""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ---------------- Attention ----------------
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block (lightweight channel attention)."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        s = self.pool(x).view(b, c)
        a = self.fc(s).view(b, c, 1, 1)
        return x * a


# ---------------- Parallel Dilated Conv (fixed) ----------------
class ParallelDilatedConv(nn.Module):
    """Multiple parallel dilated convs -> concatenate -> 1x1 fuse."""
    def __init__(self, inplanes, planes, dilations=None, activation=None):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 3]#, 4] #, 6, 8, 9, 18, 24, 36]  # many scales (original code had similar)
        self.branches = nn.ModuleList()
        for d in dilations:
            # kernel_size=3, padding computed via dilation
            pad = d
            self.branches.append(nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=pad, dilation=d, bias=False),
                nn.BatchNorm2d(planes),
                activation if activation is not None else nn.ReLU(inplace=True)
            ))
        self.fuse = nn.Sequential(
            nn.Conv2d(planes * len(dilations), planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        outs = [b(x) for b in self.branches]
        out = torch.cat(outs, dim=1)
        out = self.fuse(out)
        return out


class conv_2nV1(nn.Module):
    def __init__(self, in_hc=64, in_lc=256, out_c=64, main=0):
        super(conv_2nV1, self).__init__()
        self.main = main
        mid_c = min(in_hc, in_lc)
        self.relu = nn.ReLU(True)
        self.h2l_pool = nn.AvgPool2d((2, 2), stride=2)
        self.l2h_up = nn.Upsample(scale_factor=2, mode="nearest")

        # stage 0
        self.h2h_0 = nn.Conv2d(in_hc, mid_c, 3, 1, 1)
        self.l2l_0 = nn.Conv2d(in_lc, mid_c, 3, 1, 1)
        self.bnh_0 = nn.BatchNorm2d(mid_c)
        self.bnl_0 = nn.BatchNorm2d(mid_c)

        # stage 1 (separable-style dilations)
        self.h2h_1 = nn.Sequential(
            nn.Conv2d(mid_c, mid_c, (3, 1), stride=1, padding=(2, 0), bias=True, dilation=(2, 1)),
            nn.Conv2d(mid_c, mid_c, (1, 3), stride=1, padding=(0, 2), bias=True, dilation=(1, 2))
        )
        self.h2l_1 = nn.Sequential(
            nn.Conv2d(mid_c, mid_c, (3, 1), stride=1, padding=(5, 0), bias=True, dilation=(5, 1)),
            nn.Conv2d(mid_c, mid_c, (1, 3), stride=1, padding=(0, 5), bias=True, dilation=(1, 5))
        )
        self.l2h_1 = nn.Sequential(
            nn.Conv2d(mid_c, mid_c, (3, 1), stride=1, padding=(9, 0), bias=True, dilation=(9, 1)),
            nn.Conv2d(mid_c, mid_c, (1, 3), stride=1, padding=(0, 9), bias=True, dilation=(1, 9))
        )
        self.l2l_1 = nn.Sequential(
            nn.Conv2d(mid_c, mid_c, (3, 1), stride=1, padding=(11, 0), bias=True, dilation=(11, 1)),
            nn.Conv2d(mid_c, mid_c, (1, 3), stride=1, padding=(0, 11), bias=True, dilation=(1, 11))
        )
        self.bnl_1 = nn.BatchNorm2d(mid_c)
        self.bnh_1 = nn.BatchNorm2d(mid_c)

        # stage 2+3 depending on `main`
        if self.main == 0:
            self.h2h_2 = nn.Sequential(
                nn.Conv2d(mid_c, mid_c, (3, 1), stride=1, padding=(2, 0), bias=True, dilation=(2, 1)),
                nn.Conv2d(mid_c, mid_c, (1, 3), stride=1, padding=(0, 2), bias=True, dilation=(1, 2))
            )
            self.l2h_2 = nn.Sequential(
                nn.Conv2d(mid_c, mid_c, (3, 1), stride=1, padding=(2, 0), bias=True, dilation=(2, 1)),
                nn.Conv2d(mid_c, mid_c, (1, 3), stride=1, padding=(0, 2), bias=True, dilation=(1, 2))
            )
            self.bnh_2 = nn.BatchNorm2d(mid_c)
            self.h2h_3 = nn.Conv2d(mid_c, out_c, 3, 1, 1)
            self.bnh_3 = nn.BatchNorm2d(out_c)
            self.identity = nn.Conv2d(in_hc, out_c, 1)
        else:
            self.h2l_2 = nn.Sequential(
                nn.Conv2d(mid_c, mid_c, (3, 1), stride=1, padding=(2, 0), bias=True, dilation=(2, 1)),
                nn.Conv2d(mid_c, mid_c, (1, 3), stride=1, padding=(0, 2), bias=True, dilation=(1, 2))
            )
            self.l2l_2 = nn.Sequential(
                nn.Conv2d(mid_c, mid_c, (3, 1), stride=1, padding=(2, 0), bias=True, dilation=(2, 1)),
                nn.Conv2d(mid_c, mid_c, (1, 3), stride=1, padding=(0, 2), bias=True, dilation=(1, 2))
            )
            self.bnl_2 = nn.BatchNorm2d(mid_c)
            self.l2l_3 = nn.Conv2d(mid_c, out_c, 3, 1, 1)
            self.bnl_3 = nn.BatchNorm2d(out_c)
            self.identity = nn.Conv2d(in_lc, out_c, 1)
  
    def forward(self, in_h, in_l):

        def _upsample_to(x, target):
            return F.interpolate(x, size=target.shape[2:], mode="bilinear", align_corners=False)

        def _downsample_to(x, target):
            return F.interpolate(x, size=target.shape[2:], mode="bilinear", align_corners=False)

        # Stage 0
        h = self.relu(self.bnh_0(self.h2h_0(in_h)))
        l = self.relu(self.bnl_0(self.l2l_0(in_l)))

        # Stage 1
        h2h = self.h2h_1(h)
        h2l = self.h2l_1(_downsample_to(h, l))    # h down to l size
        l2h = self.l2h_1(_upsample_to(l, h))      # l up to h size
        l2l = self.l2l_1(l)

        h = self.relu(self.bnh_1(h2h + l2h))
        l = self.relu(self.bnl_1(l2l + h2l))

        # Stage 2 / 3 depending on main
        if self.main == 0:
            h2h = self.h2h_2(h)
            l2h = self.l2h_2(_upsample_to(l, h2h))
            h_fuse = self.relu(self.bnh_2(h2h + l2h))
            out = self.relu(self.bnh_3(self.h2h_3(h_fuse)) + self.identity(in_h))
        else:
            h2l = self.h2l_2(_downsample_to(h, l))
            l2l = self.l2l_2(l)
            l_fuse = self.relu(self.bnl_2(h2l + l2l))
            out = self.relu(self.bnl_3(self.l2l_3(l_fuse)) + self.identity(in_l))

        return out



class conv_3nV1(nn.Module):
    def __init__(self, in_hc=64, in_mc=256, in_lc=512, out_c=64):
        super(conv_3nV1, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.downsample = nn.AvgPool2d((2, 2), stride=2)

        mid_c = min(in_hc, in_mc, in_lc)
        self.relu = nn.ReLU(True)

        # stage 0
        self.h2h_0 = nn.Conv2d(in_hc, mid_c, 3, 1, 1)
        self.m2m_0 = nn.Conv2d(in_mc, mid_c, 3, 1, 1)
        self.l2l_0 = nn.Conv2d(in_lc, mid_c, 3, 1, 1)
        self.bnh_0 = nn.BatchNorm2d(mid_c)
        self.bnm_0 = nn.BatchNorm2d(mid_c)
        self.bnl_0 = nn.BatchNorm2d(mid_c)

        # stage 1 (dilated mixture)
        self.h2h_1 = nn.Conv2d(mid_c, mid_c, 3, stride=1, padding=2, bias=True, dilation=2)
        self.h2m_1 = nn.Conv2d(mid_c, mid_c, 3, stride=1, padding=5, bias=True, dilation=5)
        self.m2h_1 = nn.Conv2d(mid_c, mid_c, 3, stride=1, padding=9, bias=True, dilation=9)
        self.m2m_1 = nn.Conv2d(mid_c, mid_c, 5, stride=1, padding=6, bias=True, dilation=3)
        self.m2l_1 = nn.Conv2d(mid_c, mid_c, 7, stride=1, padding=15, bias=True, dilation=5)
        self.l2m_1 = nn.Conv2d(mid_c, mid_c, 9, stride=1, padding=28, bias=True, dilation=7)
        self.l2l_1 = nn.Conv2d(mid_c, mid_c, 11, stride=1, padding=45, bias=True, dilation=9)
        self.bnh_1 = nn.BatchNorm2d(mid_c)
        self.bnm_1 = nn.BatchNorm2d(mid_c)
        self.bnl_1 = nn.BatchNorm2d(mid_c)

        # stage 2
        self.h2m_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.l2m_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.m2m_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.bnm_2 = nn.BatchNorm2d(mid_c)

        # stage 3
        self.m2m_3 = nn.Conv2d(mid_c, out_c, 3, 1, 1)
        self.bnm_3 = nn.BatchNorm2d(out_c)

        self.identity = nn.Conv2d(in_mc, out_c, 1)
    # replace self.upsample or self.downsample in conv_3nV1 with:
    def _upsample_to(x, target):
        return F.interpolate(x, size=target.shape[2:], mode="bilinear", align_corners=False)

    def _downsample_to(x, target):
        return F.interpolate(x, size=target.shape[2:], mode="bilinear", align_corners=False)


    def forward(self, in_h, in_m, in_l):
        h = self.relu(self.bnh_0(self.h2h_0(in_h)))
        m = self.relu(self.bnm_0(self.m2m_0(in_m)))
        l = self.relu(self.bnl_0(self.l2l_0(in_l)))

        # Upsample/downsample dynamically to match
        m2h = F.interpolate(m, size=h.shape[2:], mode="bilinear", align_corners=False)
        l2m = F.interpolate(l, size=m.shape[2:], mode="bilinear", align_corners=False)
        h2m = F.interpolate(h, size=m.shape[2:], mode="bilinear", align_corners=False)
        m2l = F.interpolate(m, size=l.shape[2:], mode="bilinear", align_corners=False)

        h = self.relu(self.bnh_1(self.h2h_1(h) + m2h))
        m = self.relu(self.bnm_1(h2m + self.m2m_1(m) + l2m))
        l = self.relu(self.bnl_1(m2l + self.l2l_1(l)))

        # final fusion
        h2m = F.interpolate(h, size=m.shape[2:], mode="bilinear", align_corners=False)
        l2m = F.interpolate(l, size=m.shape[2:], mode="bilinear", align_corners=False)
        m = self.relu(self.bnm_2(h2m + self.m2m_2(m) + l2m))

        out = self.relu(self.bnm_3(self.m2m_3(m)) + self.identity(in_m))
        return out


# ---------------- AIM block (kept similar, but use new ParallelDilatedConv) ----------------
class AIM(nn.Module):
    def __init__(self, iC_list, oC_list):
        super(AIM, self).__init__()
        ic0, ic1, ic2, ic3, ic4 = iC_list
        oc0, oc1, oc2, oc3, oc4 = oC_list

        self.conv0 = conv_2nV1(in_hc=ic0, in_lc=ic1, out_c=oc0, main=0)
        self.conv1 = conv_3nV1(in_hc=ic0, in_mc=ic1, in_lc=ic2, out_c=oc1)
        self.aspconv1 = ParallelDilatedConv(oc1, oc1)

        self.conv2 = conv_3nV1(in_hc=ic1, in_mc=ic2, in_lc=ic3, out_c=oc2)
        self.aspconv2 = ParallelDilatedConv(oc2, oc2)

        self.conv3 = conv_3nV1(in_hc=ic2, in_mc=ic3, in_lc=ic4, out_c=oc3)
        self.aspconv3 = ParallelDilatedConv(oc3, oc3)

        self.conv4 = conv_2nV1(in_hc=ic3, in_lc=ic4, out_c=oc4, main=1)
        self.aspconv4 = ParallelDilatedConv(oc4, oc4)

    def forward(self, *xs):
        # xs: p0 (initial), p1, p2, p3, p4
        outconv4 = self.aspconv4(self.conv4(xs[3], xs[4]))
        outconv3 = self.aspconv3(self.conv3(xs[2], xs[3], outconv4))
        outconv2 = self.aspconv2(self.conv2(xs[1], xs[2], outconv3))
        outconv1 = self.aspconv1(self.conv1(xs[0], xs[1], outconv2))
        outconv0 = self.conv0(xs[0], outconv1)

      
        return [outconv0, outconv1, outconv2, outconv3, outconv4]


# ---------------- SPPF & ASPP (clean) ----------------
class SPPF(nn.Module):
    """SPPF as in YOLOv5."""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        y0 = self.cv1(x)
        y1 = self.m(y0)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([y0, y1, y2, y3], dim=1))


class ASPP(nn.Module):
    """ASPP close to DeepLab style — keeps input dims."""
    def __init__(self, in_dim, out_dim=None, rates=(1, 6, 12, 18)):
        super().__init__()
        if out_dim is None:
            out_dim = in_dim
        mid = in_dim // 2
        self.branches = nn.ModuleList()
        # 1x1
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_dim, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True)
        ))
        # dilated branches
        for r in rates:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_dim, mid, kernel_size=3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(mid), nn.ReLU(inplace=True)
            ))
        # image pool
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_dim, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(mid * (2 + len(rates)), out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        res = [b(x) for b in self.branches]
        img = self.image_pool(x)
        img = F.interpolate(img, size=x.shape[2:], mode='bilinear', align_corners=False)
        res.append(img)
        return self.fuse(torch.cat(res, dim=1))


# ---------------- Decoder Block with SE + dropout ----------------
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_se=True, dropout_prob=0.15):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_prob) if dropout_prob and dropout_prob > 0 else nn.Identity()
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

    def forward(self, x, skip):
        # x: decoder input (smaller spatial), skip: skip from encoder / AIM
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.se(x)
        return x


# ---------------- ResNet50 Encoder ----------------
class ResNet50Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # respect pretrained flag
        if pretrained:
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            resnet = models.resnet50(weights=None)
        self.resnet = resnet
        self.initial = nn.Sequential(
            resnet.conv1,  # /2
            resnet.bn1,
            resnet.relu,
            resnet.maxpool  # /2 -> total /4
        )
        self.layer1 = resnet.layer1  # /4
        self.layer2 = resnet.layer2  # /8
        self.layer3 = resnet.layer3  # /16
        self.layer4 = resnet.layer4  # /32

    def forward(self, x):
        l0 = self.initial(x)   # after maxpool -> /4
        l1 = self.layer1(l0)   # /4
        l2 = self.layer2(l1)   # /8
        l3 = self.layer3(l2)   # /16
        l4 = self.layer4(l3)   # /32
        return l0, l1, l2, l3, l4


# ---------------- Full Model ----------------
class UNET_OCT(nn.Module):
    def __init__(self, num_classes=4, pretrained=True, name="UOCNet"):
        super().__init__()
        self.name = name
        self.encoder = ResNet50Encoder(pretrained=pretrained)

        # context modules
        self.sppf = SPPF(2048, 2048, k=5)
        self.aspp = ASPP(2048, 2048, rates=(1, 2, 3, 5))  # rates=(1, 6, 12, 18) (1, 3, 5, 7)

        # AIM (multi-scale fusion)
        self.aim = AIM(iC_list=(64, 256, 512, 1024, 2048), oC_list=(64, 256, 512, 1024, 2048))

        # decoders (in_channels carefully set)
        # decoder4 expects concatenation of p4, sppf(p4), aim_l4 (each 2048)
        self.decoder4 = DecoderBlock(in_channels=2048 * 3, skip_channels=1024, out_channels=512, use_se=True, dropout_prob=0.2)
        self.decoder3 = DecoderBlock(in_channels=512, skip_channels=512, out_channels=256, use_se=True, dropout_prob=0.15)
        self.decoder2 = DecoderBlock(in_channels=256, skip_channels=256, out_channels=128, use_se=True, dropout_prob=0.1)

        self.upconv = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        p0, p1, p2, p3, p4 = self.encoder(x)  # p0:/4, p1:/4, p2:/8, p3:/16, p4:/32
        # print('Encoder:', p0.shape, p1.shape, p2.shape, p3.shape, p4.shape)
        # AIM multi-scale fusion outputs (same channel dims as inputs)
        l0, l1, l2, l3, l4 = self.aim(p0, p1, p2, p3, p4)
        # print('AIM:', l0.shape, l1.shape, l2.shape, l3.shape, l4.shape)
        # Context modules on deepest features
        # sppf_out = self.sppf(p4)        # 2048
        # aspp_out = self.aspp(l4)        # 2048 (from AIM)
        sppf_out = self.sppf(l4)        # 2048
        aspp_out = self.aspp(p4)  

        # fuse deep features (concatenate 3x2048 => 6144) to feed decoder4
        x = torch.cat([p4, sppf_out, aspp_out], dim=1)  # 2048*3 -> decoder4 in_channels

        # decoder stages with corresponding AIM outputs as skips (l3, l2, l1)
        x = self.decoder4(x, l3)  # output spatial = l3 (/16)
        x = self.decoder3(x, l2)  # -> /8
        x = self.decoder2(x, l1)  # -> /4

        # up to /2 via transposed conv and then upsample to original input resolution
        x = self.upconv(x)  # /2
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # /1 (original size)
        x = self.final_conv(x)
        return x


# ---------------- Quick smoke test ----------------
if __name__ == "__main__":
    model = UNET_OCT(num_classes=4, pretrained=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dummy = torch.randn(2, 3, 256, 256).to(device)
    out = model(dummy)
    print("Output shape:", out.shape)  # expect (2, num_classes, 256, 256)
