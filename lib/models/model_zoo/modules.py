import torch
from torch.nn.modules.activation import PReLU
from torch.nn.modules.batchnorm import BatchNorm2d
import torchvision
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Conv2d, Parameter, Softmax
from typing import Optional


class DACBlock(nn.Module):
    def __init__(self, in_channels):
        super(DACBlock, self).__init__()
        
        # First branch: simple 3x3 conv (rate=1)
        self.branch1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1)
        
        # Second branch: 3x3 conv (rate=3) -> 1x1 conv
        self.branch2_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=3, dilation=3)
        self.branch2_2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Third branch: 3x3 conv (rate=1) -> 3x3 conv (rate=3) -> 1x1 conv
        self.branch3_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1)
        self.branch3_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=3, dilation=3)
        self.branch3_3 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Fourth branch: 3x3 conv (rate=1) -> 3x3 conv (rate=3) -> 3x3 conv (rate=5) -> 1x1 conv
        self.branch4_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1)
        self.branch4_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=3, dilation=3)
        self.branch4_3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=5, dilation=5)
        self.branch4_4 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Branch outputs
        b1 = self.relu(self.branch1(x))

        b2 = self.relu(self.branch2_1(x))
        b2 = self.relu(self.branch2_2(b2))

        b3 = self.relu(self.branch3_1(x))
        b3 = self.relu(self.branch3_2(b3))
        b3 = self.relu(self.branch3_3(b3))

        b4 = self.relu(self.branch4_1(x))
        b4 = self.relu(self.branch4_2(b4))
        b4 = self.relu(self.branch4_3(b4))
        b4 = self.relu(self.branch4_4(b4))

        # Sum all paths + original feature map
        out = x + b1 + b2 + b3 + b4
        return out


class DACBlock2(nn.Module):
    def __init__(self, in_channels):
        super(DACBlock2, self).__init__()
        
        # Branch 1: 3x3 conv (rate=1)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: 3x3 (rate=3) -> 1x1
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 3: 3x3 (rate=1) -> 3x3 (rate=3) -> 1x1
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 4: 3x3 (rate=1) -> 3x3 (rate=3) -> 3x3 (rate=5) -> 1x1
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=5, dilation=5),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        
        # Sum of all branches + original input
        out = x + out1 + out2 + out3 + out4
        return out


class UAFM(nn.Module):
    """
    Unified Attention Fusion Module, which uses mean and max values across the spatial dimensions.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        up_factor: int,
        #upsample_mode: Union[UpsampleMode, str] = UpsampleMode.BILINEAR,
        align_corners: bool = False,
    ):
        """
        :params in_channels: num_channels of input feature map.
        :param skip_channels: num_channels of skip connection feature map.
        :param out_channels: num out channels after features fusion.
        :param up_factor: upsample scale factor of the input feature map.
        :param upsample_mode: see UpsampleMode for valid options.
        """
        super().__init__()
        self.conv_atten = nn.Sequential(
             BasicConv(4, 2, kernel_size=3, padding=1, bias=False),  BasicConv(2, 1, kernel_size=3, padding=1, bias=False, use_activation=False)
        )

        self.proj_skip = nn.Identity() if skip_channels == in_channels else  BasicConv(skip_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.up_x = nn.Identity() if up_factor == 1 else nn.Upsample(scale_factor=up_factor, mode="nearest")
        self.conv_out =  BasicConv(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x, skip):
        """
        :param x: input feature map to upsample before fusion.
        :param skip: skip connection feature map.
        """
        x = self.up_x(x)
        skip = self.proj_skip(skip)

        atten = torch.cat([*self._avg_max_spatial_reduce(x, use_concat=False), *self._avg_max_spatial_reduce(skip, use_concat=False)], dim=1)
        atten = self.conv_atten(atten)
        atten = torch.sigmoid(atten)

        out = x * atten + skip * (1 - atten)
        out = self.conv_out(out)
        return out



"""Parallel Factorize Convolution"""
# Triplat Attention 
#### we are using this conve block as basic building block for all modules to make it symetric.
# #### or we can use directly nn.conv2d
#         self.conv3x1_22 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(2, 0), bias=True, dilation=(2, 1))
#         self.conv1x3_22 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 2), bias=True, dilation=(1, 2))

#         self.conv3x1_25 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(5, 0), bias=True, dilation=(5, 1))
#         self.conv1x3_25 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 5), bias=True, dilation=(1, 5))

#         self.conv3x1_29 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(9, 0), bias=True, dilation=(9, 1))
#         self.conv1x3_29 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 9), bias=True, dilation=(1, 9))
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
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

        # stage 1
        # self.h2h_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        # self.h2l_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        # self.l2h_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        # self.l2l_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.h2h_1 = nn.Sequential(nn.Conv2d(mid_c, mid_c, (3, 1), stride=1, padding=(2, 0), bias=True, dilation=(2, 1)),
                                   nn.Conv2d(mid_c, mid_c, (1, 3), stride=1, padding=(0, 2), bias=True, dilation=(1, 2)))
        self.h2l_1 = nn.Sequential(nn.Conv2d(mid_c, mid_c, (3, 1), stride=1, padding=(5, 0), bias=True, dilation=(5, 1)),
                                   nn.Conv2d(mid_c, mid_c, (1, 3), stride=1, padding=(0, 5), bias=True, dilation=(1, 5)))
        self.l2h_1 = nn.Sequential(nn.Conv2d(mid_c, mid_c, (3, 1), stride=1, padding=(9, 0), bias=True, dilation=(9, 1)),
                                   nn.Conv2d(mid_c, mid_c, (1, 3), stride=1, padding=(0, 9), bias=True, dilation=(1, 9)))
        self.l2l_1 = nn.Sequential(nn.Conv2d(mid_c, mid_c, (3, 1), stride=1, padding=(11, 0), bias=True, dilation=(11, 1)),
                                   nn.Conv2d(mid_c, mid_c, (1, 3), stride=1, padding=(0, 11), bias=True, dilation=(1, 11)))
        self.bnl_1 = nn.BatchNorm2d(mid_c)
        self.bnh_1 = nn.BatchNorm2d(mid_c)

        if self.main == 0:
            # stage 2
            self.h2h_2 = nn.Sequential(nn.Conv2d(mid_c, mid_c, (3, 1), stride=1, padding=(2, 0), bias=True, dilation=(2, 1)),
                                   nn.Conv2d(mid_c, mid_c, (1, 3), stride=1, padding=(0, 2), bias=True, dilation=(1, 2)))
            self.l2h_2 = nn.Sequential(nn.Conv2d(mid_c, mid_c, (3, 1), stride=1, padding=(2, 0), bias=True, dilation=(2, 1)),
                                   nn.Conv2d(mid_c, mid_c, (1, 3), stride=1, padding=(0, 2), bias=True, dilation=(1, 2)))
            self.bnh_2 = nn.BatchNorm2d(mid_c)

            # stage 3
            self.h2h_3 = nn.Conv2d(mid_c, out_c, 3, 1, 1)
            self.bnh_3 = nn.BatchNorm2d(out_c)

            self.identity = nn.Conv2d(in_hc, out_c, 1)

        elif self.main == 1:
            # stage 2
            self.h2l_2 = nn.Sequential(nn.Conv2d(mid_c, mid_c, (3, 1), stride=1, padding=(2, 0), bias=True, dilation=(2, 1)),
                                   nn.Conv2d(mid_c, mid_c, (1, 3), stride=1, padding=(0, 2), bias=True, dilation=(1, 2)))
            self.l2l_2 = nn.Sequential(nn.Conv2d(mid_c, mid_c, (3, 1), stride=1, padding=(2, 0), bias=True, dilation=(2, 1)),
                                   nn.Conv2d(mid_c, mid_c, (1, 3), stride=1, padding=(0, 2), bias=True, dilation=(1, 2)))
            self.bnl_2 = nn.BatchNorm2d(mid_c)

            # stage 3
            self.l2l_3 = nn.Conv2d(mid_c, out_c, 3, 1, 1)
            self.bnl_3 = nn.BatchNorm2d(out_c)

            self.identity = nn.Conv2d(in_lc, out_c, 1)

        else:
            raise NotImplementedError

    def forward(self, in_h, in_l):
        # stage 0
        h = self.relu(self.bnh_0(self.h2h_0(in_h)))
        l = self.relu(self.bnl_0(self.l2l_0(in_l)))

        # stage 1
        h2h = self.h2h_1(h)
        h2l = self.h2l_1(self.h2l_pool(h))
        l2l = self.l2l_1(l)
        l2h = self.l2h_1(self.l2h_up(l))
        h = self.relu(self.bnh_1(h2h + l2h))
        l = self.relu(self.bnl_1(l2l + h2l))

        if self.main == 0:
            # stage 2
            h2h = self.h2h_2(h)
            l2h = self.l2h_2(self.l2h_up(l))
            h_fuse = self.relu(self.bnh_2(h2h + l2h))

            # stage 3
            out = self.relu(self.bnh_3(self.h2h_3(h_fuse)) + self.identity(in_h))
            # 这里使用的不是in_h，而是h
        elif self.main == 1:
            # stage 2
            h2l = self.h2l_2(self.h2l_pool(h))
            l2l = self.l2l_2(l)
            l_fuse = self.relu(self.bnl_2(h2l + l2l))

            # stage 3
            out = self.relu(self.bnl_3(self.l2l_3(l_fuse)) + self.identity(in_l))
        else:
            raise NotImplementedError

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

        # stage 1
        #mid_c, mid_c, (3, 1), stride=1, padding=(2, 0), bias=True, dilation=(2, 1)
        # self.h2h_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        # self.h2m_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        # self.m2h_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        # self.m2m_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        # self.m2l_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        # self.l2m_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        # self.l2l_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
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

    def forward(self, in_h, in_m, in_l):
        # stage 0
        h = self.relu(self.bnh_0(self.h2h_0(in_h)))
        m = self.relu(self.bnm_0(self.m2m_0(in_m)))
        l = self.relu(self.bnl_0(self.l2l_0(in_l)))

        # stage 1
        h2h = self.h2h_1(h)
        m2h = self.m2h_1(self.upsample(m))

        h2m = self.h2m_1(self.downsample(h))
        m2m = self.m2m_1(m)
        l2m = self.l2m_1(self.upsample(l))

        m2l = self.m2l_1(self.downsample(m))
        l2l = self.l2l_1(l)

        h = self.relu(self.bnh_1(h2h + m2h))
        m = self.relu(self.bnm_1(h2m + m2m + l2m))
        l = self.relu(self.bnl_1(m2l + l2l))

        # stage 2
        h2m = self.h2m_2(self.downsample(h))
        m2m = self.m2m_2(m)
        l2m = self.l2m_2(self.upsample(l))
        m = self.relu(self.bnm_2(h2m + m2m + l2m))

        # stage 3
        out = self.relu(self.bnm_3(self.m2m_3(m)) + self.identity(in_m))
        return out


# class AIM(nn.Module):
#     def __init__(self, iC_list, oC_list):
#         super(AIM, self).__init__()
#         ic0, ic1, ic2, ic3, ic4 = iC_list
#         oc0, oc1, oc2, oc3, oc4 = oC_list
#         self.conv0 = conv_2nV1(in_hc=ic0, in_lc=ic1, out_c=oc0, main=0)
#         self.conv1 = conv_3nV1(in_hc=ic0, in_mc=ic1, in_lc=ic2, out_c=oc1)
#         #self.aspconv1 = ASPP(oc4,oc4)
#         self.conv2 = conv_3nV1(in_hc=ic1, in_mc=ic2, in_lc=ic3, out_c=oc2)
#         #self.aspconv2 = ASPP(oc2,oc2)
#         self.conv3 = conv_3nV1(in_hc=ic2, in_mc=ic3, in_lc=ic4, out_c=oc3)
#         #self.aspconv3 = ASPP(oc3,oc3)
#         self.conv4 = conv_2nV1(in_hc=ic3, in_lc=ic4, out_c=oc4, main=1)
#         #self.aspconv4 = ASPP(oc4,oc4)
#     def forward(self, *xs):
#         # in_data_2, in_data_4, in_data_8, in_data_16, in_data_32
#         out_xs = []
#         out_xs.append(self.conv0(xs[0], xs[1]))
#         out_xs.append(self.conv1(xs[0], xs[1], xs[2]))
#         out_xs.append(self.conv2(xs[1], xs[2], xs[3]))
#         out_xs.append(self.conv3(xs[2], xs[3], xs[4]))
#         out_xs.append(self.conv4(xs[3], xs[4]))

#         return out_xs

class AIM(nn.Module):
    def __init__(self, iC_list, oC_list):
        super(AIM, self).__init__()
        ic0, ic1, ic2, ic3, ic4 = iC_list
        oc0, oc1, oc2, oc3, oc4 = oC_list
        self.conv0 = conv_2nV1(in_hc=ic0, in_lc=ic1, out_c=oc0, main=0)
        self.conv1 = conv_3nV1(in_hc=ic0, in_mc=ic1, in_lc=ic2, out_c=oc1)
        # self.aspconv1 = ASPP(oc1,oc1)
        self.aspconv1 = ParallelDilatedConv(oc1,oc1)
        # self.aspconv1 = FPEBlock(oc1,oc1, [1,2,4,8])

        self.conv2 = conv_3nV1(in_hc=ic1, in_mc=ic2, in_lc=ic3, out_c=oc2)
        # self.aspconv2 = ASPP(oc2,oc2)
        self.aspconv2 = ParallelDilatedConv(oc2,oc2)
        # self.aspconv2 = FPEBlock(oc2,oc2, [1,2,4,8])

        self.conv3 = conv_3nV1(in_hc=ic2, in_mc=ic3, in_lc=ic4, out_c=oc3)
        # self.aspconv3 = ASPP(oc3,oc3)
        self.aspconv3 = ParallelDilatedConv(oc3,oc3)
        # self.aspconv3 = FPEBlock(oc3,oc3, [1,2,4,8])

        self.conv4 = conv_2nV1(in_hc=ic3, in_lc=ic4, out_c=oc4, main=1)
        # self.aspconv4 = ASPP(oc4,oc4)
        self.aspconv4 = ParallelDilatedConv(oc4,oc4)
        # self.aspconv4 = FPEBlock(oc4,oc4, [1,2,4,8])

    def forward(self, *xs):
        # in_data_2, in_data_4, in_data_8, in_data_16, in_data_32
        out_xs = []
                                
        outconv4 = self.aspconv4(self.conv4(xs[3], xs[4]))
        

        outconv3 =self.aspconv3(self.conv3(xs[2], xs[3], outconv4))
        

        outconv2 =self.aspconv2(self.conv2(xs[1], xs[2], outconv3))
        

        outconv1 =self.aspconv1(self.conv1(xs[0], xs[1], outconv2))
        
        outconv0 = self.conv0(xs[0], outconv1)
        out_xs.append(outconv0)
        out_xs.append(outconv1)
        out_xs.append(outconv2)
        out_xs.append(outconv3)
        out_xs.append(outconv4)
        return out_xs










class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(TripletAttention, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.SpatialGate(x)
            x_out = (1 / 3) * (x_out + x_out11 + x_out21)
        else:
            x_out = (1 / 2) * (x_out11 + x_out21)
        return x_out
###################################################################
# Non Local Block
# https://arxiv.org/abs/1711.07971
###################################################################

class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlock, self).__init__()

        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                          kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                               kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

    def forward(self, x):

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z
class PFCU(nn.Module):
    def __init__(self, chann):
        """
        Parallel Factorized Convolution Unit

        """

        super(PFCU, self).__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_22 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(8, 0), bias=True, dilation=(8, 1))
        self.conv1x3_22 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 8), bias=True, dilation=(1, 8))

        self.conv3x1_25 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(16, 0), bias=True, dilation=(16, 1))
        self.conv1x3_25 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 16), bias=True, dilation=(1, 16))

        self.conv3x1_29 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(24, 0), bias=True, dilation=(24, 1))
        self.conv1x3_29 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 24), bias=True, dilation=(1, 24))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(0.3)
        self.output = TripletAttention(chann)

    def forward(self, input):
        residual = input
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output2 = self.conv3x1_22(output)
        output2 = F.relu(output2)
        output2 = self.conv1x3_22(output2)
        output2 = self.bn2(output2)
        if (self.dropout.p != 0):
            output2 = self.dropout(output2)

        output5 = self.conv3x1_25(output)
        output5 = F.relu(output5)
        output5 = self.conv1x3_25(output5)
        output5 = self.bn2(output5)
        if (self.dropout.p != 0):
            output5 = self.dropout(output5)

        output9 = self.conv3x1_29(output)
        output9 = F.relu(output9)
        output9 = self.conv1x3_29(output9)
        output9 = self.bn2(output9)
        if (self.dropout.p != 0):
            output9 = self.dropout(output9)
        out = residual + output2 + output5 + output9
        out = self.output(out)
        return F.relu(out, inplace=True)
"""FSSNet: Fast Semantic Segmentation for Scene Perception"""        
NON_LINEARITY = {
    'ReLU': nn.ReLU(inplace=True),
    'PReLU': nn.PReLU(),
    'ReLu6': nn.ReLU6(inplace=True),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid()
}

class Factorized_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 dropout_prob=0., bias=False, non_linear='ReLU'):
        super(Factorized_Block, self).__init__()
        self.relu = NON_LINEARITY[non_linear]
        self.internal_channels = in_channels // 4
        self.compress_conv1 = nn.Conv2d(in_channels, self.internal_channels, 1, padding=0, bias=bias)
        self.conv1_bn = nn.BatchNorm2d(self.internal_channels)
        # here is relu
        self.conv2_1 = nn.Conv2d(self.internal_channels, self.internal_channels, (kernel_size, 1), stride=(stride, 1),
                                 padding=(int((kernel_size - 1) / 2 * dilation), 0), dilation=(dilation, 1), bias=bias)
        self.conv2_1_bn = nn.BatchNorm2d(self.internal_channels)
        self.conv2_2 = nn.Conv2d(self.internal_channels, self.internal_channels, (1, kernel_size), stride=(1, stride),
                                 padding=(0, int((kernel_size - 1) / 2 * dilation)), dilation=(1, dilation), bias=bias)
        self.conv2_2_bn = nn.BatchNorm2d(self.internal_channels)
        # here is relu
        self.extend_conv3 = nn.Conv2d(self.internal_channels, out_channels, 1, padding=0, bias=bias)

        self.conv3_bn = nn.BatchNorm2d(out_channels)
        self.regul = nn.Dropout2d(p=dropout_prob)

    def forward(self, x):
        residual = x
        main = self.relu((self.conv1_bn(self.compress_conv1(x))))
        main = self.relu(self.conv2_1_bn(self.conv2_1(main)))
        main = self.relu(self.conv2_2_bn(self.conv2_2(main)))

        main = self.conv3_bn(self.extend_conv3(main))
        main = self.regul(main)
        out = self.relu((torch.add(residual, main)))
        return out
"""FPENet:Feature Pyramid Encoding Network for Real-time Semantic Segmentation"""
def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, groups=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, dilation=dilation, groups=groups,bias=bias)

"""FPENet:Feature Pyramid Encoding Network for Real-time Semantic Segmentation"""
def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

"""FPENet:Feature Pyramid Encoding Network for Real-time Semantic Segmentation"""
class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x

"""FPENet:Feature Pyramid Encoding Network for Real-time Semantic Segmentation"""
class FPEBlock(nn.Module):

    def __init__(self, inplanes, outplanes, dilat, downsample=None, stride=1, t=1, scales=4, se=True, norm_layer=None):
        super(FPEBlock, self).__init__()
        if inplanes % scales != 0:
            raise ValueError('Planes must be divisible by scales')
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        bottleneck_planes = inplanes * t
        self.conv1 = conv1x1(inplanes, bottleneck_planes, stride)
        self.bn1 = norm_layer(bottleneck_planes)
        self.conv2 = nn.ModuleList([conv3x3(bottleneck_planes // scales, bottleneck_planes // scales,
                                            groups=(bottleneck_planes // scales),dilation=dilat[i],
                                            padding=1*dilat[i]) for i in range(scales)])
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_planes // scales) for _ in range(scales)])
        self.conv3 = conv1x1(bottleneck_planes, outplanes)
        self.bn3 = norm_layer(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.se = TripletAttention(outplanes) if se else None
        self.downsample = downsample
        self.stride = stride
        self.scales = scales

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        xs = torch.chunk(out, self.scales, 1)
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(self.relu(self.bn2[s](self.conv2[s](xs[s]))))
            else:
                ys.append(self.relu(self.bn2[s](self.conv2[s](xs[s] + ys[-1]))))
        out = torch.cat(ys, 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out

###CVPR2017 Pyramid Scene Parsing Network
class PPM(nn.Module): # pspnet
    def __init__(self, down_dim):
        super(PPM, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(2048,down_dim , 3,padding=1),nn.BatchNorm2d(down_dim),
             nn.PReLU())

        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),nn.Conv2d(down_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(2, 2)), nn.Conv2d(down_dim, down_dim, kernel_size=1),
            nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(3, 3)),nn.Conv2d(down_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(6, 6)), nn.Conv2d(down_dim, down_dim, kernel_size=1),
            nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(4 * down_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

    def forward(self, x):
        x = self.down_conv(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv1_up = F.upsample(conv1, size=x.size()[2:], mode='bilinear')
        conv2_up = F.upsample(conv2, size=x.size()[2:], mode='bilinear')
        conv3_up = F.upsample(conv3, size=x.size()[2:], mode='bilinear')
        conv4_up = F.upsample(conv4, size=x.size()[2:], mode='bilinear')

        return self.fuse(torch.cat((conv1_up, conv2_up, conv3_up, conv4_up), 1))
#==============================FPE============================
"""Replace depwise convolution"""
"""FPENet:Feature Pyramid Encoding Network for Real-time Semantic Segmentation"""
def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, groups=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, dilation=dilation, groups=groups,bias=bias)

"""FPENet:Feature Pyramid Encoding Network for Real-time Semantic Segmentation"""
def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

"""FPENet:Feature Pyramid Encoding Network for Real-time Semantic Segmentation"""
class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x

"""FPENet:Feature Pyramid Encoding Network for Real-time Semantic Segmentation"""
class FPEBlock(nn.Module):

    def __init__(self, inplanes, outplanes, dilat, downsample=None, stride=1, t=1, scales=4, se=True, norm_layer=None):
        super(FPEBlock, self).__init__()
        if inplanes % scales != 0:
            raise ValueError('Planes must be divisible by scales')
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        bottleneck_planes = inplanes * t
        self.conv1 = conv1x1(inplanes, bottleneck_planes, stride)
        self.bn1 = norm_layer(bottleneck_planes)
        self.conv2 = nn.ModuleList([conv3x3(bottleneck_planes // scales, bottleneck_planes // scales,
                                            groups=(bottleneck_planes // scales),dilation=dilat[i],
                                            padding=1*dilat[i]) for i in range(scales)])
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_planes // scales) for _ in range(scales)])
        self.conv3 = conv1x1(bottleneck_planes, outplanes)
        self.bn3 = norm_layer(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEModule(outplanes) if se else None
        self.downsample = downsample
        self.stride = stride
        self.scales = scales

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        xs = torch.chunk(out, self.scales, 1)
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(self.relu(self.bn2[s](self.conv2[s](xs[s]))))
            else:
                ys.append(self.relu(self.bn2[s](self.conv2[s](xs[s] + ys[-1]))))
        out = torch.cat(ys, 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out
#====================================PPDConv=================================================
"""from Speeding up Semantic Segementation for Autonomous Driving,,, can be placed at the end of encoder as global context selection"""
class ParallelDilatedConv(nn.Module):
    def __init__(self, inplanes, planes):
        super(ParallelDilatedConv, self).__init__()
        self.dilated_conv_1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1) 
        self.dilated_conv_2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dilated_conv_3 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=3, dilation=3)
        self.dilated_conv_4 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=4, dilation=4)
        #--------------------------
        self.dilated_conv_33 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=6, dilation=6)
        self.dilated_conv_44 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=8, dilation=8)
# ------------------------------
        self.dilated_conv_5 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=9, dilation=9) 
        self.dilated_conv_6 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=18, dilation=18)
        self.dilated_conv_7 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=24, dilation=24)
        self.dilated_conv_8 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=36, dilation=36)

        self.relu1 = nn.ELU(inplace=True)
        self.relu2 = nn.ELU(inplace=True)
        self.relu3 = nn.ELU(inplace=True)
        self.relu4 = nn.ELU(inplace=True)

        self.relu33 = nn.ELU(inplace=True)
        self.relu44 = nn.ELU(inplace=True)

        self.relu5 = nn.ELU(inplace=True)
        self.relu6 = nn.ELU(inplace=True)
        self.relu7 = nn.ELU(inplace=True)
        self.relu8 = nn.ELU(inplace=True)
        self.fuseconv = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out1 = self.dilated_conv_1(x)
        out2 = self.dilated_conv_2(x)
        out3 = self.dilated_conv_3(x)
        out4 = self.dilated_conv_4(x)

        out33 = self.dilated_conv_33(x)
        out44 = self.dilated_conv_44(x)

        out5 = self.dilated_conv_5(x)
        out6 = self.dilated_conv_6(x)
        out7 = self.dilated_conv_7(x)
        out8 = self.dilated_conv_8(x)

        out1 = self.relu1(out1)
        out2 = self.relu2(out2)
        out3 = self.relu3(out3)
        out4 = self.relu4(out4)

        out33 = self.relu33(out33)
        out44 = self.relu44(out44)

        out5 = self.relu5(out5)
        out6 = self.relu6(out6)
        out7 = self.relu7(out7)
        out8 = self.relu8(out8)
        # out = torch.cat((out1, out2, out3, out4),1)
        out = out1 + out2 + out3 + out4 + out5 + out6 + out7 + out8 + out33 + out44
        out = self.fuseconv(out) # possible option to fuse features or SE module
        return out
"""#LEDNet: A Lightweight Encoder-Decoder Network for Real-Time Semantic Segmentation"""
#=============================================================================================

###TPAMI2017 Deeplabv2
class ASPP(nn.Module): # deeplab

    def __init__(self, dim,in_dim):
        super(ASPP, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(dim,in_dim , 3,padding=1),nn.BatchNorm2d(in_dim),
             nn.PReLU())
        down_dim = in_dim // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=3, padding=3), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=9, padding=9), nn.BatchNorm2d(down_dim), nn.PReLU()
         )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1),nn.BatchNorm2d(down_dim),  nn.PReLU()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
        )

    def forward(self, x):
        x = self.down_conv(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        # conv5 = F.upsample(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear')
        conv5 = F.interpolate(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear')
        return self.fuse(torch.cat((conv1, conv2, conv3,conv4, conv5), 1))

###CVPR2019 AFNet: Attentive Feedback Network for Boundary-aware Salient Object Detection
class GPM(nn.Module): # cvpr19 AFNet -rgb_sod

    def __init__(self, in_dim):
        super(GPM, self).__init__()
        down_dim = 512
        n1, n2, n3 = 2, 4, 6
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(down_dim * n1 * n1, down_dim * n1 * n1, kernel_size=3, padding=1),
            nn.BatchNorm2d(down_dim * n1 * n1), nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(down_dim * n2 * n2, down_dim * n2 * n2, kernel_size=3, padding=1),
            nn.BatchNorm2d(down_dim * n2 * n2), nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(down_dim * n3 * n3, down_dim * n3 * n3, kernel_size=3, padding=1),
            nn.BatchNorm2d(down_dim * n3 * n3), nn.PReLU()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(3 * down_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        ###########################################################################
        gm_2_a = torch.chunk(conv1, 2, 2)
        c = []
        for i in range(len(gm_2_a)):
            b = torch.chunk(gm_2_a[i], 2, 3)
            c.append(torch.cat((b[0], b[1]), 1))
        gm1 = torch.cat((c[0], c[1]), 1)
        gm1 = self.conv2(gm1)
        gm1 = torch.chunk(gm1, 2 * 2, 1)
        d = []
        for i in range(2):
            d.append(torch.cat((gm1[2 * i], gm1[2 * i + 1]), 3))
        gm1 = torch.cat((d[0], d[1]), 2)
        ###########################################################################
        gm_4_a = torch.chunk(conv1, 4, 2)
        e = []
        for i in range(len(gm_4_a)):
            f = torch.chunk(gm_4_a[i], 4, 3)
            e.append(torch.cat((f[0], f[1], f[2], f[3]), 1))
        gm2 = torch.cat((e[0], e[1], e[2], e[3]), 1)
        gm2 = self.conv3(gm2)
        gm2 = torch.chunk(gm2, 4 * 4, 1)
        g = []
        for i in range(4):
            g.append(torch.cat((gm2[4 * i], gm2[4 * i + 1], gm2[4 * i + 2], gm2[4 * i + 3]), 3))
        gm2 = torch.cat((g[0], g[1], g[2], g[3]), 2)
        ###########################################################################
        gm_6_a = torch.chunk(conv1, 6, 2)
        h = []
        for i in range(len(gm_6_a)):
            k = torch.chunk(gm_6_a[i], 6, 3)
            h.append(torch.cat((k[0], k[1], k[2], k[3], k[4], k[5]), 1))

        gm3 = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5]), 1)
        gm3 = self.conv4(gm3)
        gm3 = torch.chunk(gm3, 6 * 6, 1)
        j = []
        for i in range(6):
            j.append(
                torch.cat((gm3[6 * i], gm3[6 * i + 1], gm3[6 * i + 2], gm3[6 * i + 3], gm3[6 * i + 4], gm3[6 * i + 5]),
                          3))
        gm3 = torch.cat((j[0], j[1], j[2], j[3], j[4], j[5]), 2)
        ###########################################################################

        return self.fuse(torch.cat((gm1, gm2, gm3), 1))


###ECCV2020 A Single Stream Network for Robust and Real-time RGB-D Salient Object Detection
class PAFEM(nn.Module):
    def __init__(self, dim,in_dim):
        super(PAFEM, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(dim,in_dim , 3,padding=1),nn.BatchNorm2d(in_dim),
             nn.PReLU())
        down_dim = in_dim // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv2 = Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        self.key_conv2 = Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        self.value_conv2 = Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma2 = Parameter(torch.zeros(1))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv3 = Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        self.key_conv3 = Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        self.value_conv3 = Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma3 = Parameter(torch.zeros(1))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv4 = Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        self.key_conv4 = Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        self.value_conv4 = Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma4 = Parameter(torch.zeros(1))

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1),nn.BatchNorm2d(down_dim),  nn.PReLU()  #如果batch=1 ，进行batchnorm会有问题
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
        )
        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        x = self.down_conv(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        m_batchsize, C, height, width = conv2.size()
        proj_query2 = self.query_conv2(conv2).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key2 = self.key_conv2(conv2).view(m_batchsize, -1, width * height)
        energy2 = torch.bmm(proj_query2, proj_key2)
        attention2 = self.softmax(energy2)
        proj_value2 = self.value_conv2(conv2).view(m_batchsize, -1, width * height)
        out2 = torch.bmm(proj_value2, attention2.permute(0, 2, 1))
        out2 = out2.view(m_batchsize, C, height, width)
        out2 = self.gamma2* out2 + conv2
        conv3 = self.conv3(x)
        m_batchsize, C, height, width = conv3.size()
        proj_query3 = self.query_conv3(conv3).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key3 = self.key_conv3(conv3).view(m_batchsize, -1, width * height)
        energy3 = torch.bmm(proj_query3, proj_key3)
        attention3 = self.softmax(energy3)
        proj_value3 = self.value_conv3(conv3).view(m_batchsize, -1, width * height)
        out3 = torch.bmm(proj_value3, attention3.permute(0, 2, 1))
        out3 = out3.view(m_batchsize, C, height, width)
        out3 = self.gamma3 * out3 + conv3
        conv4 = self.conv4(x)
        m_batchsize, C, height, width = conv4.size()
        proj_query4 = self.query_conv4(conv4).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key4 = self.key_conv4(conv4).view(m_batchsize, -1, width * height)
        energy4 = torch.bmm(proj_query4, proj_key4)
        attention4 = self.softmax(energy4)
        proj_value4 = self.value_conv4(conv4).view(m_batchsize, -1, width * height)
        out4 = torch.bmm(proj_value4, attention4.permute(0, 2, 1))
        out4 = out4.view(m_batchsize, C, height, width)
        out4 = self.gamma4 * out4 + conv4
        conv5 = F.upsample(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear') # 如果batch设为1，这里就会有问题。

        return self.fuse(torch.cat((conv1, out2, out3,out4, conv5), 1))

class ConvBNAct(nn.Module):
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 kernel_s: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 groups: int = 1,
                 act: Optional[nn.Module] = nn.ReLU(inplace=True)):
        super(ConvBNAct, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_c,
                              out_channels=out_c,
                              kernel_size=kernel_s,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=False)

        self.bn = nn.BatchNorm2d(out_c)
        self.act = act if act is not None else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class RegHead(nn.Module):
    def __init__(self,
                 in_unit: int = 368,
                 out_unit: int = 1000,
                 output_size: tuple = (1, 1),
                 drop_ratio: float = 0.25):
        super(RegHead, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size)

        if drop_ratio > 0:
            self.dropout = nn.Dropout(p=drop_ratio)
        else:
            self.dropout = nn.Identity()

        self.fc = nn.Linear(in_features=in_unit, out_features=out_unit)

    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class SqueezeExcitation(nn.Module):
    def __init__(self, input_c: int, expand_c: int, se_ratio: float = 0.25):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = int(input_c * se_ratio)
        self.fc1 = nn.Conv2d(expand_c, squeeze_c, 1)
        self.ac1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_c, expand_c, 1)
        self.ac2 = nn.Sigmoid()

    def forward(self, x):
        scale = x.mean((2, 3), keepdim=True)
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        return scale * x


class Bottleneck(nn.Module):
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 stride: int = 1,
                 group_width: int = 1,
                 se_ratio: float = 0.,
                 drop_ratio: float = 0.):
        super(Bottleneck, self).__init__()

        self.conv1 = ConvBNAct(in_c=in_c, out_c=out_c, kernel_s=1)
        self.conv2 = ConvBNAct(in_c=out_c,
                               out_c=out_c,
                               kernel_s=3,
                               stride=stride,
                               padding=1,
                               groups=out_c // group_width)

        if se_ratio > 0:
            self.se = SqueezeExcitation(in_c, out_c, se_ratio)
        else:
            self.se = nn.Identity()

        self.conv3 = ConvBNAct(in_c=out_c, out_c=out_c, kernel_s=1, act=None)
        self.ac3 = nn.ReLU(inplace=True)

        if drop_ratio > 0:
            self.dropout = nn.Dropout(p=drop_ratio)
        else:
            self.dropout = nn.Identity()

        if (in_c != out_c) or (stride != 1):
            self.downsample = ConvBNAct(in_c=in_c, out_c=out_c, kernel_s=1, stride=stride, act=None)
        else:
            self.downsample = nn.Identity()

    def zero_init_last_bn(self):
        nn.init.zeros_(self.conv3.bn.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.se(x)
        x = self.conv3(x)

        x = self.dropout(x)

        shortcut = self.downsample(shortcut)

        x += shortcut
        x = self.ac3(x)
        return x


class RegStage(nn.Module):
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 depth: int,
                 group_width: int,
                 se_ratio: float):
        super(RegStage, self).__init__()
        for i in range(depth):
            block_stride = 2 if i == 0 else 1
            block_in_c = in_c if i == 0 else out_c

            name = "b{}".format(i + 1)
            self.add_module(name,
                            Bottleneck(in_c=block_in_c,
                                       out_c=out_c,
                                       stride=block_stride,
                                       group_width=group_width,
                                       se_ratio=se_ratio))

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x

###ECCV2020 Suppress and Balance: A Simple Gated Network for Salient Object Detection
class FoldConv_aspp(nn.Module):
    def __init__(self, in_channel, out_channel,
                 kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 win_size=3, win_dilation=1, win_padding=0):
        super(FoldConv_aspp, self).__init__()
        #down_C = in_channel // 8
        self.down_conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3,padding=1),nn.BatchNorm2d(out_channel),
             nn.PReLU())
        self.win_size = win_size
        self.unfold = nn.Unfold(win_size, win_dilation, win_padding, win_size)
        fold_C = out_channel * win_size * win_size
        down_dim = fold_C // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(fold_C, down_dim,kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(fold_C, down_dim, kernel_size, stride, padding, dilation, groups),
            nn.BatchNorm2d(down_dim),
            nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(fold_C, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(fold_C, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d( down_dim), nn.PReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(fold_C, down_dim, kernel_size=1),nn.BatchNorm2d(down_dim),  nn.PReLU()  #如果batch=1 ，进行batchnorm会有问题
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, fold_C, kernel_size=1), nn.BatchNorm2d(fold_C), nn.PReLU()
        )

        # self.fold = nn.Fold(out_size, win_size, win_dilation, win_padding, win_size)

        self.up_conv = nn.Conv2d(out_channel, out_channel, 1)

    def forward(self, in_feature):
        N, C, H, W = in_feature.size()
        in_feature = self.down_conv(in_feature)
        in_feature = self.unfold(in_feature)
        in_feature = in_feature.view(in_feature.size(0), in_feature.size(1),
                                     H // self.win_size, W // self.win_size)
        in_feature1 = self.conv1(in_feature)
        in_feature2 = self.conv2(in_feature)
        in_feature3 = self.conv3(in_feature)
        in_feature4 = self.conv4(in_feature)
        in_feature5 = F.upsample(self.conv5(F.adaptive_avg_pool2d(in_feature, 1)), size=in_feature.size()[2:], mode='bilinear')
        in_feature = self.fuse(torch.cat((in_feature1, in_feature2, in_feature3,in_feature4,in_feature5), 1))
        in_feature = in_feature.reshape(in_feature.size(0), in_feature.size(1), -1)


        in_feature = F.fold(input=in_feature, output_size=H, kernel_size=2, dilation=1, padding=0, stride=2)
        in_feature = self.up_conv(in_feature)
        return in_feature



def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, groups=inp, bias=False),
        FPEBlock(oup,oup,[1,2,4,8]),
        nn.BatchNorm2d(inp),
        nn.PReLU(inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.PReLU(inplace=True),

    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
 
        self.use_res_connect = self.stride == 1 and inp == oup
 
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
 
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class InvertedResidual1(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual1, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
 
        self.use_res_connect = self.stride == 1 and inp == oup
 
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            #nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            conv_dw(inp * expand_ratio, inp * expand_ratio, stride),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
 
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class reverse_attention(nn.Module):
    def __init__(self, in_channel, channel, depth=3, kernel_size=3):
        super(reverse_attention, self).__init__()
        self.conv_in = conv(in_channel, channel, 1)
        self.conv_mid = nn.ModuleList()
        for i in range(depth):
            self.conv_mid.append(conv(channel, channel, kernel_size))
        self.conv_out = conv(channel, 1, 3 if kernel_size == 3 else 1)

    def forward(self, x, map):
        map = F.interpolate(map, size=x.shape[-2:], mode='bilinear', align_corners=False)
        rmap = -1 * (torch.sigmoid(map)) + 1
        
        x = rmap.expand(-1, x.shape[1], -1, -1).mul(x)
        x = self.conv_in(x)
        for conv_mid in self.conv_mid:
            x = F.relu(conv_mid(x))
        out = self.conv_out(x)
        out = out + map

        return x, out

class simple_attention(nn.Module):
    def __init__(self, in_channel, channel, depth=3, kernel_size=3):
        super(simple_attention, self).__init__()
        self.conv_in = conv(in_channel, channel, 1)
        self.conv_mid = nn.ModuleList()
        for i in range(depth):
            self.conv_mid.append(conv(channel, channel, kernel_size))
        self.conv_out = conv(channel, 1, 1)

    def forward(self, x, map):
        map = F.interpolate(map, size=x.shape[-2:], mode='bilinear', align_corners=False)
        amap = torch.sigmoid(map)
        
        x = amap.expand(-1, x.shape[1], -1, -1).mul(x)
        x = self.conv_in(x)
        for conv_mid in self.conv_mid:
            x = F.relu(conv_mid(x))
        out = self.conv_out(x)
        out = out + map

        return x, out

class CA(nn.Module):
    def __init__(self, in_channel, channel):
        super(CA, self).__init__()
        self.channel = channel

        self.conv_query = nn.Sequential(conv(in_channel, channel, 3, relu=True),
                                        conv(channel, channel, 3, relu=True))
        self.conv_key = nn.Sequential(conv(in_channel, channel, 1, relu=True),
                                      conv(channel, channel, 1, relu=True))
        self.conv_value = nn.Sequential(conv(in_channel, channel, 1, relu=True),
                                        conv(channel, channel, 1, relu=True))

        self.conv_out1 = conv(channel, channel, 3, relu=True)
        self.conv_out2 = conv(in_channel + channel, channel, 3, relu=True)
        self.conv_out3 = conv(channel, channel, 3, relu=True)
        self.conv_out4 = conv(channel, 1, 1)

    def forward(self, x, map):
        b, c, h, w = x.shape
        
        # compute class probability
        map = F.interpolate(map, size=x.shape[-2:], mode='bilinear', align_corners=False)
        fg = torch.sigmoid(map)
        
        p = fg - .5

        fg = torch.clip(p, 0, 1) # foreground
        bg = torch.clip(-p, 0, 1) # background

        prob = torch.cat([fg, bg], dim=1)

        # reshape feature & prob
        f = x.view(b, h * w, -1)
        prob = prob.view(b, 2, h * w)
        
        # compute context vector
        context = torch.bmm(prob, f).permute(0, 2, 1).unsqueeze(3) # b, 2, c

        # k q v compute
        query = self.conv_query(x).view(b, self.channel, -1).permute(0, 2, 1)
        key = self.conv_key(context).view(b, self.channel, -1)
        value = self.conv_value(context).view(b, self.channel, -1).permute(0, 2, 1)

        # compute similarity map
        sim = torch.bmm(query, key) # b, hw, c x b, c, 2
        sim = (self.channel ** -.5) * sim
        sim = F.softmax(sim, dim=-1)

        # compute refined feature
        context = torch.bmm(sim, value).permute(0, 2, 1).contiguous().view(b, -1, h, w)
        context = self.conv_out1(context)

        x = torch.cat([x, context], dim=1)
        x = self.conv_out2(x)
        x = self.conv_out3(x)
        out = self.conv_out4(x)
        out = out + map
        return x, out




class conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, padding='same', bias=False, bn=True, relu=False):
        super(conv, self).__init__()
        if '__iter__' not in dir(kernel_size):
            kernel_size = (kernel_size, kernel_size)
        if '__iter__' not in dir(stride):
            stride = (stride, stride)
        if '__iter__' not in dir(dilation):
            dilation = (dilation, dilation)

        if padding == 'same':
            width_pad_size = kernel_size[0] + (kernel_size[0] - 1) * (dilation[0] - 1)
            height_pad_size = kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1)
        elif padding == 'valid':
            width_pad_size = 0
            height_pad_size = 0
        else:
            if '__iter__' in dir(padding):
                width_pad_size = padding[0] * 2
                height_pad_size = padding[1] * 2
            else:
                width_pad_size = padding * 2
                height_pad_size = padding * 2

        width_pad_size = width_pad_size // 2 + (width_pad_size % 2 - 1)
        height_pad_size = height_pad_size // 2 + (height_pad_size % 2 - 1)
        pad_size = (width_pad_size, height_pad_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_size, dilation, groups, bias=bias)
        self.reset_parameters()

        if bn is True:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        
        if relu is True:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv.weight)


class self_attn(nn.Module):
    def __init__(self, in_channels, mode='hw'):
        super(self_attn, self).__init__()

        self.mode = mode

        self.query_conv = conv(in_channels, in_channels // 8, kernel_size=(1, 1))
        self.key_conv = conv(in_channels, in_channels // 8, kernel_size=(1, 1))
        self.value_conv = conv(in_channels, in_channels, kernel_size=(1, 1))

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channel, height, width = x.size()

        axis = 1
        if 'h' in self.mode:
            axis *= height
        if 'w' in self.mode:
            axis *= width

        view = (batch_size, -1, axis)

        projected_query = self.query_conv(x).view(*view).permute(0, 2, 1)
        projected_key = self.key_conv(x).view(*view)

        attention_map = torch.bmm(projected_query, projected_key)
        attention = self.softmax(attention_map)
        projected_value = self.value_conv(x).view(*view)

        out = torch.bmm(projected_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channel, height, width)

        out = self.gamma * out + x
        return out

class UACA(nn.Module):
    def __init__(self, in_channel, channel):
        super(UACA, self).__init__()
        self.channel = channel

        self.conv_query = nn.Sequential(conv(in_channel, channel, 3, relu=True),
                                        conv(channel, channel, 3, relu=True))
        self.conv_key = nn.Sequential(conv(in_channel, channel, 1, relu=True),
                                      conv(channel, channel, 1, relu=True))
        self.conv_value = nn.Sequential(conv(in_channel, channel, 1, relu=True),
                                        conv(channel, channel, 1, relu=True))

        self.conv_out1 = conv(channel, channel, 3, relu=True)
        self.conv_out2 = conv(in_channel + channel, channel, 3, relu=True)
        self.conv_out3 = conv(channel, channel, 3, relu=True)
        self.conv_out4 = conv(channel, 1, 1)

    def forward(self, x, map):
        b, c, h, w = x.shape
        
        # compute class probability
        map = F.interpolate(map, size=x.shape[-2:], mode='bilinear', align_corners=False)
        fg = torch.sigmoid(map)
        
        p = fg - .5

        fg = torch.clip(p, 0, 1) # foreground
        bg = torch.clip(-p, 0, 1) # background
        cg = .5 - torch.abs(p) # confusion area

        prob = torch.cat([fg, bg, cg], dim=1)

        # reshape feature & prob
        f = x.view(b, h * w, -1)
        prob = prob.view(b, 3, h * w)
        
        # compute context vector
        context = torch.bmm(prob, f).permute(0, 2, 1).unsqueeze(3) # b, 3, c

        # k q v compute
        query = self.conv_query(x).view(b, self.channel, -1).permute(0, 2, 1)
        key = self.conv_key(context).view(b, self.channel, -1)
        value = self.conv_value(context).view(b, self.channel, -1).permute(0, 2, 1)

        # compute similarity map
        sim = torch.bmm(query, key) # b, hw, c x b, c, 2
        sim = (self.channel ** -.5) * sim
        sim = F.softmax(sim, dim=-1)

        # compute refined feature
        context = torch.bmm(sim, value).permute(0, 2, 1).contiguous().view(b, -1, h, w)
        context = self.conv_out1(context)

        x = torch.cat([x, context], dim=1)
        x = self.conv_out2(x)
        x = self.conv_out3(x)
        out = self.conv_out4(x)
        out = out + map
        return x, out
