import torch
from torch.nn.modules.activation import PReLU
from torch.nn.modules.batchnorm import BatchNorm2d
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from .modules import*
# from modules import*
import torchvision.models as models
from .encoders import get_encoder
from .UAFM import*
# class conv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, padding='same', bias=False, bn=True, relu=False):
#         super(conv, self).__init__()
#         if '__iter__' not in dir(kernel_size):
#             kernel_size = (kernel_size, kernel_size)
#         if '__iter__' not in dir(stride):
#             stride = (stride, stride)
#         if '__iter__' not in dir(dilation):
#             dilation = (dilation, dilation)

#         if padding == 'same':
#             width_pad_size = kernel_size[0] + (kernel_size[0] - 1) * (dilation[0] - 1)
#             height_pad_size = kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1)
#         elif padding == 'valid':
#             width_pad_size = 0
#             height_pad_size = 0
#         else:
#             if '__iter__' in dir(padding):
#                 width_pad_size = padding[0] * 2
#                 height_pad_size = padding[1] * 2
#             else:
#                 width_pad_size = padding * 2
#                 height_pad_size = padding * 2

#         width_pad_size = width_pad_size // 2 + (width_pad_size % 2 - 1)
#         height_pad_size = height_pad_size // 2 + (height_pad_size % 2 - 1)
#         pad_size = (width_pad_size, height_pad_size)
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_size, dilation, groups, bias=bias)
#         self.reset_parameters()

#         if bn is True:
#             self.bn = nn.BatchNorm2d(out_channels)
#         else:
#             self.bn = None
        
#         if relu is True:
#             self.relu = nn.ReLU(inplace=True)
#         else:
#             self.relu = None

#     def forward(self, x):
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x

#     def reset_parameters(self):
#         nn.init.kaiming_normal_(self.conv.weight)


class PPD(nn.Module):
    def __init__(self, channel):
        super(PPD, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = lambda img, size: F.interpolate(img, size=size, mode='bilinear', align_corners=True)
        # self.conv_upsample1 = conv(channel, channel, 3)
        # self.conv_upsample2 = conv(channel, channel, 3)
        # self.conv_upsample3 = conv(channel, channel, 3)
        # self.conv_upsample4 = conv(channel, channel, 3)
        self.conv_upsample1 = BasicConv(channel, channel, 5,stride=1,padding=6,dilation=3)
        self.conv_upsample2 = BasicConv(channel, channel, 7,stride=1,padding=15,dilation=5)
        self.conv_upsample3 = BasicConv(channel, channel, 9,stride=1,padding=28,dilation=7)
        self.conv_upsample4 = BasicConv(channel, channel, 11,stride=1,padding=45,dilation=9)
        #self.conv_upsample5 = BasicConv(2 * channel, 2 * channel, 3)

        self.conv_concat2 = BasicConv(2 * channel, 2 * channel, 3)
        self.conv_concat3 = BasicConv(3 * channel, 3 * channel, 3)
        self.conv4 = BasicConv(3 * channel, 3 * channel, 3)
        self.conv5 = BasicConv(3 * channel, 1, 1, bn=False, bias=True)

    def forward(self, f1, f2, f3):
        # print(f1.shape)
        # print(f2.shape)
        # print(f3.shape)
        f1x2 = self.upsample(f1, f2.shape[-2:])
        f1x4 = self.upsample(f1, f3.shape[-2:])
        f2x2 = self.upsample(f2, f3.shape[-2:])
        #print(f1x2.shape)
        f2_1 = self.conv_upsample1(f1x2) * f2
        f3_1 = self.conv_upsample2(f1x4) * self.conv_upsample3(f2x2) * f3
        #print('f3_1',f3_1.shape)
        f1_2 = self.conv_upsample4(f1x2)
        
        f2_2 = torch.cat([f2_1, f1_2], 1)
        f2_2 = self.conv_concat2(f2_2)

        f2_2x2 = self.upsample(f2_2, f3.shape[-2:])
        #print('f2_22',f2_2x2.shape)
        #f2_2x2 = self.conv_upsample5(f2_2x2)
        #print('f2_22',f2_2x2.shape)
        f3_2 = torch.cat([f3_1, f2_2x2], 1)
        f3_2 = self.conv_concat3(f3_2)

        f3_2 = self.conv4(f3_2)
        out = self.conv5(f3_2)

        return f3_2, out
class mobilenet_v2(nn.Module):
    def __init__(self, nInputChannels=3):
        super(mobilenet_v2, self).__init__()
        # 1
        self.head_conv = nn.Sequential(nn.Conv2d(nInputChannels, 32, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU())
        # 1
        self.block_1 = InvertedResidual(32, 16, 1, 1)
        # 1/2 
        self.block_2 = nn.Sequential( 
            InvertedResidual(16, 24, 2, 6),
            InvertedResidual(24, 24, 1, 6)
            )
        # 1/4 
        self.block_3 = nn.Sequential( 
            InvertedResidual(24, 32, 2, 6),
            InvertedResidual(32, 32, 1, 6),
            InvertedResidual(32, 32, 1, 6)
            )
        # 1/8 
        self.block_4 = nn.Sequential( 
            InvertedResidual(32, 64, 2, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6)            
            )
        # 1/16
        self.block_5 = nn.Sequential( 
            InvertedResidual(64, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6)          
            )
        # 1/32 
        self.block_6 = nn.Sequential( 
            InvertedResidual(96, 160, 2, 6),
            InvertedResidual(160, 160, 1, 6),
            InvertedResidual(160, 160, 1, 6)          
            )
        # 1/32
        self.block_7 = InvertedResidual(160, 320, 1, 6)
 
    def forward(self, x):
        x = self.head_conv(x)
        # 1
        s1 = self.block_1(x)
        # 1/2 
        s2 = self.block_2(s1)
        # 1/4
        s3 = self.block_3(s2)
        # 1/8
        s4 = self.block_4(s3)
        s4 = self.block_5(s4)
        # 1/16
        s5 = self.block_6(s4)
        s5 = self.block_7(s5)
 
        return s1, s2, s3, s4, s5


class tnet(nn.Module):
    def __init__(self, classes=4, name="TNet"):
        super(tnet, self).__init__()
        self.name = name
        # ===================================efficient encoder===========================
        # encoder from hydranet
        compound_coef =0
        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6, 7]
        self.encoder = get_encoder(
            'efficientnet-b' + str(self.backbone_compound_coef[compound_coef]),
            in_channels=3,
            depth=5,
            weights='imagenet',
        )
        # ============================MobileNet Encoder==================================
        #self.feature = mobilenet_v2()

        #self.aspp = ASPP(320,320)
        self.pfcu = PFCU(320)
        self.trans = AIM(iC_list=(32, 24, 40, 112, 320), oC_list=(32, 24, 40, 112, 320))
        #self.uacap5 = UACA(320,320)
        # self.uacap4 = ASPP(112,112)
        # self.uacap3 = ASPP(40,40)
        #self.uacap2 = UACA(24,24)
        self.ufamspat1 = UAFM(112,320,112)
        self.ufamspat2 = UAFM(40,112,40)
        self.ufamcha = UAFM(32,40,32)

        # self.pd1 = ParallelDilatedConv(112, 112)
        # self.pd2 = ParallelDilatedConv(40, 40)
        # self.pd3 = ParallelDilatedConv(32, 32)

        #self.fuse =conv_2nV1(96, 320,32)
        #self.upcahnl = nn.Conv2d(24,32,1)
        #self.ppd = PPD(32)
        # -----------------------------------------------------------------
        # decoder 
        # ---------------------
        self.cls_conv = nn.Conv2d(32, classes, 1, stride=1)
        #self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
    def forward(self, input):
        H, W = input.size(2), input.size(3)
        # -----------------------------------------------
        p1, p2, p3, p4, p5 = self.encoder(input)[-5:]
        # print('pp1', p1.shape)
        # print('pp2', p2.shape)
        # print('pp3', p3.shape)
        # print('pp4', p4.shape)
        # print('pp5', p5.shape)
        # p5 = self.pfcu(p5)
        p1, p2, p3, p4, p5 = self.trans(p1, p2, p3, p4, p5)
        # print('p1', p1.shape)
        # print('p2', p2.shape)
        # print('p3', p3.shape)
        # print('p4', p4.shape)
        # print('p5', p5.shape)

        p4update  = self.ufamspat1(p4,p5)
        # p4update = self.pd1(p4update)
        #print('s4update', s4update.shape
        #p4update1 = self.uacap4(p4update)
        p3update = self.ufamspat2(p3,p4update)
        # p3update = self.pd2(p3update)
        
        #print('s3update', s3update.shape)
        #p3update1 = self.uacap3(p3update)
        p2update  =self.ufamcha(p1,p3update)
        # p2update = self.pd3(p2update)

        x = self.cls_conv(p2update)
        # ---------------------
        ##s1, s2, s3, s4, s5 = self.feature(input)
        # print('s5::', s5.shape)
        ##s5 = self.pfcu(s5)
        # print('s5::', s5.shape)
        ##s1, s2, s3, s4, s5 = self.trans(s1, s2, s3, s4, s5)
        # print('s1', s1.shape)
        # print('s2', s2.shape)
        # print('s3', s3.shape)
        # print('s4', s4.shape)
        # print('s5', s5.shape)
        ##s4update  = self.ufamspat1(s4,s5)
        #print('s4update', s4update.shape)
        ##s3update  = self.ufamspat2(s3,s4update)
        #print('s3update', s3update.shape)
        ##s2update  =self.ufamcha(s2,s3update)
        #print('s2update', s2update.shape)
        #fuse = self.fuse(s4,s5)
        #upchannel = self.upcahnl(s2)
        #f3_2, sigout= self.ppd(upchannel,s3,fuse)
        # -----------------------------------------------
        # decoder
        # ---------------------
        # x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
        # x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        ##x = self.cls_conv(s2update)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x



if __name__ == "__main__":

    #in_data = torch.randn((1, 3, 320, 320))
    #net = mobilenet_v2().cuda()
    # net = tnet().cuda()
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # net = tnet().to(device)
    #model = models.mobilenet_v2(pretrained=True).to(device)
    # summary(net, (3, 224, 224))
    # encod = models.mobilenet_v2(pretrained=True)
    # #print(encod)
    # encod_layers = list(encod.children())
    #print(encod_layers)
    # blocks1 = nn.Sequential(*encod_layers[0:1])
    # print(blocks1)
    model =tnet(classes=4)
    input_ = torch.randn((1, 3, 512, 512))
    # gt_ = torch.rand((1, 2, 256, 256))

#     model_out,SAD_out, lout = model(input_)
#     model_out,SAD_out, lout = model(input_)
#     detects, dring_area_seg, lane_line_seg = model_out
    dring_area_seg= model(input_)
    # Da_fmap, LL_fmap = SAD_out

    print(dring_area_seg.shape)
    # print(dring_area_seg)