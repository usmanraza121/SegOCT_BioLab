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
from paddleseg.models import layers




class UAFM(nn.Layer):
    """
    The base of Unified Attention, he Fusion Module.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """
 
    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__()
 
        self.conv_x = layers.ConvBNReLU(
            x_ch, y_ch, kernel_size=ksize, padding=ksize // 2, bias_attr=False)
        self.conv_out = layers.ConvBNReLU(
            y_ch, out_ch, kernel_size=3, padding=1, bias_attr=False)
        self.resize_mode = resize_mode
 
    def check(self, x, y):
        assert x.ndim == 4 and y.ndim == 4
        x_h, x_w = x.shape[2:]
        y_h, y_w = y.shape[2:]
        assert x_h >= y_h and x_w >= y_w
 
    def prepare(self, x, y):
        x = self.prepare_x(x, y)
        y = self.prepare_y(x, y)
        return x, y
 
    def prepare_x(self, x, y):
        x = self.conv_x(x)
        return x
 
    def prepare_y(self, x, y):
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode=self.resize_mode)
        return y_up
 
    def fuse(self, x, y):
        out = x + y
        out = self.conv_out(out)
        return out
 
    def forward(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        self.check(x, y)
        x, y = self.prepare(x, y)
        out = self.fuse(x, y)
        return out

#==================================================================================

class UAFM_SpAtten(UAFM):
    """
    The UAFM with spatial attention, which uses mean and max values.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """
 
    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)
 
        self.conv_xy_atten = nn.Sequential(
            layers.ConvBNReLU(
                4, 2, kernel_size=3, padding=1, bias_attr=False),
            layers.ConvBN(
                2, 1, kernel_size=3, padding=1, bias_attr=False))
 
    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        atten = helper.avg_max_reduce_channel([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))
 
        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out

#================================================================================================
class UAFM_ChAtten(UAFM):
    """
    The UAFM with channel attention, which uses mean and max values.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """
 
    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)
 
        self.conv_xy_atten = nn.Sequential(
            layers.ConvBNAct(
                4 * y_ch,
                y_ch // 2,
                kernel_size=1,
                bias_attr=False,
                act_type="leakyrelu"),
            layers.ConvBN(
                y_ch // 2, y_ch, kernel_size=1, bias_attr=False))
 
    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        atten = helper.avg_max_reduce_hw([x, y], self.training)
        atten = F.sigmoid(self.conv_xy_atten(atten))
 
        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out

#===============================================================================
class PPContextModule(nn.Layer):
    """
    Simple Context module.
    Args:
        in_channels (int): The number of input channels to pyramid pooling module.
        inter_channels (int): The number of inter channels to pyramid pooling module.
        out_channels (int): The number of output channels after pyramid pooling module.
        bin_sizes (tuple, optional): The out size of pooled feature maps. Default: (1, 3).
        align_corners (bool): An argument of F.interpolate. It should be set to False
            when the output size of feature is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
    """
 
    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 bin_sizes,
                 align_corners=False):
        super().__init__()
 
        self.stages = nn.LayerList([
            self._make_stage(in_channels, inter_channels, size)
            for size in bin_sizes
        ])
 
        self.conv_out = layers.ConvBNReLU(
            in_channels=inter_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1)
 
        self.align_corners = align_corners
 
    def _make_stage(self, in_channels, out_channels, size):
        prior = nn.AdaptiveAvgPool2D(output_size=size)
        conv = layers.ConvBNReLU(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        return nn.Sequential(prior, conv)
 
    def forward(self, input):
        out = None
        input_shape = paddle.shape(input)[2:]
 
        for stage in self.stages:
            x = stage(input)
            x = F.interpolate(
                x,
                input_shape,
                mode='bilinear',
                align_corners=self.align_corners)
            if out is None:
                out = x
            else:
                out += x
 
        out = self.conv_out(out)
        return out

#===============================paddle loss============================================

import paddle
from paddle import nn
import paddle.nn.functional as F
 
from paddleseg.cvlibs import manager
 
 
@manager.LOSSES.add_component
class OhemCrossEntropyLoss(nn.Layer):
    """
    Implements the ohem cross entropy loss function.
    Args:
        thresh (float, optional): The threshold of ohem. Default: 0.7.
        min_kept (int, optional): The min number to keep in loss computation. Default: 10000.
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
    """
 
    def __init__(self, thresh=0.7, min_kept=10000, ignore_index=255):
        super(OhemCrossEntropyLoss, self).__init__()
        self.thresh = thresh
        self.min_kept = min_kept
        self.ignore_index = ignore_index
        self.EPS = 1e-5
 
    def forward(self, logit, label):
        """
        Forward computation.
        Args:
            logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C), where C is number of classes, and if shape is more than 2D, this
                is (N, C, D1, D2,..., Dk), k >= 1.
            label (Tensor): Label tensor, the data type is int64. Shape is (N), where each
                value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
                (N, D1, D2,..., Dk), k >= 1.
        """
        if len(label.shape) != len(logit.shape):
            label = paddle.unsqueeze(label, 1)
 
        # get the label after ohem
        n, c, h, w = logit.shape
        label = label.reshape((-1, ))
        valid_mask = (label != self.ignore_index).astype('int64')
        num_valid = valid_mask.sum()
        label = label * valid_mask
 
        prob = F.softmax(logit, axis=1)
        prob = prob.transpose((1, 0, 2, 3)).reshape((c, -1))
 
        if self.min_kept < num_valid and num_valid > 0:
            # let the value which ignored greater than 1
            prob = prob + (1 - valid_mask)
 
            # get the prob of relevant label
            label_onehot = F.one_hot(label, c)
            label_onehot = label_onehot.transpose((1, 0))
            prob = prob * label_onehot
            prob = paddle.sum(prob, axis=0)
 
            threshold = self.thresh
            if self.min_kept > 0:
                index = prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                threshold_index = int(threshold_index.numpy()[0])
                if prob[threshold_index] > self.thresh:
                    threshold = prob[threshold_index]
                kept_mask = (prob < threshold).astype('int64')
                label = label * kept_mask
                valid_mask = valid_mask * kept_mask
 
        # make the invalid region as ignore
        label = label + (1 - valid_mask) * self.ignore_index
 
        label = label.reshape((n, 1, h, w))
        valid_mask = valid_mask.reshape((n, 1, h, w)).astype('float32')
        loss = F.softmax_with_cross_entropy(
            logit, label, ignore_index=self.ignore_index, axis=1)
        loss = loss * valid_mask
        avg_loss = paddle.mean(loss) / (paddle.mean(valid_mask) + self.EPS)
 
        label.stop_gradient = True
        valid_mask.stop_gradient = True
        return avg_loss

#===============================================================