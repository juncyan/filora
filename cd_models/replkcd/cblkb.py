import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import os
import paddleseg.models.layers as layers
import sys
from .reparams import Reparams, RepConvBn
# from ..utils import *


class RepConv(Reparams):
    def __init__(self, in_channels, small_kernels=9, dilation = 4):
        super(RepConv, self).__init__()

        # self.kernel_sizes = [5, 9, 3, 3, 3] lk = 17
        # self.dilates = [1, 2, 4, 5, 7]

        self.in_channels = in_channels
        self.dilation = dilation
        self.sk = small_kernels
        self.lk = (self.sk -1)*self.dilation + 1

        # self.conv1 = layers.ConvBN(in_channels, in_channels, self.sk, groups=in_channels)
        # self.conv2 = layers.ConvBN(in_channels, in_channels, 3, dilation=4, groups=in_channels)
        # self.conv3 = layers.ConvBN(in_channels, in_channels, 3, dilation=8, groups=in_channels)
        # self.lkc = layers.ConvBN(in_channels, in_channels, self.sk, dilation=self.dilation, groups=in_channels)

        self.conv1 = nn.Conv2D(in_channels,in_channels,self.sk,padding=self.sk//2,groups=in_channels)
        self.conv2 = nn.Conv2D(in_channels,in_channels,3,padding=4,dilation=4,groups=in_channels)
        self.conv3 = nn.Conv2D(in_channels,in_channels,3,padding=8,dilation=8,groups=in_channels)
        self.lkc = nn.Conv2D(in_channels,in_channels,self.sk,padding=self.lk//2,dilation=dilation,groups=in_channels)
    
    def forward(self, x):
        if not self.training and hasattr(self, "repc"):
            # print("repc")
            y = self.repc(x)
            y = F.relu(y)
            return y
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y3 = self.conv3(x)
        y4 = self.lkc(x)
        y = y1 + y2 + y3 + y4
        y = F.relu(y)
        return y

    def get_equivalent_kernel_bias(self):
        kernel1, bias1 = self._fuse_conv_bn(self.conv1)  
        kernel2, bias2 = self._fuse_conv_bn(self.conv2)
        kernel3, bias3 = self._fuse_conv_bn(self.conv3)
        klkc, biaslkc = self._fuse_conv_bn(self.lkc)
        # print(kernel1.shape, kernel2.shape, klkc.shape, bias1.shape, bias2.shape, biaslkc.shape)
        return klkc + kernel1 + kernel2 + kernel3,  bias1 + bias2 + biaslkc + bias3
 

class RepC3(RepConvBn):
    def __init__(self, in_channels):
        super(RepC3, self).__init__()
        self.in_channels = in_channels
        self.lk = 3
        self.conv1 = nn.Conv2D(in_channels, in_channels, 3,1,1,groups=in_channels)
        self.bn1 = None#nn.BatchNorm2D(in_channels)
        self.conv2 = nn.Conv2D(in_channels, in_channels, 1,1,0,groups=in_channels)
        self.bn2 = None#nn.BatchNorm2D(in_channels)
        # self.conv2 = layers.ConvBN(in_channels, in_channels, 1, groups=in_channels, bias_attr=False)


    def get_equivalent_kernel_bias(self):
        
        kernel1, bias1 = self._repparams(self.conv1, self.bn1)  
        kernel2, bias2 = self._repparams(self.conv2, self.bn2)    
        return kernel1 + kernel2, bias1 + bias2
    
    
    def forward(self, x):
        if hasattr(self, "repc"):
            y = self.repc(x)
            return y
        
        y1 = self.conv1(x)
        y2 = self.conv2(x)

        y = y1 + y2
        return y



if __name__ == "__main__":
    print("replknet")
    from paddleseg.utils import op_flops_funs
    from visualdl import LogWriter
    x = paddle.rand([1,128,256,256]).cuda()
    m = RepConv(128).to('gpu:0')
    y1 = m(x)
    paddle.save(m.state_dict(), "testrepc.pdparams")
    layer_state_dict = paddle.load("testrepc.pdparams")
    # print(layer_state_dict)
    
    # with paddle.no_grad():
    m.set_state_dict(layer_state_dict)
    m.eval()
    
    y2 = m(x)
    print(y1==y2)
    # paddle.flops(
    #     m, [1, 6, 16, 16],
    #     custom_ops={paddle.nn.SyncBatchNorm: op_flops_funs.count_syncbn})
    

