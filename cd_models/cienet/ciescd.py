
import re
import time
import math
import numpy as np
from functools import partial
from typing import Optional, Union, Type, List, Tuple, Callable, Dict

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.models import layers
from paddleseg.utils import load_entire_model
from paddleclas.ppcls.arch.backbone import ViT_base_patch16_224, SwinTransformer_base_patch4_window7_224, ResNet50_vd, ResNet34_vd
from cd_models.vmamba import VSSBackbone
from einops import rearrange, repeat
from .mobilesam import MobileSAM


class CIENetRes50vd(nn.Layer):
    def __init__(self, img_size, num_seg=7, **kwargs):
        super().__init__()
        self.backbone = ResNet50_vd(pretrained=True)
        self.backbone.eval()

        self.img_size = [img_size, img_size]
        self.clim = CrossDimensionalFeatureFusion(2048, 256, 64)
        
        self.bdia = BitemporalDifferenceInformationAggregation(64, 64)
        self.cls = nn.Sequential(layers.ConvBNAct(64,64,3, act_type='gelu'), 
                                 nn.Conv2D(64, 1, 3, 1, 1))

        self.dmfe = SemanticInformationEnhancement(64)
        self.scls1 = nn.Sequential(layers.ConvBNAct(64,64,3, act_type='gelu'), 
                                   nn.Conv2D(64, num_seg, 3, 1, 1))
    
    def forward(self, x1, x2=None):
        if x2 is None:
            x = paddle.split(x1, 2, axis=1)
            x1 = x[0]
            x2 = x[1]
    
        f1, f2 = self.encoder(x1)
        p1, p2 = self.encoder(x2)

        f2 = self.clim(f2, f1)
        p2 = self.clim(p2, p1)
 
        y = self.bdia(f2, p2)
        y = F.interpolate(y, size=self.img_size, mode='bilinear', align_corners=True)
        y = self.cls(y)

        s1 = self.dmfe(f2)
        s2 = self.dmfe(p2)
        s1 = F.interpolate(s1, size=self.img_size, mode='bilinear', align_corners=True)
        s2 = F.interpolate(s2, size=self.img_size, mode='bilinear', align_corners=True)
        s1 = self.scls1(s1)
        s2 = self.scls1(s2)
        return y, s1, s2
    
    def encoder(self, x):   
        x = self.backbone.stem(x)
        if self.backbone.max_pool:
            x = self.backbone.max_pool(x)
        y = None
        for b in self.backbone.blocks:
            x = b(x)
            if y is None:
                y = x
        return y, x

class CIENetSwinT(nn.Layer):
    def __init__(self, img_size, num_seg=7, **kwargs):
        super().__init__()
        # in_channels=3,
        # pretrained=None
        self.backbone = SwinTransformer_base_patch4_window7_224(pretrained=True)
        self.backbone.eval()

        self.img_size = [img_size, img_size]
        self.clim = CrossDimensionalFeatureFusion(1024, 128, 64)
        
        self.bdia = BitemporalDifferenceInformationAggregation(64, 64)
        self.cls = nn.Sequential(layers.ConvBNAct(64,64,3, act_type='gelu'), 
                                 nn.Conv2D(64, 1, 3, 1, 1))

        self.dmfe = SemanticInformationEnhancement(64)
        self.scls1 = nn.Sequential(layers.ConvBNAct(64,64,3, act_type='gelu'), 
                                   nn.Conv2D(64, num_seg, 3, 1, 1))
    
    def forward(self, x1, x2=None):
        if x2 is None:
            x = paddle.split(x1, 2, axis=1)
            x1 = x[0]
            x2 = x[1]
    
        f1, f2 = self.encoder(x1)
        p1, p2 = self.encoder(x2)

        f2 = self.clim(f2, f1)
        p2 = self.clim(p2, p1)
 
        y = self.bdia(f2, p2)
        y = F.interpolate(y, size=self.img_size, mode='bilinear', align_corners=True)
        y = self.cls(y)

        s1 = self.dmfe(f2)
        s2 = self.dmfe(p2)
        s1 = F.interpolate(s1, size=self.img_size, mode='bilinear', align_corners=True)
        s2 = F.interpolate(s2, size=self.img_size, mode='bilinear', align_corners=True)
        s1 = self.scls1(s1)
        s2 = self.scls1(s2)
        return y, s1, s2
    
    def encoder(self, x):   
        x, output_dimensions = self.backbone.patch_embed(x)
        if self.backbone.ape:
            x = x + self.backbone.absolute_pos_embed
        x = self.backbone.pos_drop(x)
        y = x
        L = int(math.sqrt(y.shape[1]))
        y = rearrange(y, 'b (h w) c -> b c h w', h=L, w=L)
        for layer in self.backbone.layers:
            x, output_dimensions = layer(x, output_dimensions)
        L = int(math.sqrt(x.shape[1]))
        x = rearrange(x, 'b (h w) c -> b c h w', h=L, w=L) 
        return y, x

class CIENetTinyViT(nn.Layer):
    def __init__(self, img_size, num_seg=7, sam_checkpoint=r"/home/jq/Code/weights/vit_t.pdparams", **kwargs):
        super(CIENetTinyViT, self).__init__()
        self.sam = MobileSAM(img_size=img_size)
        self.sam.eval()
        if sam_checkpoint is not None:
            load_entire_model(self.sam, sam_checkpoint)
            self.sam.image_encoder.build_abs()

        self.img_size = [img_size, img_size]
        # self.conv = layers.ConvBNReLU(256, 320, 3)
        self.clim = CrossDimensionalFeatureFusion(256, 64, 64)
        
        self.bdia = BitemporalDifferenceInformationAggregation(64, 64)
        self.cls = nn.Sequential(layers.ConvBNAct(64,64,3, act_type='gelu'), 
                                 nn.Conv2D(64, 1, 3, 1, 1))

        self.dmfe = SemanticInformationEnhancement(64)
        self.scls1 = nn.Sequential(layers.ConvBNAct(64,64,3, act_type='gelu'), 
                                   nn.Conv2D(64, num_seg, 3, 1, 1))
    
    def forward(self, x1, x2=None):
        if x2 is None:
            x = paddle.split(x1, 2, axis=1)
            x1 = x[0]
            x2 = x[1]
    
        f1, f2 = self.tinyvit(x1)
        p1, p2 = self.tinyvit(x2)

        f2 = self.clim(f2, f1)
        p2 = self.clim(p2, p1)
 
        y = self.bdia(f2, p2)
        y = F.interpolate(y, size=self.img_size, mode='bilinear', align_corners=True)
        y = self.cls(y)

        s1 = self.dmfe(f2)
        s2 = self.dmfe(p2)
        s1 = F.interpolate(s1, size=self.img_size, mode='bilinear', align_corners=True)
        s2 = F.interpolate(s2, size=self.img_size, mode='bilinear', align_corners=True)
        s1 = self.scls1(s1)
        s2 = self.scls1(s2)
        return y, s1, s2
    
    def tinyvit(self, x):   
        x = self.sam.image_encoder.patch_embed(x)
        x0 = x
        x = self.sam.image_encoder.layers[0](x)
        start_i = 1

        for i in range(start_i, len(self.sam.image_encoder.layers)):
            layer = self.sam.image_encoder.layers[i]
            x = layer(x)
           
        B, wh, C = x.shape
        w = int(math.sqrt(wh))
        x = x.reshape((B, w, w, C))
        x = x.transpose((0, 3, 1, 2))
        x = self.sam.image_encoder.neck(x)
       
        return x0, x

class SemanticInformationEnhancement(nn.Layer):
    def __init__(self, in_c):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2D(in_c, 2*in_c, 1), LayerNorm2d(2*in_c))

        self.gconv1 = nn.Sequential(GhostConv2D(in_c, in_c, 3, 1, 1),
                                   nn.BatchNorm2D(in_c),
                                   nn.ReLU())
        self.dwc1 = layers.ConvBNReLU(in_c, in_c, 3)

        self.mlp2 = nn.Sequential(nn.Conv2D(in_c, in_c, 1),
                                  nn.GELU(),
                                  nn.Conv2D(in_c, in_c, 1))

        self.conv = layers.ConvBNAct(2*in_c, in_c, 1, act_type='gelu')
    
    def forward(self, x):
        y = self.proj(x)
        x1, x2 = paddle.split(y, 2, axis=1)
        y1 = self.gconv1(x1)
        y1 = self.dwc1(y1)

        y2 = F.adaptive_avg_pool2d(x2, 1)
        y2 = self.mlp2(y2)
        y2 = x2 * y2
        z = paddle.concat([y1, y2], axis=1)
        z = self.conv(z)
        return z

class CrossDimensionalFeatureFusion(nn.Layer):   
    def __init__(self, in_ch1, in_ch2, out_chs):
        super().__init__()
        self.proj = layers.ConvBNReLU(in_ch1, out_chs, 3)
        self.proj2 = layers.ConvBNReLU(in_ch2, out_chs, 3)
        self.sa = nn.Sequential(
                    nn.Conv2D(out_chs, in_ch1, kernel_size=1),
                    nn.BatchNorm2D(in_ch1),
                    nn.Conv2D(in_ch1, out_chs, kernel_size=1),
                    nn.AdaptiveAvgPool2D(1),
                    nn.Sigmoid(),
                    )
        self.lamda = paddle.create_parameter(
            shape=[1, out_chs, 1, 1],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(0.5))
        
        self.conv3d = nn.Sequential(nn.Conv3D(2, 2, 3, padding=1), nn.BatchNorm3D(2), nn.ReLU(), 
                                    nn.Conv3D(2, 1, 3, padding=1))
        
        # self.dwc0 = layers.ConvBN(out_chs, out_chs, 1)
        # self.dwc1 = layers.DepthwiseConvBN(out_chs, out_chs, 5)
        # self.dwc2 = layers.DepthwiseConvBN(out_chs, out_chs, 7)
        self.dwc1 = layers.DepthwiseConvBN(out_chs, out_chs, 3, dilation=3)
        self.dwc2 = layers.DepthwiseConvBN(out_chs, out_chs, 3, dilation=5)
       
        self.cbr = layers.ConvBNReLU(out_chs, out_chs, 3)
    
    def forward(self, x, y):
        x = self.proj(x)
        sax = self.sa(x)

        x1 = F.interpolate(x, size=y.shape[-2:], mode='bilinear', align_corners=True)

        y = self.proj2(y)
        x3 = paddle.stack([x1, y], axis=1)
        x3 = self.conv3d(x3)

        x3 = x3.squeeze(1)
        x2 = x1 + self.lamda * x3
        # f = self.dwc0(x2) + self.dwc1(x2) + self.dwc2(x2)
        f = self.dwc1(x2) + self.dwc2(x2)
        f = f * sax
        f = self.cbr(f)
        return f


class BitemporalDifferenceInformationAggregation(nn.Layer):
    def __init__(self, in_c, out_c):
        super().__init__()
        # self.zip = nn.Conv2D(256, in_c, 1)
        self.conv1 = nn.Conv2D(in_c, in_c, 3, padding=1)
        self.dwc = layers.DepthwiseConvBN(in_c, in_c, 3, dilation=3)
        self.conv2 = nn.Conv2D(in_c, in_c, 3, padding=1)
        # self.cbr1 = layers.ConvBNReLU(in_c, in_c, 3)

        self.cbr = layers.ConvBNReLU(in_c, out_c, 3)
    
    def forward(self, x, y):
        # x = self.zip(x)
        # x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        x1 = self.conv1(x)
        y1 = self.conv1(y)
        f = x1 - y1
        f = self.conv2(f)
        x2 = self.dwc(x1)
        y2 = self.dwc(y1)
        # x2 = x2 + f
        # x2 = self.cbr1(x2)
        # y2 = y2 + f
        # y2 = self.cbr1(y2)
        # f = x2+y2
        f = (x2 + y2) * f
        f = self.cbr(f)
        return f

class LayerNorm2d(nn.Layer):
    def __init__(self, num_channels: int, eps: float=1e-06) -> None:
        super().__init__()
        self.weight = paddle.create_parameter(
            shape=[num_channels],
            dtype='float32',
            default_initializer=nn.initializer.Constant(value=1.0))
        self.bias = paddle.create_parameter(
            shape=[num_channels],
            dtype='float32',
            default_initializer=nn.initializer.Constant(value=0.0))
        self.eps = eps

    def forward(self, x: paddle.tensor) -> paddle.tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / paddle.sqrt(s + self.eps)
        x = self.weight[:, (None), (None)] * x + self.bias[:, (None), (None)]
        return x


class GhostConv2D(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.kernel_size = kernel_size
        init_ch = out_channels // 2
        
        self.prim_conv = nn.Sequential(
            nn.Conv2D(in_channels, init_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias_attr=False),
            nn.BatchNorm2D(init_ch),
            nn.ReLU()
        )
        self.cheap_conv = nn.Sequential(
            nn.Conv2D(init_ch, init_ch, kernel_size=kernel_size, stride=stride, padding=padding, groups=init_ch),
            nn.BatchNorm2D(init_ch),
            nn.ReLU()
        )
    def forward(self, x):
        x1 = self.prim_conv(x)
        x2 = self.cheap_conv(x1)
        output = paddle.concat([x1, x2], axis=1)
        
        return output