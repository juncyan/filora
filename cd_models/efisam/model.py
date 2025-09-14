import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from typing import Type
import numpy as np
import math

from paddleseg.models import layers
from paddleseg.models.losses import LovaszSoftmaxLoss, BCELoss
from paddleseg.utils import load_entire_model

from .mobilesam import MobileSAM
from .attention import ECA, RFFA


features_shape = {256:np.array([[1024, 128],[256,160],[256,320],[256,320]]),
                  512:np.array([[4096, 128],[1024,160],[1024,320],[1024,320]]),
                  1024:np.array([[16384, 128],[4096,160],[4096,320],[4096,320]])}

def features_transfer(x):
        x = x.transpose((0, 2, 1))
        B, C, S = x.shape
        wh = int(math.sqrt(S))
        x = x.reshape((B, C, wh, wh))
        return x

class EFISam(nn.Layer):
#     @ARTICLE{11058393,
#   author={Huang, Junqing and Bao, Junqi and Xia, Min and Yuan, Xiaochen},
#   journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
#   title={SAM-Based Efficient Feature Integration Network for Remote Sensing Change Detection: A Case Study on Macao Sea Reclamation}, 
#   year={2025},
#   volume={18},
#   number={},
#   pages={16916-16928},
#   keywords={Feature extraction;Remote sensing;Land surface;Transformers;Semantics;Data mining;Accuracy;Visualization;Decoding;Computational modeling;Random Fourier features;remote sensing change detection (RSCD);sea reclamation;segment anything model (SAM)},
#   doi={10.1109/JSTARS.2025.3584145}}
    def __init__(self, img_size=256,sam_checkpoint=r"/home/jq/Code/weights/vit_t.pdparams"):
        super().__init__()
        self.sam = MobileSAM(img_size=img_size)
        self.sam.eval()
        if sam_checkpoint is not None:
            load_entire_model(self.sam, sam_checkpoint)
            self.sam.image_encoder.build_abs()
        
        self.bff1 = LSFI(128,64)
        self.bff3 = LSFI(320,64)
        self.bff4 = LSFI(320,64)
        self.sfa4 = RFFA(320,320)
        self.fusion = CCIA(64)
        

        self.cls1 = layers.ConvBN(64,2,7)

    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1[:,3:,:,:]
            x1 = x1[:,:3,:,:]
        
        f1, f3, f4 = self.extractor(x1, x2)
        f = self.fusion(f1, f3, f4)
        f = self.cls1(f)

        return f
  
    def feature_extractor(self, x):
        x = self.sam.image_encoder.patch_embed(x)
        y0 = self.sam.image_encoder.layers[0](x)
        y1 = self.sam.image_encoder.layers[1](y0)
        y2 = self.sam.image_encoder.layers[2](y1)
        y3 = self.sam.image_encoder.layers[3](y2)
        y3 = self.sfa4(y3) + y3
        
        return y0, y1, y2, y3
    
    
    def extractor(self, x1, x2):
        b1, b2, b3, b4 = self.feature_extractor(x1)
        p1, p2, p3, p4 = self.feature_extractor(x2)
        # print(b1.shape, b2.shape, b3.shape, b4.shape)

        f1 = self.bff1(b1, p1)
        f3 = self.bff3(b3, p3)
        f4 = self.bff4(b4, p4)
        f1 = features_transfer(f1)
        f3 = features_transfer(f3)
        f4 = features_transfer(f4)
        return f1, f3, f4
    
    @staticmethod
    def predict(pred):
        return pred
    
    @staticmethod
    def loss(pred, label):  
        l1 = BCELoss()(pred, label)
        label = paddle.argmax(label, axis=1)
        l2 = LovaszSoftmaxLoss()(pred, label)
        loss = l1 + 0.75*l2
        return loss

class LSFI(nn.Layer):
    def __init__(self, dims, out_channels=64):
        super().__init__()
        self.cov1 = nn.Conv1D(2*dims, out_channels,3,padding=1,data_format='NLC')
        self.bn1 = nn.BatchNorm1D(out_channels, data_format='NLC')
        self.mlp1 = MLPBlock(out_channels, out_channels*2)

        self.lamda = self.create_parameter(shape=[1], default_initializer=nn.initializer.Constant(1.0), dtype='float16')
        self.fc = nn.Linear(2,1)
        self.mlp = MLPBlock(out_channels, out_channels)

    def forward(self, x1, x2):
        x = paddle.concat([x1, x2], -1)
        x = self.cov1(x)
        x = self.bn1(x)
        x = self.mlp1(x)

        xa = F.adaptive_avg_pool1d(x, 1)
        xm = F.adaptive_max_pool1d(x, 1)
        xt = paddle.concat([xa, xm], -1)
        xt = self.fc(xt)
        xt = F.relu(xt)
        y = x * xt
        y = y* self.lamda + x
        y = self.mlp(y)
        y = F.relu(y)
        return y

class ResBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = layers.ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=stride)
        self.conv2 = layers.ConvBN(out_channels, out_channels, kernel_size=3)
        
        self.shortcut = layers.ConvBN(in_channels, out_channels, kernel_size=3, stride=stride)
        

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        return F.relu(x)

class CCIA(nn.Layer):
    def __init__(self, in_channels=64, out_dims=64):
        super().__init__()
        dims = 64 
        self.st1conv1 = layers.ConvBNReLU(dims, in_channels, 1)
        self.st1conv2 = EFC(in_channels)

        self.st2conv1 = layers.ConvBNReLU(in_channels, in_channels, 1)
        self.st2conv2 = EFC(in_channels)
        
        self.st3conv1 = layers.ConvBNReLU(in_channels, in_channels, 1)
        self.st3conv2 = EFC(in_channels)
        
        self.conv1 = layers.ConvBNReLU(3*in_channels, in_channels, 1)
        self.conv2 = layers.ConvBNReLU(in_channels, in_channels, 3)
        self.deconv6 = layers.ConvBNReLU(in_channels,out_dims,1)

    def forward(self, x1, x2, x3):
        f3 = self.st1conv1(x3)
        f3 = self.st1conv2(f3)

        f2 = x2 + f3
        f2 = self.st2conv1(f2)
        f2 = self.st2conv2(f2)

        f3 = F.interpolate(f3, x1.shape[-2:], mode='bilinear', align_corners=True)
        f2 = F.interpolate(f2, x1.shape[-2:], mode='bilinear', align_corners=True)
        # print(x1.shape, f2.shape, f3.shape)
        f1 = x1 + f2 + f3
        f1 = self.st3conv1(f1)
        f1 = self.st3conv2(f1)

        f = paddle.concat([f1, f2, f3], 1)
        f = self.conv1(f)
        fd = self.conv2(f)
        f = f + fd
        f = F.interpolate(f, scale_factor=8, mode='bilinear', align_corners=True)
        f = self.deconv6(f)
        return f


class EFC(nn.Layer):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.eca = ECA(kernel_size)
        self.cbr = layers.ConvBNReLU(in_channels, in_channels, kernel_size=kernel_size)
    def forward(self, x):
        y = self.eca(x)
        y = self.cbr(y)
        return y


class MLPBlock(nn.Layer):
    def __init__(self,
                 embedding_dim: int,
                 mlp_dim: int,
                 act: Type[nn.Layer]=nn.GELU) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: paddle.tensor) -> paddle.tensor:
        return self.lin2(self.act(self.lin1(x)))