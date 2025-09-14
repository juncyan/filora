import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.models import layers
# from paddleseg.models.backbones import ResNet34_vd
from models.backbone.replknet import RepLKNet31B
from models.backbone.resnet import ResbackBone
from models.backbone.swin_transformer import SwinTransBackbone
from models.backbone.vit import ViTB_patch16_512

from .lkblocks import *
from .utils import *


class LKPSNet_k27(nn.Layer):
    #large kernel pseudo siamese network
    def __init__(self, in_channels=3, kernels=27):
        super().__init__()

        self.fa = PSAA([64, 128, 256, 512], kernels)

        self.stage1 = STAF(in_channels, 64, kernels)
        self.stage2 = BFELKB(64, 128, kernels)
        self.stage3 = BFELKB(128, 256, kernels)
        self.stage4 = BFELKB(256, 512, kernels)
        
        self.cbr1 = MF(128,64)
        self.cbr2 = MF(256,128)
        self.cbr3 = MF(512,256)
        self.cbr4 = MF(1024,512)

        self.up1 = UpBlock(512+256, 256)
        self.up2 = UpBlock(256+128, 128)
        self.up3 = UpBlock(128+64, 64)

        self.classiier = layers.ConvBNAct(64, 2, 7, act_type="sigmoid")
    
    def forward(self, x):
        x1, x2 = x[:, :3, :, :], x[:, 3:, :, :]
        _, _, w, h = x1.shape
        a1, a2, a3, a4 = self.fa(x1, x2)

        f1 = self.stage1(x1, x2)
        m1 = self.cbr1(f1, a1)
        f2 = self.stage2(m1)
        m2 = self.cbr2(f2, a2)
        f3 = self.stage3(m2)
        m3 = self.cbr3(f3, a3)
        f4 = self.stage4(m3)
        m4 = self.cbr4(f4, a4)
        
        r1 = self.up1(m4, m3)
        r2 = self.up2(r1, m2)
        r3 = self.up3(r2, m1)

        y = F.interpolate(r3, size=[w, h],mode='bilinear')
        y = self.classiier(y)

        return y #, l1
    


