import paddle
import math
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.models import layers
from paddleseg.utils import load_entire_model
from paddleseg.models.losses import LovaszSoftmaxLoss, BCELoss

from paddleclas.ppcls.arch.backbone.model_zoo.repvgg import RepVGGBlock


from .adsam import build_sam_vit_t
from .modules import CoarseDifferenceFeaturesExtraction
from .modules import FrequencyDomainFeatureEnhance, GlobalFeatureEnhancement

class FFINetVT_BCD(nn.Layer):
    #Frequency-domain Feature Inteaction Network
    def __init__(self, img_size, num_cls=2):
        super().__init__()
        self.img_size = [img_size, img_size]

        self.sam = build_sam_vit_t(img_size=img_size, rank=16)
        
        self.cife = CoarseDifferenceFeaturesExtraction(256)
        self.upc1 = FrequencyDomainFeatureEnhance(256, 128, 64)
        self.cls = nn.Conv2D(64, num_cls, 3, 1, 1)

        for name, param in self.sam.named_parameters():
            if "lora" in name:
                param.stop_gradient = False
     
    def forward(self, x1, x2=None):
        if x2 is None:
            x = paddle.split(x1, 2, axis=1)
            x1 = x[0]
            x2 = x[1]
    
        f, p = self.sam.image_encoder(x1, x2)
        # self.sam.decoder(f)
        
        y = self.cife(f, p)

        y = self.upc1(y)
       
        y = F.interpolate(y, size=self.img_size, mode='bilinear', align_corners=True)
        y = self.cls(y)
       
        return y

class FFINetVT_SCD(FFINetVT_BCD):
    def __init__(self, img_size, num_seg=7):
        super().__init__(img_size=img_size, num_cls=1)

        self.img_size = [img_size, img_size]
        
        self.up = FrequencyDomainFeatureEnhance(256, 128, 64)
        self.scls1 = nn.Conv2D(64, num_seg, 3, 1, 1)
    
    def forward(self, x1, x2=None):
        if x2 is None:
            x = paddle.split(x1, 2, axis=1)
            x1 = x[0]
            x2 = x[1]
    
        f, p = self.sam.image_encoder(x1, x2)

        y = self.cife(f, p)
        y = self.upc1(y)
       
        y = F.interpolate(y, size=self.img_size, mode='bilinear', align_corners=True)
        y = self.cls(y)
        
        s1 = self.up(f)
        s2 = self.up(p)
        s1 = F.interpolate(s1, size=self.img_size, mode='bilinear', align_corners=True)
        s2 = F.interpolate(s2, size=self.img_size, mode='bilinear', align_corners=True)
        s1 = self.scls1(s1)
        s2 = self.scls1(s2)
        return y, s1, s2
        