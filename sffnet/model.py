import paddle
import math
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.models import layers
from paddleseg.utils import load_entire_model
from paddleseg.models.losses import LovaszSoftmaxLoss, BCELoss

from paddleclas.ppcls.arch.backbone.model_zoo.repvgg import RepVGGBlock

from einops import rearrange, repeat

from .sam import MSAM
from .modules import BitemporalChangeInformationFusion, MultiScaleFeatureAggregation, Local_Feature_Gather

class SFFNet_BCD(nn.Layer):
    #SSM Feature Inteaction Network
    def __init__(self, img_size, num_cls=2, sam_checkpoint=r"/home/jq/Code/weights/vit_t.pdparams"):
        super().__init__()
        self.img_size = [img_size, img_size]

        self.sam = MSAM(img_size=img_size)
        self.sam.image_encoder.build_abs()
        self.sam.eval()
        if sam_checkpoint is not None:
            load_entire_model(self.sam, sam_checkpoint)
        
        self.cife = BitemporalChangeInformationFusion(320, 128)
        self.conv = layers.ConvBNAct(128, 128, 1, padding=1, act_type='gelu')

        self.upc = MultiScaleFeatureAggregation(128)
        self.cls = nn.Sequential(layers.ConvBNAct(128, 64, 1, act_type='gelu'),
                                  nn.Conv2D(64, num_cls, 3, 1, 1))

        for name, param in self.sam.named_parameters():
            if "adaptor" in name:
                param.stop_gradient = False
      
    def forward(self, x1, x2=None):
        if x2 is None:
            x = paddle.split(x1, 2, axis=1)
            x1 = x[0]
            x2 = x[1]
    
        f, p = self.sam.image_encoder(x1, x2)
        # self.sam.decoder(f)
        
        y = self.cife(f, p)

        B, H, C = y.shape
        w = int(math.sqrt(H))
        y = y.reshape((B, w, w, -1))
        y = y.transpose((0, 3, 1, 2))
        y = self.conv(y)
        y = self.upc(y)
        
        y = F.interpolate(y, size=self.img_size, mode='bilinear', align_corners=True)
        y = self.cls(y)
       
        return y
    

class SFFNet_SCD(SFFNet_BCD):
    def __init__(self, img_size, num_seg=7, sam_checkpoint=r"/home/jq/Code/weights/vit_t.pdparams"):
        super().__init__(img_size=img_size, num_cls=1, sam_checkpoint=sam_checkpoint)

        self.img_size = [img_size, img_size]
        
        self.up = MultiScaleFeatureAggregation(256)
        self.scls1 = nn.Sequential(layers.ConvGNAct(256, 64, 1, act_type='gelu'),
                                      nn.Conv2D(64, num_seg, 3, 1, 1))
    
    def forward(self, x1, x2=None):
        if x2 is None:
            x = paddle.split(x1, 2, axis=1)
            x1 = x[0]
            x2 = x[1]
    
        f, p = self.sam.image_encoder(x1, x2)

        y = self.cife(f, p)

        B, H, C = y.shape
        w = int(math.sqrt(H))
        y = y.reshape((B, w, w, -1))
        y = y.transpose((0, 3, 1, 2))

        y = self.conv(y)
        y = self.upc(y)
        
        y = F.interpolate(y, size=self.img_size, mode='bilinear', align_corners=True)
        y = self.cls(y)
        
        s1 = f.reshape((B, w, w, -1))
        s1 = s1.transpose((0, 3, 1, 2))
        s1 = self.sam.image_encoder.neck(s1)
        s1 = self.up(s1)

        s2 = p.reshape((B, w, w, -1))
        s2 = s2.transpose((0, 3, 1, 2))
        s2 = self.sam.image_encoder.neck(s2)
        s2 = self.up(s2)

        s1 = F.interpolate(s1, size=self.img_size, mode='bilinear', align_corners=True)
        s2 = F.interpolate(s2, size=self.img_size, mode='bilinear', align_corners=True)
        s1 = self.scls1(s1)
        s2 = self.scls1(s2)
        return y, s1, s2
        
