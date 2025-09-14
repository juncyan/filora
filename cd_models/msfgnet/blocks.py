import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddleseg.models.layers as layers

class BMF(nn.Layer):
    #Bitemporal Image Multi-level Fusion Module
    def __init__(self, in_channels, out_channels=64):
        super().__init__()

        self.cbr1 = layers.ConvBNReLU(in_channels, 32, 3)
        self.cbr2 = layers.ConvBNReLU(in_channels, 32, 3)

        self.cond1 = nn.Conv2D(64, 64, 3, padding=1)
        self.cond3 = nn.Conv2D(64, 64, 3, padding=3, dilation=3)
        self.cond5 = nn.Conv2D(64, 64, 3, padding=5, dilation=5)

        self.bn = nn.BatchNorm2D(64)
        self.relu = nn.ReLU()

        self.shift = layers.ConvBNReLU(64, out_channels, 3, 1, stride=2)

    def forward(self, x1, x2):
        y1 = self.cbr1(x1)
        y2 = self.cbr2(x2)

        y = paddle.concat([y1, y2], 1)

        y10 = self.cond1(y)
        y11 = self.cond3(y)
        y12 = self.cond5(y)
       
        yc = self.relu(self.bn(y10 + y11 + y12))
        return self.shift(yc)



class LRFE(nn.Layer):
    #Large receptive field fusion
    def __init__(self, in_channels, dw_channels, block_lk_size, stride=1):
        super().__init__()
        self.cbr1 = layers.ConvBNReLU(in_channels, dw_channels, 3, stride=1)
        
        self.dec = layers.DepthwiseConvBN(dw_channels, dw_channels, block_lk_size, stride=stride)
        self.gelu = nn.GELU()

        self.c2 = nn.Conv2D(dw_channels, in_channels, 1, stride=1)
        self.bn = nn.BatchNorm2D(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.cbr1(x)
        y = self.dec(y)
        y = self.gelu(y)
        y = self.c2(y)
        return self.relu(self.bn(x + y))

class MSIF(nn.Layer):
    #multi-scale information fusion
    def __init__(self, in_channels, internal_channels):
        super().__init__()
        self.cbr1 = layers.ConvBNReLU(in_channels, internal_channels, 1)

        self.cond1 = nn.Conv2D(internal_channels, internal_channels, 1)
        self.cond3 = nn.Conv2D(internal_channels, internal_channels, 3, padding=3, dilation=3, groups=internal_channels)
        self.cond5 = nn.Conv2D(internal_channels, internal_channels, 3, padding=5, dilation=5, groups=internal_channels)

        self.bn1 = nn.BatchNorm2D(internal_channels)
        self.relu1 = nn.ReLU()

        self.cbr2 = layers.ConvBNReLU(internal_channels, in_channels, 1)
        
        self.lastbn = nn.BatchNorm2D(in_channels)
        self.relu = nn.ReLU()
        

    def forward(self, x):
        y = self.cbr1(x)
        y1 = self.cond1(y)
        y2 = self.cond3(y)
        y3 = self.cond5(y)
        y = self.relu1(self.bn1(y1 + y2 + y3))
        y = self.cbr2(y)
        return self.relu(self.lastbn(x + y))