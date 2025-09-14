import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddleseg.models.layers as layers


class MF(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cbr = layers.ConvBNReLU(in_channels, out_channels, 3)
    
    def forward(self, x1, x2):
        x = paddle.concat([x1, x2], 1)
        y = self.cbr(x)
        return y

class UpBlock(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__() 
        mid_c = out_channels * 2
        self.cbr1 = layers.ConvBNReLU(in_channels, out_channels, 3)
        # self.cbr2 = layers.ConvBNReLU(out_channels, mid_c, 1)
        self.cbr3 = layers.ConvBNReLU(out_channels, out_channels, 3)

    def forward(self, x1, x2):
        if x1.shape != x2.shape:
            x1 = F.interpolate(x1,paddle.shape(x2)[2:],mode='bilinear')
        x = paddle.concat([x1, x2], axis=1)
        y = self.cbr1(x) 
        # y = self.cbr2(y)
        res = self.cbr3(y)
        return res

class GAM(nn.Layer):
    def __init__(self, channels):
        super().__init__()

        self.max = nn.AdaptiveMaxPool2D(1)
        self.mean = nn.AdaptiveAvgPool2D(1)
        self.bn = nn.BatchNorm2D(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        m1 = self.max(x)
        m2 = self.mean(x)
        a = m1 + m2
        y = a * x
        y = self.relu(self.bn(y))

        return y


class SEModule(nn.Layer):
    def __init__(self, channels, reductions = 8):
        super(SEModule, self).__init__()
        reduction_channels = channels // reductions
        self.avg = nn.AdaptiveAvgPool2D(1)
        self.fc1 = nn.Conv2D(channels, reduction_channels, 1, bias_attr=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2D(reduction_channels, channels, 1, bias_attr=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # b, c , _, _ = x.shape
        avg = self.avg(x)
        y = self.fc1(avg)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sig(y)
        y = y.expand_as(x)
        return x * y

