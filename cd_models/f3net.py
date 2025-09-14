import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddleseg.models.layers as layers

class F3Net(nn.Layer):
    def __init__(self,in_channels=6, num_classes=2):
        super().__init__()
        kernels = 7
        self.in_channels = in_channels
        self.encode1 = backbone(3)#(3, 34)
        # self.encode2 = backbone(3)

        self.lkff1 = FFA(64,kernels)
        self.lkff2 = FFA(128,kernels)
        self.lkff3 = FFA(256,kernels)
        self.lkff4 = FFA(512,kernels)

        self.ppm = layers.attention.PAM(512)

        self.up1 = CFDF(1024, 256, kernels)
        self.up2 = CFDF(512, 128, kernels)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)

        self.classier = nn.Sequential(layers.ConvBN(64, num_classes,7), nn.Sigmoid())

    def forward(self, x):
        x1 ,x2 = x[:, :self.in_channels//2, :, :], x[:, self.in_channels//2:, :, :]
        self.feature1 = self.encode1(x1)
        self.feature2 = self.encode1(x2)

        # for i in self.feature1:
        #     print(i.shape)
            
        self.augf1 = self.lkff1(self.feature1[0], self.feature2[0])
        self.augf2 = self.lkff2(self.feature1[1], self.feature2[1])
        self.augf3 = self.lkff3(self.feature1[2], self.feature2[2])
        self.augf4 = self.lkff4(self.feature1[3], self.feature2[3])

        self.flast = self.ppm(self.augf4)

        y = self.up1(self.flast, self.augf4)
        y = self.up2(y, self.augf3)
        y = self.up3(y, self.augf2)
        y = self.up4(y, self.augf1)
        # y = nn.functional.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        return self.classier(y)


class FFA(nn.Layer):
    # feature fusing and aggregation
    def __init__(self, in_channels, kernels=7):
        super().__init__()
        dims = in_channels * 2
        self.zip_channels = layers.ConvBNReLU(dims, in_channels, 1)
        self.lfc = layers.DepthwiseConvBN(in_channels, in_channels, kernels)
    
        self.sa = layers.ConvBNAct(2, 1, 3, act_type='sigmoid')

        self.outcbn = layers.ConvBNReLU(in_channels, in_channels, 3)

    def forward(self, x1, x2):
        x = paddle.concat([x1, x2], 1)
        x = self.zip_channels(x)
        y = self.lfc(x)
        y = nn.GELU()(y)
        
        max_feature = paddle.max(y, axis=1, keepdim=True)
        mean_feature = paddle.mean(y, axis=1, keepdim=True)
        
        att_feature = paddle.concat([max_feature, mean_feature], axis=1)
        # y1 = max_feature * y
        # y2 = mean_feature * y
        # att_feature = paddle.concat([y1, y2], axis=1)
        y = self.sa(att_feature)
        y = y * x
        y = self.outcbn(y)
        return y


class CISConv(nn.Layer):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=3, groups=1, dilation_set=4,
                 bias=False):
        super().__init__()
        self.prim = nn.Conv2D(in_ch, out_ch, kernel_size, stride, padding=dilation, dilation=dilation,
                              groups=groups * dilation_set)
        self.prim_shift = nn.Conv2D(in_ch, out_ch, kernel_size, stride, padding=2 * dilation, dilation=2 * dilation,
                                    groups=groups * dilation_set)
        self.conv = nn.Conv2D(in_ch, out_ch, kernel_size, stride, padding, groups=groups)
        self.groups = groups

    def forward(self, x):
        x_split = (z.chunk(2, axis=1) for z in x.chunk(self.groups, axis=1))
        x_merge = paddle.concat(tuple(paddle.concat((x2, x1), 1) for (x1, x2) in x_split), 1)
        x_shift = self.prim_shift(x_merge)
        return self.prim(x) + self.conv(x) + x_shift


class CFDF(nn.Layer):
    def __init__(self, in_channels, out_channels, kernels=7):
        super().__init__()
        self.inter_dim = in_channels // 2
        self.zip_ch = layers.ConvBNReLU(in_channels, self.inter_dim, 3)

        self.native = nn.Sequential(layers.DepthwiseConvBN(self.inter_dim, self.inter_dim,kernels,3),
                                    nn.GELU(),
            layers.ConvBNReLU(self.inter_dim, self.inter_dim//2, 1))

        self.aux = nn.Sequential(
            CISConv(self.inter_dim//2, self.inter_dim//2, 3, 1, padding=1, groups=int(self.inter_dim / 8), dilation=1,bias=False),
            nn.BatchNorm2D(self.inter_dim//2),
            nn.ReLU())

        self.outcbr = layers.ConvBNReLU(self.inter_dim, out_channels, 3, 1)

    def forward(self, x1, x2):
        if x1.shape != x2.shape:
            x1 = F.interpolate(x1, x2.shape[-2:], mode='bilinear')

        x = paddle.concat([x2, x1], axis=1)
        y = self.zip_ch(x)
        y1 = self.native(y)
        y2 = self.aux(y1)
        # y1 = self.ppmn(y1)
        y = paddle.concat([y1, y2], 1)
        return self.outcbr(y)

class BTF(nn.Layer):
    def __init__(self, in_channels):
        super().__init__()
        dims = in_channels * 2
        self.zip_channels = layers.ConvBNReLU(dims, in_channels, 3)
    
    def forward(self, x1, x2):
        x = paddle.concat([x1,x2], 1)
        y = self.zip_channels(x)
        return y
        
class DoubleConv(nn.Layer):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2D(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2D(mid_channels),
            nn.ReLU(),
            nn.Conv2D(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2D(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Layer):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        self.bilinear = bilinear
        if bilinear:
            #self.up = nn.functional.interpolate(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.Conv2DTranspose(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        if(self.bilinear):
            x1 =nn.functional.interpolate(x1,scale_factor=2, mode='bilinear', align_corners=True)
        else:
            x1 = self.up(x1)
        # print("x2.size():", x2.shape)
        diffY = x2.shape[2] - x1.shape[2]
        diffX = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = paddle.concat([x2, x1], axis=1)
        return self.conv(x)

class Down(nn.Layer):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            #nn.MaxPool2d(2),
            nn.Conv2D(in_channels,in_channels,3,2,1),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class backbone(nn.Layer):
    def __init__(self, in_channels):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        # factor = 2 #if bilinear else 1
        # self.down4 = Down(512, 512)
        
    def forward(self, x):
        res = []
        y = self.inc(x)
        res.append(y)
        y = self.down1(y)
        res.append(y)
        y = self.down2(y)
        res.append(y)
        y = self.down3(y)
        # res.append(y)
        # y = self.down4(y)
        res.append(y)
        return res

if __name__ == "__main__":
    print("spp")
    