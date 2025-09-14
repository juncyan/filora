import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddleseg.models.layers as layers
from .cblkb import RepConv, RepC3

class BFELKB(nn.Layer):
    #bi-temporal feature integrating block
    def __init__(self, in_channels, out_channels, kernels = 7, stride=2):
        super().__init__()
        self.fe = LKFE(in_channels, kernels)
        self.ce = LKCE(in_channels, out_channels, kernels, stride)
        
    def forward(self, x):
        y = self.fe(x)
        
        y = self.ce(y)
        return y

class LKFE(nn.Layer):
    #large kernel feature extraction
    def __init__(self, in_channels, kernels = 7):
        super().__init__()
        self.conv1 = layers.ConvBNReLU(in_channels, 2 * in_channels, 3,)
        self.dwc = CBLKBlock(2*in_channels, kernels)
        self.conv2 = layers.ConvBNReLU(2 * in_channels, in_channels, 3,)
        self.ba = nn.Sequential(nn.BatchNorm2D(in_channels), nn.ReLU())

    def forward(self, x):
        m = self.conv1(x)
        m = self.dwc(m)
        m = self.conv2(m)
        y = x + m
        return self.ba(y)
    
class LKCE(nn.Layer):
    #large kernel channel expansion
    def __init__(self, in_channels, out_channels, kernels = 7, stride=1):
        super().__init__()
        self.conv1 = layers.ConvBNReLU(in_channels, 2*in_channels, 3, stride=stride)
        self.dwc = CBLKBlock(2*in_channels, kernels)
        self.conv3 = layers.ConvBNReLU(2*in_channels, out_channels, 3)

    def forward(self, x):
        y = self.conv1(x)
        m = self.dwc(y)
        m = self.conv3(m)
        return m

class CBLKBlock(nn.Layer):
    def __init__(self, in_channels, kernels=7):
        super().__init__()
        # print(mid_c, in_channels)
        self.c1 = nn.Conv2D(in_channels, in_channels, 1)
        self.bn1 = nn.BatchNorm2D(in_channels)

        self.lkcs = RepConv(in_channels, 9, 4)#nn.Conv2D(in_channels, in_channels, kernels, padding=kernels//2, groups=in_channels)
        self.bn = nn.BatchNorm2D(in_channels)
        self.gelu = nn.GELU()

        self.cbr1 = layers.ConvBNReLU(in_channels, in_channels, 3)
        # self.br = nn.ReLU()#nn.Sequential(nn.BatchNorm2D(in_channels), nn.ReLU())
        
    def forward(self, x):
        y1 = self.c1(x)
        y1 = self.bn1(y1)
        y2 = self.lkcs(x)
        y = y1 + y2
        my = self.gelu(self.bn(y))
        y = self.cbr1(my)
        # y = self.br(y)
        return y

class FEAA(nn.Layer):
    #assimilating assistant feature extraction
    def __init__(self, in_channels, out_channels, kernels=7):
        super().__init__()
        self.cbr1 = layers.ConvBNReLU(in_channels, out_channels, 3, 1, stride=2)
        
        # CBLKB ------------------------------------------------------
        self.c11 = nn.Conv2D(out_channels, out_channels, 1)
        self.gck = nn.Conv2D(out_channels, out_channels, kernels, 1, kernels//2, groups=out_channels)
        
        self.bnr = nn.Sequential(nn.BatchNorm2D(out_channels), nn.GELU())
        self.lastcbr = layers.ConvBNReLU(out_channels, out_channels, 3)
        # CBLKB ------------------------------------------------------
        # self.cblk = CBLKBlock(out_channels)
        
    def forward(self, x):
        z = self.cbr1(x)
        z1 = self.c11(z)
        z2 = self.gck(z)
        y = z1 + z2
        return self.lastcbr(self.bnr(y))
        # z1 = self.cblk(z)
        return z1


class FEBranch(nn.Layer):
    def __init__(self, in_channels, mid_channels: list = [16, 32, 64, 128], kernels=7):
        super(FEBranch, self).__init__()
        self.layers = nn.LayerList()
        in_channels = 3
        for c in mid_channels:
            self.layers.append(FEAA(in_channels, c, kernels))
            in_channels = c

    def forward(self, x):
        y = x
        res = []
        for layer in self.layers:
            y = layer(y)
            res.append(y)
        return res
    
class PSAA(nn.Layer):
    #pseudo siamese assimilating assistant module
    def __init__(self, mid_channels=[64, 128, 256, 512], kernels=7):
        super().__init__()
        self.branch1 = FEBranch(3, mid_channels, kernels)
        self.branch2 = FEBranch(3, mid_channels, kernels)

    def forward(self, x1, x2):
        # x1, x2 = x[:, :3, :, :], x[:, 3:, :, :]
        y1 = self.branch1(x1)
        y2 = self.branch2(x2)
        res = []
        for i, j in zip(y1, y2):
            z = i + j
            res.append(z)
        return res

class FAA(nn.Layer):
    #siamese features assimilating assistant module
    def __init__(self, mid_channels=[64, 128, 256, 512], kernels=7):
        super().__init__()
        self.branch1 = FEBranch(3, mid_channels, kernels)
        # self.branch2 = FEBranch(3, mid_channels, kernels)

    def forward(self, x1, x2):
        # x1, x2 = x[:, :3, :, :], x[:, 3:, :, :]
        y1 = self.branch1(x1)
        y2 = self.branch1(x2)
        res = []
        for i, j in zip(y1, y2):
            z = i + j
            res.append(z)
        return res

class STAF(nn.Layer):
    #Spatial and Temporal Adaptive Fusion Module
    def __init__(self, in_channels=3, out_channels=64, kernels=7):
        super().__init__()

        self.conv1 = layers.DepthwiseConvBN(in_channels, in_channels, kernels)
        self.conv2 = layers.DepthwiseConvBN(in_channels, in_channels, kernels)
        self.cbr1 = layers.ConvBNReLU(2*in_channels, out_channels, 3, stride=2)
        self.dws = layers.ConvBNReLU(out_channels, out_channels, 3)
        self.cbr2 = layers.ConvBNReLU(out_channels, out_channels, 3)
        
        self.tdcbrs2 = layers.ConvBNReLU(2*in_channels, out_channels, 1, stride=2)
        self.tdc11 = nn.Conv2D(out_channels, out_channels, 1, 1)
        self.tddsc = nn.Conv2D(out_channels, out_channels, 7, 1, 3, groups=out_channels)
        self.tdcbr2 = layers.ConvBNReLU(out_channels, out_channels, 3)

    def forward(self, x1, x2):
        y1 = self.conv1(x1)
        y2 = self.conv2(x2)
        ym = paddle.concat([y1, y2], 1)
        ym = self.cbr1(ym)
        y = self.dws(ym)
        y = self.cbr2(y)

        Td = paddle.concat([x1, x2],1)
        td = self.tdcbrs2(Td)
        td1 = self.tdc11(td)
        td2 = self.tddsc(td)
        tc = td1 + td2
        td = self.tdcbr2(tc)
        res = y + td
        return res


class PAM(nn.Layer):
    """
    Position attention module.
    Args:
        in_channels (int): The number of input channels.
    """

    def __init__(self, in_channels):
        super().__init__()
        mid_channels = in_channels // 8
        self.mid_channels = mid_channels
        self.in_channels = in_channels

        self.query_conv = nn.Conv2D(in_channels, mid_channels, 1, 1)
        self.key_conv = nn.Conv2D(in_channels, mid_channels, 1, 1)
        self.value_conv = nn.Conv2D(in_channels, in_channels, 1, 1)

        self.gamma = self.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=nn.initializer.Constant(0))

    def forward(self, x):
        x_shape = paddle.shape(x)

        # query: n, h * w, c1
        query = self.query_conv(x)
        query = paddle.reshape(query, (0, self.mid_channels, -1))
        query = paddle.transpose(query, (0, 2, 1))

        # key: n, c1, h * w
        key = self.key_conv(x)
        key = paddle.reshape(key, (0, self.mid_channels, -1))

        # sim: n, h * w, h * w
        sim = paddle.bmm(query, key)
        sim = F.softmax(sim, axis=-1)

        value = self.value_conv(x)
        value = paddle.reshape(value, (0, self.in_channels, -1))
        sim = paddle.transpose(sim, (0, 2, 1))

        # feat: from (n, c2, h * w) -> (n, c2, h, w)
        feat = paddle.bmm(value, sim)
        feat = paddle.reshape(feat,
                              (0, self.in_channels, x_shape[2], x_shape[3]))

        out = self.gamma * feat + x
        return out