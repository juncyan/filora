import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.models import layers

from einops import rearrange, repeat
from .utils import FFN

class CoarseDifferenceFeaturesExtraction(nn.Layer):
    def __init__(self, dim=512):
        super().__init__()
        self.conv1 = nn.Conv2D(2*dim, dim, 1)
        self.fourier = Fourier(dim)
        self.ffn = FFN(dim) 
    
    def forward(self, x1, x2):
        x = paddle.concat((x1, x2), axis=1)
        y = self.conv1(x)
        y1 = self.fourier(y)
        y1 = y1 + y
        y2 = self.ffn(y1)
        return y2
        
class FrequencyDomainFeatureEnhance(nn.Layer):
    def __init__(self, inc, dim, outc):
        super().__init__()
        self.conv = nn.Conv2D(inc, dim, 1)
        self.fft = Fourier(dim) 
        self.conv2 = nn.Conv2DTranspose(
                in_channels=dim,
                out_channels=outc,
                kernel_size=2,
                stride=2)
        self.ffn = FFN(outc) 

    def forward(self, x):
        x = self.conv(x)
        y = self.fft(x)
        y = self.conv2(y)
        y = self.ffn(y)
        return y


class Fourier(nn.Layer):
    def __init__(self, in_channels, out_channels = None, groups=1):
        super().__init__()
        self.groups = groups
        if out_channels is None:
            out_channels = in_channels
        self.conv = nn.Conv2D(in_channels=in_channels * 2, 
                            out_channels=out_channels * 2,
                            kernel_size=1, stride=1, padding=0, 
                            groups=self.groups, bias_attr=False,
                            weight_attr=nn.initializer.Constant(1.0))
        # self.bn = nn.BatchNorm2D(out_channels * 2)
        self.relu = nn.GELU()

        self.idea = layers.ConvBN(in_channels, out_channels, 1)

    def forward(self, x):
        b, _, h, w = x.shape

        avg = F.adaptive_avg_pool2d(x, (1, 1))
        avg = self.idea(avg) * x

        # (batch, c, h, w//2 + 1, 2)
        ffted = paddle.fft.rfft2(x, norm='ortho')
        x_fft_real = paddle.unsqueeze(ffted.real(), axis=-1)
        x_fft_imag = paddle.unsqueeze(ffted.imag(), axis=-1)
        ffted = paddle.concat((x_fft_real, x_fft_imag), axis=-1)
        
        # (batch, c, 2, h, w//2 + 1)
        ffted = ffted.transpose([0, 1, 4, 2, 3])
        ffted = ffted.reshape([b, -1,] + ffted.shape[3:])

        ffted = self.conv(ffted)  # (batch, c*2, h, w//2 + 1)
        # ffted = self.relu(self.bn(ffted))

        ffted = ffted.reshape([b, -1, 2,] + ffted.shape[2:]).transpose(
            [0, 1, 3, 4, 2])  # (batch, c, h, w//2 + 1, 2)
        ffted = paddle.complex(ffted[..., 0], ffted[..., 1])

        output = paddle.fft.irfft2(ffted, s=(h, w), norm='ortho')
        output = output + avg
        output = self.relu(output)
        return output

class SEBlock(nn.Layer):
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2D(
            in_channels=input_channels,
            out_channels=internal_neurons,
            kernel_size=1,
            stride=1,
            bias_attr=True)
        self.up = nn.Conv2D(
            in_channels=internal_neurons,
            out_channels=input_channels,
            kernel_size=1,
            stride=1,
            bias_attr=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.shape[3])
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = F.sigmoid(x)
        x = x.reshape([-1, self.input_channels, 1, 1])
        return inputs * x


class MFM(nn.Layer):
    def __init__(self, dim, height=2, reduction=8):
        super(MFM, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.mlp = nn.Sequential(
            nn.Conv2D(dim, d, 1, bias_attr=False),
            nn.ReLU(),
            nn.Conv2D(d, dim * height, 1, bias_attr=False)
        )

        self.softmax = nn.Softmax(axis=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = paddle.concat(in_feats, axis=1)
        in_feats = in_feats.reshape([B, self.height, C, H, W])

        feats_sum = paddle.sum(in_feats, axis=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.reshape([B, self.height, C, 1, 1]))

        out = paddle.sum(in_feats * attn, axis=1)
        return out


class Local_Feature_Gather(nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dim_sp = dim // 2

        self.CDilated_1 = nn.Conv2D(self.dim_sp, self.dim_sp, 3, stride=1, padding=1, dilation=1, groups=self.dim_sp)
        self.CDilated_2 = nn.Conv2D(self.dim_sp, self.dim_sp, 3, stride=1, padding=2, dilation=2, groups=self.dim_sp)
        self.CK1 = nn.Conv2D(dim, dim, 1)
        # self.ln = nn.BatchNorm2D(dim)
        # self.act = nn.Silu()

    def forward(self, x):
        x1, x2 = paddle.chunk(x, 2, axis=1)
        # x1 = x[:, 0::2, :, :]
        # x2 = x[:, 1::2, :, :]
        cd1 = self.CDilated_1(x1)
        cd2 = self.CDilated_2(x2)
        y = paddle.concat([cd1, cd2], axis=1)
        y = y + self.CK1(x)
        # y = self.ln(y)
        # y = self.act(y)
        return y
    

class GlobalFeatureEnhancement(nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.conv_init = nn.Sequential(
            nn.Conv2D(dim, dim, 1),
            nn.GELU())
        
        self.token = Local_Feature_Gather(self.dim)
        self.conv_fina = nn.Sequential(
            nn.Conv2D(dim, dim, 1),
            nn.GELU()) 
       
    def forward(self, x):
        x = self.conv_init(x)
        x0 = x
        x = self.token(x)
        x = x + x0
        x = self.conv_fina(x)
        return x