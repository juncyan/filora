import typing
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class LayerNorm2d(nn.Layer):
    def __init__(self, num_features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = self.create_parameter(shape=[num_features], default_initializer=nn.initializer.Constant(1.0))
        self.bias = self.create_parameter(shape=[num_features], default_initializer=nn.initializer.Constant(0.0))
        self.eps = eps

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        u = x.mean(axis=1, keepdim=True)
        s = (x - u).pow(2).mean(axis=1, keepdim=True)
        x = (x - u) / paddle.sqrt(s + self.eps)
        x = self.weight.reshape([-1, 1, 1]) * x + self.bias.reshape([-1, 1, 1])
        return x


class DepthwiseConv2d(nn.Conv2D):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1,
                 bias_attr=True):
        assert in_channels == out_channels
        super(DepthwiseConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                              groups=in_channels, bias_attr=bias_attr)


class SeparableConv2d(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias_attr=True, activation=None):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2D(in_channels, in_channels, kernel_size, stride, padding, dilation,
                                   groups=in_channels, bias_attr=False)
        self.activation = activation if activation else nn.Identity()
        self.pointwise = nn.Conv2D(in_channels, out_channels, 1, bias_attr=bias_attr)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.activation(x)
        x = self.pointwise(x)
        return x


class ConvBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias_attr=False,
                 bn=True,
                 relu=True,
                 init_fn=None):
        super(ConvBlock, self).__init__()
        layers = [nn.Conv2D(in_channels, out_channels, kernel_size, stride,
                            padding, dilation, groups,
                            bias_attr=bias_attr)]
        if bn:
            layers.append(nn.BatchNorm2D(out_channels))
        if relu:
            layers.append(nn.ReLU())
        self.block = nn.Sequential(*layers)
        if init_fn:
            self.block.apply(init_fn)

    def forward(self, x):
        return self.block(x)

    @staticmethod
    def same_padding(kernel_size, dilation):
        return dilation * (kernel_size - 1) // 2


class SeparableConvBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1,
                 bias_attr=False,
                 bn=True,
                 relu=True,
                 init_fn=None):
        super(SeparableConvBlock, self).__init__()
        layers = [SeparableConv2d(in_channels, out_channels, kernel_size, stride,
                                  padding, dilation,
                                  bias_attr)]
        if bn:
            layers.append(nn.BatchNorm2D(out_channels))
        if relu:
            layers.append(nn.ReLU())
        self.block = nn.Sequential(*layers)
        if init_fn:
            self.block.apply(init_fn)

    @staticmethod
    def same_padding(kernel_size, dilation):
        return dilation * (kernel_size - 1) // 2


class PoolBlock(nn.Layer):
    def __init__(self, output_size, in_channels, out_channels):
        super(PoolBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2D(output_size)
        self.conv = ConvBlock(in_channels, out_channels, 1)

    def forward(self, x: paddle.Tensor):
        size = x.shape[-2:]
        x = self.pool(x)
        x = self.conv(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ResidualBlock(nn.Layer):
    def __init__(self, *args):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(*args)

    def forward(self, x):
        identity = x
        x = self.block(x)
        x += identity
        return x


class ChannelReduction(nn.LayerList):
    def __init__(self, in_channels_list, out_channels):
        super(ChannelReduction, self).__init__(
            [ConvBlock(in_channels, out_channels, 1, bn=True, relu=False) for in_channels in
             in_channels_list])

    def forward(self, features: typing.List[paddle.Tensor]):
        return [m(feature) for m, feature in zip(self, features)]


class ChannelConcat(nn.Layer):
    def forward(self, features: typing.List[paddle.Tensor]):
        assert isinstance(features, (list, tuple))
        if len(features) == 1:
            return features[0]
        return paddle.concat(features, axis=1)


class Sum(nn.Layer):
    def forward(self, features: typing.List[paddle.Tensor]):
        assert isinstance(features, (list, tuple))
        if len(features) == 1:
            return features[0]
        return sum(features)


class ListIndex(nn.Layer):
    def __init__(self, *args):
        super(ListIndex, self).__init__()
        self.index = args

    def forward(self, features: typing.List[paddle.Tensor]):
        if len(self.index) == 1:
            return features[self.index[0]]
        else:
            return [features[i] for i in self.index]


class Bf16compatible(nn.Layer):
    def __init__(self, module):
        super().__init__()
        self._inner_module = module

    def forward(self, x):
        dtype = x.dtype
        if dtype == paddle.bfloat16:
            x = x.astype(paddle.float32)

        x = self._inner_module(x)
        if dtype == paddle.bfloat16:
            x = x.astype(dtype)

        return x


class ConvUpsampling(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1):
        super(ConvUpsampling, self).__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.upsample = Bf16compatible(nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False))

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return x


class Squeeze(nn.Layer):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x: paddle.Tensor):
        return paddle.squeeze(x, axis=self.dim)