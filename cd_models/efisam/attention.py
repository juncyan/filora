import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.tensor
import paddleseg.models.layers as layers
from typing import Union, Optional
import numpy as np

class MLPBlock(nn.Layer):
    def __init__(self,
                 embedding_dim: int,
                 mlp_dim: int,
                 act = nn.GELU) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: paddle.tensor) -> paddle.tensor:
        y = self.lin1(x)
        y = self.act(y)
        y = self.lin2(y)
        return y

def img_mse(output, gt):
  return 0.5 * ((output - gt) ** 2).mean()

def img_psnr(mse):
  return -10.0 * np.log10(2.0 * mse)

class RandFourierFeature(nn.Layer):
    def __init__(self, input_dim, num_features=256, sigma=1.0):
        super(RandFourierFeature, self).__init__()
        self.input_dim = input_dim
        self.num_features = num_features
        self.sigma = sigma
        self.W = self.create_parameter(
            shape=[self.input_dim, self.num_features],
            default_initializer=nn.initializer.Normal(mean=0.0, std=1.0 / sigma)
        )
        self.b = self.create_parameter(
            shape=[self.num_features],
            default_initializer=nn.initializer.Uniform(low=0.0, high=2 * np.pi)
        )
        self.lamda = np.sqrt(2.0 / num_features)

    def forward(self, x):
        # Compute the random Fourier features
        xW = paddle.matmul(x, self.W)
        xW_plus_b = xW + self.b
        cos_features = paddle.cos(xW_plus_b)
        # features = self.lamda * cos_features
        # sin_features = paddle.sin(xW_plus_b)
        features = self.lamda * cos_features
        # features = self.ln(features)
       
        features = features * x
        # features = self.ln(features)
        features = F.relu(features)
        
        return features

class FCLayer(nn.Layer):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.act = nn.ReLU()

    def forward(self, input):
        output = self.linear(input)
        output = self.act(output)
        return output

class RFFA(nn.Layer):
    def __init__(self, in_features=2, out_features=1):
        super().__init__()
        self.fc = FCLayer(in_features, out_features)
        self.rff = RandFourierFeature(out_features,out_features)
        
        # setattr(self, f'FC_{i:d}', FCLayer(in_channel, hidden_features, nn.ReLU(inplace=True)))
    def forward(self, x):
        output = self.fc(x)
        output = self.rff(output)
        return output


class FCLayer_aff(nn.Layer):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w = 32
        self.h = 32
        self.linear = nn.Linear(in_features, out_features)

        self.affine1 = self.create_parameter(shape=[1,1,self.w, self.h], default_initializer=nn.initializer.Constant(1.0))
        self.affine2 = self.create_parameter(shape=[1,1,self.w, self.h], default_initializer=nn.initializer.Constant(0.0))

        self.act = nn.ReLU()

    def forward(self, input, coord):
        output = self.linear(input)
        output = F.instance_norm(output)
        affine = paddle.concat([self.affine1, self.affine2], axis=0)
        affine = nn.functional.grid_sample(affine, coord, padding_mode='border', align_corners=True).reshape([2,-1,1])
        output = output*affine[0]+affine[1]
        output = self.act(output)
        return output


class FFN(nn.Layer):
    def __init__(self, in_features=2, out_features=1,hidden_features=256, num_layers=3, num_frequencies=256, sigma = 10, scale = -1):
        super().__init__()

        self.pos_enc = RandFourierFeature(in_features,num_frequencies = num_frequencies,sigma = sigma, scale=scale)
        self.num_layers = num_layers
        for i in range(self.num_layers):
            if i==0:
                in_channel = self.pos_enc.out_features
            else:
                in_channel = hidden_features
        setattr(self, f'FC_{i:d}', FCLayer(in_channel, hidden_features, nn.ReLU(inplace=True)))
        self.FC_final = FCLayer(hidden_features, out_features, nn.Sigmoid())

    def forward(self, coords):
        output = self.pos_enc(coords)
        for i in range(self.num_layers):
            fc = getattr(self, f'FC_{i:d}')
        output = fc(output)
        output = self.FC_final(output)
        return output

def get_mgrid(w,h, dim=2, offset=0.5):
    x = np.arange(0, w, dtype=np.float32)
    y = np.arange(0, h, dtype=np.float32)
    # size = max(w,h)
    # x = (x + offset) / size   # [0, size] -> [0, 1]
    # y = (y + offset) / size   # [0, size] -> [0, 1]
    x = (x + offset) / w   # [0, size] -> [0, 1]
    y = (y + offset) / h   # [0, size] -> [0, 1]
    X,Y = np.meshgrid(x,y, indexing='ij')
    output = np.stack([X,Y], -1)
    output = output.reshape(w*h, dim)
    return output

class ECA(nn.Layer):
    """Constructs a ECA module.
    Args:
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv = nn.Conv1D(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias_attr=None) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose([0,2,1])).transpose([0,2,1]).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class Flatten(nn.Layer):
    def forward(self, x):
        y = x.reshape([x.shape[0], x.shape[1], -1])
        y = y.transpose([0, 2, 1])
        return y
    

class ChannelGate(nn.Layer):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        # self.mlp = nn.Sequential(
        #     Flatten(),
        #     nn.Linear(gate_channels, gate_channels // reduction_ratio),
        #     nn.ReLU(),
        #     nn.Linear(gate_channels // reduction_ratio, gate_channels)
        #     )
        self.flap = Flatten()
        self.mlp = MLPBlock(gate_channels, gate_channels // reduction_ratio)
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.shape[2], x.shape[3]), stride=(x.shape[2], x.shape[3]))
                avg_pool = self.flap(avg_pool)
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.shape[2], x.shape[3]), stride=(x.shape[2], x.shape[3]))
                max_pool = self.flap(max_pool)
                channel_att_raw = self.mlp(max_pool)
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.shape[2], x.shape[3]), stride=(x.shape[2], x.shape[3]))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum)
        # .unsqueeze(2).unsqueeze(3).expand_as(x)
        scale = self.flap(scale).unsqueeze(-1)
        return x * scale

def lp_pool2d(input, norm_type,kernel_size,stride, ceil_mode = False):
    r"""
    Apply a 2D power-average pooling over an input signal composed of several input planes.

    If the sum of all inputs to the power of `p` is
    zero, the gradient is set to zero as well.

    See :class:`~paddle.nn.LPPool2d` for details.
    """
   
    kw, kh = kernel_size
    if stride is not None:
        out = F.avg_pool2d(input.pow(norm_type), kernel_size, stride, 0, ceil_mode)
    else:
        out = F.avg_pool2d(input.pow(norm_type), kernel_size, padding=0, ceil_mode=ceil_mode)

    return (paddle.sign(out) * F.relu(paddle.abs(out))).mul(kw * kh).pow(1.0 / norm_type)

def logsumexp_2d(tensor:paddle.tensor):
    tensor_flatten = tensor.reshape(tensor.shape[0], tensor.shape[1], -1)
    s, _ = paddle.max(tensor_flatten, axis=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(axis=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Layer):
    def forward(self, x):
        cm = paddle.max(x,1).unsqueeze(1)
        ca = paddle.mean(x,1).unsqueeze(1)
        return paddle.concat([cm, ca], axis=1)

class SpatialGate(nn.Layer):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = layers.ConvBN(2,1,kernel_size)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)
        return x * scale

class BAM(nn.Layer):
    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)
    def forward(self,in_tensor):
        att = 1 + F.sigmoid( self.channel_att(in_tensor) * self.spatial_att(in_tensor) )
        return att * in_tensor

class CBAM(nn.Layer):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class Attention(nn.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2D(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x)
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
        
    def _init_weights(self, m):
        weight_attr = paddle.ParamAttr(initializer=1.0)
        bias_attr = paddle.framework.ParamAttr(initializer=0.0)
        if isinstance(m, nn.Linear):
            m.bias_attr = bias_attr
            m.weight_attr = weight_attr
        elif isinstance(m, nn.LayerNorm):
            m.bias_attr = bias_attr
            m.weight_attr = weight_attr
        elif isinstance(m, nn.Conv2D):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            
            m.weight_attr = paddle.normal(0, np.sqrt(2.0 / fan_out))
            # m.weight.data.normal_(0, np.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias_attr = paddle.framework.ParamAttr(initializer=0.0)

class SEModule(nn.Layer):
    def __init__(self, channels, reduction_channels):
        super(SEModule, self).__init__()
        self.fc1 = nn.Conv2D(
            channels,
            reduction_channels,
            kernel_size=1,
            padding=0,
            bias_attr=True)
        self.ReLU = nn.ReLU()
        self.fc2 = nn.Conv2D(
            reduction_channels,
            channels,
            kernel_size=1,
            padding=0,
            bias_attr=True)

    def forward(self, x):
        x_se = x.reshape(
            [x.shape[0], x.shape[1], x.shape[2] * x.shape[3]]).mean(-1).reshape(
                [x.shape[0], x.shape[1], 1, 1])

        x_se = self.fc1(x_se)
        x_se = self.ReLU(x_se)
        x_se = self.fc2(x_se)
        return x * F.sigmoid(x_se)