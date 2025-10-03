import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Uniform
import numpy as np
import math
from typing import Type
from paddle import ParamAttr
from paddle.nn.initializer import Constant
from paddle.vision.ops import DeformConv2D
from paddle.autograd import PyLayer
from einops import rearrange

class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_c,
                 out_c,
                 filter_size,
                 stride,
                 padding,
                 num_groups=1,
                 if_act=True,
                 act='gelu',
                 dilation=1):
        super().__init__()

        self.c = nn.Conv2D(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            bias_attr=False,
            dilation=dilation)
        self.bn = nn.BatchNorm2D(out_c)
        # nn.BatchNorm(
        #     num_channels=out_c,
        #     act=None,
        #     param_attr=paddle.ParamAttr(regularizer=paddle.regularizer.L2Decay(0.0)),
        #     bias_attr=paddle.ParamAttr(regularizer=paddle.regularizer.L2Decay(0.0)))
        self.if_act = if_act
        if act == 'gelu':
            self.act = nn.GELU()
        elif act == 'silu':
            self.act = nn.Silu()
        elif act == 'swish':            
            self.act = nn.Swish()   
        else:
            self.act = nn.ReLU()

    def forward(self, x):
        x = self.c(x)
        x = self.bn(x)
        if self.if_act:
            x = self.act(x)
        return x

class DropPath(nn.Layer):
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.training and self.drop_prob > 0.0:
            keep_prob = 1.0 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
            mask = paddle.to_tensor(paddle.bernoulli(paddle.full(shape, keep_prob)))
            x = x / keep_prob * mask 
        return x

class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels):
        super().__init__(num_groups=1, num_channels=num_channels, eps=1e-6)


class LayerNorm2d(nn.Layer):
    def __init__(self, num_channels: int, eps: float=1e-06) -> None:
        super().__init__()
        self.weight = paddle.create_parameter(
            shape=[num_channels],
            dtype='float32',
            default_initializer=nn.initializer.Constant(value=1.0))
        self.bias = paddle.create_parameter(
            shape=[num_channels],
            dtype='float32',
            default_initializer=nn.initializer.Constant(value=0.0))
        self.eps = eps

    def forward(self, x: paddle.tensor) -> paddle.tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / paddle.sqrt(s + self.eps)
        x = self.weight[:, (None), (None)] * x + self.bias[:, (None), (None)]
        return x

def rotate_every_two(x):
    x1 = x[:, :, :, :, ::2]
    x2 = x[:, :, :, :, 1::2]
    x = paddle.stack([-x2, x1], axis=-1)
    x = paddle.flatten(x, start_axis=-2)
    return x

def toodd(size):
    size = [size, size]
    if size[0] % 2 == 1:
        pass
    else:
        size[0] = size[0] + 1 
    if size[1] % 2 == 1:
        pass
    else:
        size[1] = size[0] + 1
    return size

def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)


class NA2DQK(nn.Layer):
    def __init__(self, dim, window_size=7, num_heads=8):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk = nn.Linear(dim, dim * 2, weight_attr=nn.initializer.XavierUniform())

    def forward(self, x):
        B, H, W, C = x.shape
        qk = self.qk(x)
        q, k = qk.chunk(2, axis=-1)
        q = q.reshape([B, H, W, self.num_heads, self.head_dim]).transpose([0, 3, 1, 2, 4])
        k = k.reshape([B, H, W, self.num_heads, self.head_dim]).transpose([0, 3, 1, 2, 4])

        q = q.unfold(3, self.window_size, 1).unfold(4, self.window_size, 1)
        k = k.unfold(3, self.window_size, 1).unfold(4, self.window_size, 1)
        attn = paddle.einsum('bnhwijc,bnhlmkc->bnhwijlm', q, k) / (self.head_dim ** 0.5)
        return attn

class NA2DAV(nn.Layer):
    def __init__(self, dim, window_size=7, num_heads=8):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.v = nn.Linear(dim, dim, weight_attr=nn.initializer.XavierUniform())

    def forward(self, x, attn):
        B, H, W, C = x.shape
        v = self.v(x).reshape([B, H, W, self.num_heads, self.head_dim]).transpose([0, 3, 1, 2, 4])
        v = v.unfold(3, self.window_size, 1).unfold(4, self.window_size, 1)
        out = paddle.einsum('bnhwijlm,bnhlmkc->bnhwijc', attn.softmax(axis=-1), v)
        out = out.reshape([B, self.num_heads, H, W, self.head_dim]).transpose([0, 2, 3, 1, 4])
        return out.reshape([B, H, W, C])

class NA2D(nn.Layer):
    def __init__(self, dim, window_size=7, num_heads=8):
        super().__init__()
        self.qk = NA2DQK(dim, window_size, num_heads)
        self.av = NA2DAV(dim, window_size, num_heads)
    
    def forward(self, x):
        attn = self.qk(x)
        return self.av(x, attn)

class MLPBlock(nn.Layer):
    def __init__(self,
                 embedding_dim: int,
                 mlp_dim: int,
                 act: Type[nn.Layer]=nn.GELU) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: paddle.tensor) -> paddle.tensor:
        y = self.lin1(x)
        y = self.act(y)
        y = self.lin2(y)
        return y

def insert_global_token(x, global_token_pos, global_token=None):
    if global_token_pos == 'none' or global_token is None:
        return x

    B, N, C = x.shape
    if global_token_pos == 'headtail':
        x = paddle.concat([global_token, x, global_token], axis=1)
    elif global_token_pos == 'mid':
        x = paddle.concat([x[:, :N//2, ...], global_token, x[:, N//2:, ...]], axis=1)
    elif global_token_pos == 'head':
        x = paddle.concat([global_token, x], axis=1)
    else:
        raise ValueError(f'global_token_pos={global_token_pos} is not supported')
    return x

def split_global_token(x, global_token_pos, global_token=None):
    if global_token_pos == 'none':
        return x
    B, N, C = x.shape
    _, N_global, _ = global_token.shape
    if global_token_pos == 'headtail':
        x = x[:, N_global:-N_global, ...]
    elif global_token_pos == 'mid':
        x = paddle.concat([x[:, :N//2, ...], x[:, N//2+N_global:, ...]], axis=1)
    elif global_token_pos == 'head':
        x = x[:, N_global:, ...]
    else:
        raise ValueError(f'global_token_pos={global_token_pos} is not supported')
    return x

def nlc_to_nchw(x: paddle.Tensor, shape: list):
    x = rearrange(x, 'b n c -> b c n')
    x = paddle.reshape(x, shape=shape)
    return x
    
def nchw_to_nlc(x: paddle.Tensor):
    B, C, H, W = x.shape
    x = paddle.reshape(x, shape=[B, C, H * W])
    x = rearrange(x, 'b c n -> b n c')
    return x

def features_transfer(x, data_format='NCWH'):
        x = x.transpose((0, 2, 1))
        S = x.shape[-1]
        s = int(math.sqrt(S))
        if data_format == 'NWHC':
            x = rearrange(x, 'b c (h w) -> b h w c', h=s, w=s)
        elif data_format == 'NCWH':
            x = rearrange(x, 'b c (h w) -> b c h w', h=s, w=s)
        return x

class FFN(nn.Layer):
    def __init__(self, dim):
        super(FFN, self).__init__()
        self.dim = dim
        self.dim_sp = dim // 2

        self.conv_init = nn.Sequential(
            nn.Conv2D(dim, 2*dim, 1),
        )

        self.conv1_1 = nn.Sequential(
            nn.Conv2D(self.dim_sp, self.dim_sp, kernel_size=3, padding=1,
                      groups=self.dim_sp),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2D(self.dim_sp, self.dim_sp, kernel_size=5, padding=2,
                      groups=self.dim_sp),
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2D(self.dim_sp, self.dim_sp, kernel_size=7, padding=3,
                      groups=self.dim_sp),
        )

        self.gelu = nn.GELU()
        self.conv_fina = nn.Sequential(
            nn.Conv2D(self.dim_sp, dim, 1),
        )

    def forward(self, x):
        x = self.conv_init(x)
        x = paddle.split(x, 4, axis=1)
        x[1] = self.conv1_1(x[1])
        x[2] = self.conv1_2(x[2])
        x[3] = self.conv1_3(x[3])
        
        y = x[0] + x[1] + x[2] + x[3]
        y = self.gelu(y)
        y = self.conv_fina(y)

        return y


class GhostConv2D(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.kernel_size = kernel_size
        init_ch = out_channels // 2
        
        
        # 生成偏移量的卷积层
        self.prim_conv = nn.Sequential(
            nn.Conv2D(in_channels, init_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias_attr=False),
            nn.BatchNorm2D(init_ch),
            nn.ReLU()
        )
        self.cheap_conv = nn.Sequential(
            nn.Conv2D(init_ch, init_ch, kernel_size=kernel_size, stride=stride, padding=padding, groups=init_ch),
            nn.BatchNorm2D(init_ch),
            nn.ReLU()
        )
    def forward(self, x):
        x1 = self.prim_conv(x)
        x2 = self.cheap_conv(x1)
        output = paddle.concat([x1, x2], axis=1)
        
        return output

class CrossEntropyLoss(nn.Layer):
    """
    Implements the cross entropy loss function.

    Args:
        weight (tuple|list|ndarray|Tensor, optional): A manual rescaling weight
            given to each class. Its length must be equal to the number of classes.
            Default ``None``.
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
        top_k_percent_pixels (float, optional): the value lies in [0.0, 1.0].
            When its value < 1.0, only compute the loss for the top k percent pixels
            (e.g., the top 20% pixels). This is useful for hard pixel mining. Default ``1.0``.
        avg_non_ignore (bool, optional): Whether the loss is only averaged over non-ignored value of pixels. Default: True.
        data_format (str, optional): The tensor format to use, 'NCHW' or 'NHWC'. Default ``'NCHW'``.
    """

    def __init__(self,
                 weight=None,
                 ignore_index=255,
                 top_k_percent_pixels=1.0,
                 avg_non_ignore=True,
                 data_format='NCHW'):
        super(CrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.top_k_percent_pixels = top_k_percent_pixels
        self.avg_non_ignore = avg_non_ignore
        self.EPS = 1e-8
        self.data_format = data_format
        if weight is not None:
            self.weight = paddle.to_tensor(weight, dtype='float32')
        else:
            self.weight = None

    def forward(self, logit, label, semantic_weights=None):
        """
        Forward computation.

        Args:
            logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C), where C is number of classes, and if shape is more than 2D, this
                is (N, C, D1, D2,..., Dk), k >= 1.
            label (Tensor): Label tensor, the data type is int64. Shape is (N), where each
                value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
                (N, D1, D2,..., Dk), k >= 1.
            semantic_weights (Tensor, optional): Weights about loss for each pixels,
                shape is the same as label. Default: None.
        Returns:
            (Tensor): The average loss.
        """
        channel_axis = 1 if self.data_format == 'NCHW' else -1
        if self.weight is not None and logit.shape[channel_axis] != len(
                self.weight):
            raise ValueError(
                'The number of weights = {} must be the same as the number of classes = {}.'
                .format(len(self.weight), logit.shape[channel_axis]))

        if channel_axis == 1:
            logit = paddle.transpose(logit, [0, 2, 3, 1])
        label = label.astype('int64')

        loss = F.cross_entropy(
            logit,
            label,
            ignore_index=self.ignore_index,
            reduction='none',
            weight=self.weight)

        return self._post_process_loss(logit, label, semantic_weights, loss)

    def _post_process_loss(self, logit, label, semantic_weights, loss):
        """
        Consider mask and top_k to calculate the final loss.

        Args:
            logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C), where C is number of classes, and if shape is more than 2D, this
                is (N, C, D1, D2,..., Dk), k >= 1.
            label (Tensor): Label tensor, the data type is int64. Shape is (N), where each
                value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
                (N, D1, D2,..., Dk), k >= 1.
            semantic_weights (Tensor, optional): Weights about loss for each pixels,
                shape is the same as label.
            loss (Tensor): Loss tensor which is the output of cross_entropy. If soft_label
                is False in cross_entropy, the shape of loss should be the same as the label.
                If soft_label is True in cross_entropy, the shape of loss should be
                (N, D1, D2,..., Dk, 1).
        Returns:
            (Tensor): The average loss.
        """
        if self.avg_non_ignore:
            mask = paddle.cast(label != self.ignore_index, dtype='float32')
        else:
            mask = paddle.ones(label.shape, dtype='float32')
        mask.stop_gradient = True
        label.stop_gradient = True

        if loss.ndim > mask.ndim:
            loss = paddle.squeeze(loss, axis=-1)
        loss = loss * mask
        if semantic_weights is not None:
            loss = loss * semantic_weights

        if self.weight is not None:
            _one_hot = F.one_hot(label * mask, logit.shape[-1])
            coef = paddle.sum(_one_hot * self.weight, axis=-1)
        else:
            coef = paddle.ones_like(label)
        coef = paddle.cast(coef, dtype='float32')
        if self.top_k_percent_pixels == 1.0:
            avg_loss = paddle.mean(loss) / (paddle.mean(mask * coef) + self.EPS)
        else:
            loss = loss.reshape((-1, ))
            top_k_pixels = int(self.top_k_percent_pixels * loss.numel())
            loss, indices = paddle.topk(loss, top_k_pixels)
            coef = coef.reshape((-1, ))
            coef = paddle.gather(coef, indices)
            coef.stop_gradient = True
            coef = coef.astype('float32')
            avg_loss = loss.mean() / (paddle.mean(coef) + self.EPS)

        return avg_loss

class SVDLinear(nn.Layer):
    """
    SVD-based Linear Layer.
    Implements a low-rank linear transformation y = (U @ Sigma @ V.T + I) x
    where:
    - U and V are orthogonal matrices (from SVD decomposition)
    - Sigma is a diagonal matrix of singular values
    - I is the identity matrix (for residual connection)
    
    Args:
        in_features (int): Size of each input sample
        out_features (int): Size of each output sample
        rank (int): The rank of the decomposition. Must be <= min(in_features, out_features)
        bias (bool, optional): If True, adds a learnable bias. Default: True
        scale (float, optional): Scaling factor for the residual connection. Default: 1.0
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 rank: int=0, 
                 bias: bool = True,
                 scale: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scale = scale
        
        assert rank <= min(in_features, out_features), "Rank cannot be larger than min(in_features, out_features)"
        
        # Initialize orthogonal matrices U and V
        self.U = self.create_parameter(
            shape=[out_features, rank],
            default_initializer=nn.initializer.Orthogonal()
        )
        self.V = self.create_parameter(
            shape=[in_features, rank],
            default_initializer=nn.initializer.Orthogonal()
        )
        
        # Singular values (diagonal matrix, stored as vector)
        self.sigma = self.create_parameter(
            shape=[rank],
            default_initializer=nn.initializer.Constant(1.0)
        )
        
        # Bias term
        if bias:
            self.bias = self.create_parameter(
                shape=[out_features],
                default_initializer=nn.initializer.Constant(0.0)
            )
        else:
            self.bias = None
        
        # Identity matrix for residual connection
        self.register_buffer(
            "I", 
            paddle.eye(rank),
            persistable=True
        )
    
    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        # Compute SVD components: U @ (Sigma + I) @ V.T
        sigma_matrix = paddle.diag(self.sigma) + self.I * self.scale
        weight = self.U @ sigma_matrix @ self.V.T  # [out_features, in_features]
        
        # Reshape for matmul (batch processing)
        if x.ndim == 2:
            output = x @ weight.T
        else:
            # Handle higher dimensional inputs (e.g., batch x seq_len x features)
            original_shape = x.shape
            x = x.reshape([-1, original_shape[-1]])
            output = x @ weight.T
            output = output.reshape([*original_shape[:-1], -1])
        
        # Add bias if needed
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, scale={self.scale}'

if __name__ == "__main__":
    print('utils')
    x = paddle.rand([5,3,16,16], dtype=paddle.float32).cuda()
    y = paddle.to_tensor(0.2)
    print(x[0,0,0,0], x[0,0,0,0].item())

    # m = DepthWiseConv2D(3,1).to("gpu:0")
    # y = m(x)
    # print(x == y)