
import re
import time
import math
import numpy as np
from functools import partial
from typing import Optional, Union, Type, List, Tuple, Callable, Dict

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from einops import rearrange, repeat
from mamba_ssm_paddle.ops.selective_scan_interface import selective_scan_fn
# from paddlenlp_kernel.cuda.selective_scan import selective_scan_fn
# from paddle_ssm.cuda.selective_scan import selective_scan_fn

# from paddlenlp_kernel.cuda.selective_scan import selective_scan_fn
# from mamba_ssm_paddle.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref



class PatchEmbed2D(nn.Layer):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Layer, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=4, in_chans=1, embed_dim=96, norm_layer=None): # 修改通道数为1
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2D(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # print("dsm分割后的张量的尺寸(shape):",x.shape)  # 或 tensor.size()
        # print("dsm分割后的张量的维度数量(dim):", x.dim())
        x = self.proj(x).transpose([0, 2, 3, 1])
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Layer):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Layer, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias_attr=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
        
        x = paddle.concat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.reshape([B, H//2, W//2, 4 * C])  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class SS2D(nn.Layer):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
    ):
        super(SS2D, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias_attr=bias)
        self.conv2d = nn.Conv2D(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias_attr=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )
        self.act = nn.Silu()

        self.x_proj_layers = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias_attr=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias_attr=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias_attr=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias_attr=False),
        )
        # Stack weights and delete the original layers
        self.x_proj_weight = self.create_parameter(
            shape=[4, self.dt_rank + self.d_state * 2, self.d_inner],
            default_initializer=nn.initializer.Normal()  # Or use the appropriate initializer
        )
        self.x_proj_weights_temp = paddle.stack([layer.weight for layer in self.x_proj_layers], axis=0).transpose([0, 2, 1])
        self.x_proj_weight.set_value(self.x_proj_weights_temp)
        del self.x_proj_layers
        del self.x_proj_weights_temp

        self.dt_projs_layers = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
        )
        self.dt_projs_weight = self.create_parameter(
            shape=[4, self.d_inner, self.dt_rank],
            default_initializer=nn.initializer.Normal()  # Or use the appropriate initializer
        )
        self.dt_projs_weight_temp = paddle.stack([layer.weight for layer in self.dt_projs_layers], axis=0).transpose([0, 2, 1])
        self.dt_projs_weight.set_value(self.dt_projs_weight_temp)
        self.dt_projs_bias = self.create_parameter(
            shape=[4, self.d_inner],
            default_initializer=nn.initializer.Constant(0.0)  # Or use the appropriate initializer
        )
        
        del self.dt_projs_layers
        del self.dt_projs_weight_temp

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias_attr=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias_attr=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.initializer.Constant(dt_init_std)(dt_proj.weight)
        elif dt_init == "random":
            nn.initializer.Uniform(-dt_init_std, dt_init_std)(dt_proj.weight)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = paddle.exp(
            paddle.rand([d_inner]) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clip(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + paddle.log(-paddle.expm1(-dt))
        with paddle.no_grad():
            dt_proj.bias.set_value(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = paddle.arange(1, d_state + 1, dtype='float32').unsqueeze(0).expand([d_inner, -1])
        A_log = paddle.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = A_log.unsqueeze(0).expand([copies, -1, -1])
            if merge:
                A_log = A_log.reshape([-1, d_state])
        A_log = paddle.create_parameter(shape=A_log.shape, dtype='float32')
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = paddle.ones([d_inner], dtype='float32')
        if copies > 1:
            D = D.unsqueeze(0).expand([copies, -1])
            if merge:
                D = D.reshape([-1])
        D = paddle.create_parameter(shape=D.shape, dtype='float32')
        D._no_weight_decay = True
        return D

    def forward_core(self, x: paddle.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = paddle.stack([x.reshape([B, -1, L]), paddle.transpose(x, perm=[0, 2, 3, 1]).reshape([B, -1, L])], axis=1).reshape([B, 2, -1, L])
        xs = paddle.concat([x_hwwh, paddle.flip(x_hwwh, axis=[-1])], axis=1) # (b, k, d, l)

        x_dbl = paddle.einsum("b k d l, k c d -> b k c l", xs.reshape([B, K, -1, L]), self.x_proj_weight)
        dts, Bs, Cs = paddle.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], axis=2)
        dts = paddle.einsum("b k r l, k d r -> b k d l", dts.reshape([B, K, -1, L]), self.dt_projs_weight)

        xs = xs.astype('float32').reshape([B, -1, L]) # (b, k * d, l)
        dts = dts.reshape([B, -1, L]) # (b, k * d, l)
        Bs = Bs.astype('float32').reshape([B, K, -1, L]) # (b, k, d_state, l)
        Cs = Cs.astype('float32').reshape([B, K, -1, L]) # (b, k, d_state, l)
        Ds = self.Ds.astype('float32').reshape([-1]) # (k * d)
        As = -paddle.exp(self.A_logs.astype('float32')).reshape([-1, self.d_state])  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.astype('float32').reshape([-1]) # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).reshape([B, K, -1, L])
        assert out_y.dtype == paddle.float32

        inv_y = paddle.flip(out_y[:, 2:4], axis=[-1]).reshape([B, 2, -1, L])
        wh_y = paddle.transpose(out_y[:, 1].reshape([B, -1, W, H]), perm=[0, 1, 3, 2]).reshape([B, -1, L])
        invwh_y = paddle.transpose(inv_y[:, 1].reshape([B, -1, W, H]), perm=[0, 1, 3, 2]).reshape([B, -1, L])

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: paddle.Tensor):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = paddle.split(xz, 2, axis=-1) # (b, h, w, d)
        x = paddle.transpose(x, perm=[0, 3, 1, 2])
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == paddle.float32
        y = y1 + y2 + y3 + y4
        y = paddle.transpose(y, perm=[0, 2, 1]).reshape([B, H, W, -1])
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Layer):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., paddle.nn.Layer] = partial(nn.LayerNorm, epsilon=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state, **kwargs)
        self.drop_path = nn.Dropout(drop_path)

    def forward(self, input: paddle.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


class VSSLayer(nn.Layer):
    """ A basic layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Layer, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Layer | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        downsample=None, 
        use_checkpoint=False, 
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.LayerList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Layer):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.initializer.KaimingUniform(math.sqrt(5))(p)
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None


    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = paddle.load(blk, x)
            else:
                x = blk(x)
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x


class VSSBackbone(nn.Layer):
    def __init__(self, patch_size=4, in_chans=3, depths=[2, 2, 9, 2], 
                 dims=[96, 192, 384, 768], d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, 
                 norm_layer=nn.LayerNorm, patch_norm=True, 
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims
 
        # PatchEmbed2D
        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None) 

        # WASTED absolute position embedding ======================
        self.ape = False
        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            # self.absolute_pos_embed = nn.Parameter(paddle.zeros(1, *self.patches_resolution, self.embed_dim))
            # trunc_normal_(self.absolute_pos_embed, std=.02)
            self.absolute_pos_embed = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.LayerList()
        self.downsamples = nn.LayerList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)
            if i_layer < self.num_layers - 1:
                self.downsamples.append(PatchMerging2D(dim=dims[i_layer], norm_layer=norm_layer))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Layer):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless
        
        Conv2D is not intialized !!!
        """
        if isinstance(m, nn.Linear):
            nn.initializer.Normal(std=.02)(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.initializer.Constant(value=0.)(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.initializer.Constant(value=0.)(m.bias)
            nn.initializer.Constant(value=1.)(m.weight)

    def no_weight_decay(self):
        return {'absolute_pos_embed'}
    
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        x_ret = []
        # x_ret.append(x)

        x = self.patch_embed(x)
        # print('分割后的的rgb:',x.shape)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for s, layer in enumerate(self.layers):
            x = layer(x)
            x_ret.append(x)
            if s < len(self.downsamples):
                x = self.downsamples[s](x)

        return x_ret
