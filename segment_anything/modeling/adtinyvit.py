import itertools

import paddle
from paddle import nn
from paddle.nn import functional as F

import math
from .common import LayerNorm2d
import filora.loralib as lora

# NOTE: All the DropPath and parameters initialization are commented out due to sam do not support to train.


class TinyViT(nn.Layer):
    def __init__(
            self,
            img_size=224,
            in_chans=3,
            num_classes=1000,
            embed_dims=[96, 192, 384, 768],
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.0,
            drop_rate=0.0,
            drop_path_rate=0.1,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=1.0, 
            rank=8):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.depths = depths
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio

        activation = nn.GELU

        self.patch_embed = PatchEmbed(
            in_chans=in_chans,
            embed_dim=embed_dims[0],
            resolution=img_size,
            activation=activation, )

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # stochastic depth
        dpr = [
            x.item() for x in paddle.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.LayerList()
        for i_layer in range(self.num_layers):
            kwargs = dict(
                dim=embed_dims[i_layer],
                input_resolution=(
                    patches_resolution[0] // (2**(i_layer - 1 if i_layer == 3
                                                  else i_layer)),
                    patches_resolution[1] //
                    (2**(i_layer - 1 if i_layer == 3 else i_layer)), ),
                depth=depths[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                downsample=PatchMerging
                if (i_layer < self.num_layers - 1) else None,
                out_dim=embed_dims[min(i_layer + 1, len(embed_dims) - 1)],
                activation=activation, )
            if i_layer == 0:
                layer = ConvLayer(
                    conv_expand_ratio=mbconv_expand_ratio,
                    **kwargs, )
            else:
                layer = BasicLayer(
                    num_heads=num_heads[i_layer],
                    window_size=window_sizes[i_layer],
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    local_conv_size=local_conv_size,
                    rank=rank,
                    **kwargs, )
            self.layers.append(layer)

        # Classifier head
        self.norm_head = nn.LayerNorm(embed_dims[-1])
        self.head = (nn.Linear(embed_dims[-1], num_classes)
                     if num_classes > 0 else nn.Identity())

        # init weights
        # self.apply(self._init_weights)
        self.set_layer_lr_decay(layer_lr_decay)
        self.neck = nn.Sequential(
            nn.Conv2D(
                embed_dims[-1],
                256,
                kernel_size=1,
                bias_attr=False, ),
            LayerNorm2d(256),
            nn.Conv2D(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias_attr=False, ),
            LayerNorm2d(256), )

    def set_layer_lr_decay(self, layer_lr_decay):
        decay_rate = layer_lr_decay

        # layers -> blocks (depth)
        depth = sum(self.depths)
        lr_scales = [decay_rate**(depth - i - 1) for i in range(depth)]

        def _set_lr_scale(m, scale):
            for p in m.parameters():
                p.lr_scale = scale

        self.patch_embed.apply(lambda x: _set_lr_scale(x, lr_scales[0]))
        i = 0
        for layer in self.layers:
            for block in layer.blocks:
                block.apply(lambda x: _set_lr_scale(x, lr_scales[i]))
                i += 1
            if layer.downsample is not None:
                layer.downsample.apply(
                    lambda x: _set_lr_scale(x, lr_scales[i - 1]))
        assert i == depth
        for m in [self.norm_head, self.head]:
            m.apply(lambda x: _set_lr_scale(x, lr_scales[-1]))

        for k, p in self.named_parameters():
            p.param_name = k

        def _check_lr_scale(m):
            for p in m.parameters():
                assert hasattr(p, "lr_scale"), p.param_name

        self.apply(_check_lr_scale)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.initializer.Normal(std=.02)(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.initializer.Constant(0)(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.initializer.Constant(0)(m.bias)
            nn.initializer.Constant(0)(1.0)(m.weight)

    @paddle.no_grad()
    def build_abs(self):
        for m in self.sublayers():
            if isinstance(m, Attention):
                m.build_ab()

    def forward(self, x, y):
        # x: (N, C, H, W)
        x = self.patch_embed(x)
        x = self.layers[0](x)

        y = self.patch_embed(y)
        y = self.layers[0](y)
        start_i = 1

        for i in range(start_i, len(self.layers)):
            layer = self.layers[i]
            x, y = layer(x, y)  

        B, H, C = x.shape
        w = int(math.sqrt(H))
        x = x.reshape((B, w, w, C))
        x = x.transpose((0, 3, 1, 2))
      
        x = self.neck(x)

        y = y.reshape((B, w, w, C))
        y = y.transpose((0, 3, 1, 2))
        y = self.neck(y)
        return x, y
    
    def extract_features(self, x):
        # x: (N, C, H, W)
        x = self.patch_embed(x)
        x = self.layers[0](x)
        start_i = 1
        y = [x]
        for i in range(start_i, len(self.layers)):
            layer = self.layers[i]
            x = layer(x)
            y.append(x)
        B, wh, C = x.shape
        w = int(math.sqrt(wh))
        x = x.reshape((B, w, w, C))
        x = x.transpose((0, 3, 1, 2))
        x = self.neck(x)
        return y, x


class Conv2d_BN(nn.Sequential):
    def __init__(self,
                 a,
                 b,
                 ks=1,
                 stride=1,
                 pad=0,
                 dilation=1,
                 groups=1,
                 bn_weight_init=1.0):
        super().__init__()
        self.add_sublayer(
            "c",
            nn.Conv2D(
                a, b, ks, stride, pad, dilation, groups, bias_attr=False))
        bn = nn.BatchNorm2D(b)
        self.add_sublayer("bn", bn)


class PatchEmbed(nn.Layer):
    def __init__(self, in_chans, embed_dim, resolution, activation):
        super().__init__()
        if isinstance(resolution, int):
            resolution = (resolution, ) * 2
        img_size = resolution
        self.patches_resolution = (img_size[0] // 4, img_size[1] // 4)
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[
            1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        n = embed_dim
        self.seq = nn.Sequential(
            Conv2d_BN(in_chans, n // 2, 3, 2, 1),
            activation(),
            Conv2d_BN(n // 2, n, 3, 2, 1), )

    def forward(self, x):
        return self.seq(x)


class MBConv(nn.Layer):
    def __init__(self, in_chans, out_chans, expand_ratio, activation,
                 drop_path):
        super().__init__()
        self.in_chans = in_chans
        self.hidden_chans = int(in_chans * expand_ratio)
        self.out_chans = out_chans

        self.conv1 = Conv2d_BN(in_chans, self.hidden_chans, ks=1)
        self.act1 = activation()

        self.conv2 = Conv2d_BN(
            self.hidden_chans,
            self.hidden_chans,
            ks=3,
            stride=1,
            pad=1,
            groups=self.hidden_chans, )
        self.act2 = activation()

        self.conv3 = Conv2d_BN(
            self.hidden_chans, out_chans, ks=1, bn_weight_init=0.0)
        self.act3 = activation()

        # self.drop_path = DropPath(
        #     drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.act2(x)

        x = self.conv3(x)

        # x = self.drop_path(x)

        x += shortcut
        x = self.act3(x)

        return x


class PatchMerging(nn.Layer):
    def __init__(self, input_resolution, dim, out_dim, activation):
        super().__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.act = activation()
        self.conv1 = Conv2d_BN(dim, out_dim, 1, 1, 0)
        stride_c = 2
        if out_dim == 320 or out_dim == 448 or out_dim == 576:
            stride_c = 1
        self.conv2 = Conv2d_BN(out_dim, out_dim, 3, stride_c, 1, groups=out_dim)
        self.conv3 = Conv2d_BN(out_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        if x.ndim == 3:
            H, W = self.input_resolution
            B = x.shape[0]
            # (B, C, H, W)
            x = x.reshape((B, H, W, -1)).transpose((0, 3, 1, 2))

        x = self.conv1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = x.flatten(2).transpose((0, 2, 1))
        return x


class ConvLayer(nn.Layer):
    def __init__(
            self,
            dim,
            input_resolution,
            depth,
            activation,
            drop_path=0.0,
            downsample=None,
            out_dim=None,
            conv_expand_ratio=4.0, ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.LayerList([
            MBConv(
                dim,
                dim,
                conv_expand_ratio,
                activation,
                drop_path[i] if isinstance(drop_path, list) else drop_path, )
            for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution,
                dim=dim,
                out_dim=out_dim,
                activation=activation)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class Mlp(nn.Layer):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.0, 
            ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x, y):
        x = self.norm(x)
        y = self.norm(y)

        x = self.fc1(x)
        y = self.fc1(y)
        x = self.act(x)
        y = self.act(y)
        x = self.drop(x)
        y = self.drop(y)
        x = self.fc2(x)
        y = self.fc2(y)
        x = self.drop(x)
        y = self.drop(y)
        return x, y


class Attention(nn.Layer):
    def __init__(self,
                 dim,
                 key_dim,
                 num_heads=8,
                 attn_ratio=4,
                 resolution=(14, 14),
                 rank=8):
        super().__init__()
        assert isinstance(resolution, tuple) and len(resolution) == 2
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2

        self.norm = nn.LayerNorm(dim)
        # self.qkv = nn.Linear(dim, h)
        self.qkv = lora.BiLinear(dim, h, r = rank)
        self.proj = nn.Linear(self.dh, dim)
        # self.proj = lora.BiLinear(self.dh, dim, r = rank)

        points = list(
            itertools.product(range(resolution[0]), range(resolution[1])))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = self.create_parameter(
            shape=(num_heads, len(attention_offsets)),
            dtype="float32",
            default_initializer=nn.initializer.Constant(0.0), )
        self.register_buffer(
            "attention_bias_idxs",
            paddle.to_tensor(
                idxs, dtype=paddle.int64).reshape((N, N)),
            persistable=False, )

    @paddle.no_grad()
    def build_ab(self):
        # self.ab = self.attention_biases[:, self.attention_bias_idxs]
        idxs = self.attention_bias_idxs.reshape([-1]).tolist()
        ab = []
        for i in range(self.attention_biases.shape[0]):
            temp = self.attention_biases[i]
            ab.append(temp[idxs])
        self.ab = paddle.concat(ab)
        self.ab = self.ab.reshape((-1, *self.attention_bias_idxs.shape))

    def forward(self, x, y):  # x (B,N,C)
        B, N, _ = x.shape

        # Normalization
        x = self.norm(x)
        y = self.norm(y)

        # qkv= self.qkv(x)
        # yqkv = self.qkv(y)
        qkv, yqkv = self.qkv(x, y)
        # (B, N, num_heads, d)
        q, k, v = qkv.reshape((B, N, self.num_heads, -1)).split(
            [self.key_dim, self.key_dim, self.d], axis=3)
        
        qy, ky, vy = yqkv.reshape((B, N, self.num_heads, -1)).split(
            [self.key_dim, self.key_dim, self.d], axis=3)
        # (B, num_heads, N, d)
        q = q.transpose((0, 2, 1, 3))
        k = k.transpose((0, 2, 1, 3))
        v = v.transpose((0, 2, 1, 3))

        qy = qy.transpose((0, 2, 1, 3))
        ky = ky.transpose((0, 2, 1, 3))
        vy = vy.transpose((0, 2, 1, 3))

        # only for infer
        attn = (q @k.transpose((0, 1, 3, 2))) * self.scale + self.ab
        attn = F.softmax(attn, axis=-1)
        x = (attn @v).transpose((0, 2, 1, 3)).reshape((B, N, self.dh))
        x = self.proj(x)

        attny = (qy @ky.transpose((0, 1, 3, 2))) * self.scale + self.ab
        attny = F.softmax(attny, axis=-1)
        y = (attny @vy).transpose((0, 2, 1, 3)).reshape((B, N, self.dh))
        y = self.proj(y)
        # x, y = self.proj(x, y)

        return x, y


class BiTinyViTBlock(nn.Layer):
    def __init__(
            self,
            dim,
            input_resolution,
            num_heads,
            window_size=7,
            mlp_ratio=4.0,
            drop=0.0,
            drop_path=0.0,
            local_conv_size=3,
            activation=nn.GELU,
            rank=8):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        assert window_size > 0, "window_size must be greater than 0"
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        # self.drop_path = DropPath(
        #     drop_path) if drop_path > 0. else nn.Identity()

        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        head_dim = dim // num_heads

        window_resolution = (window_size, window_size)
        self.attn = Attention(
            dim,
            head_dim,
            num_heads,
            attn_ratio=1,
            resolution=window_resolution,
            rank=rank)

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_activation = activation
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=mlp_activation,
            drop=drop)
        
        pad = local_conv_size // 2
        self.local_conv = Conv2d_BN(
            dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim)

    def forward(self, x, y):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        res_x = x
        res_y = y
        if H == self.window_size and W == self.window_size:
            x, y= self.attn(x, y)
        else:
            x = x.reshape((B, H, W, C))
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0

            y = y.reshape((B, H, W, C))


            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, 0))
                y = F.pad(y, (0, 0, 0, pad_r, 0, pad_b, 0, 0))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_size
            nW = pW // self.window_size
            # window partition
            x = (x.reshape((B, nH, self.window_size, nW, self.window_size, C))
                 .transpose((0, 1, 3, 2, 4, 5)).reshape(
                     (B * nH * nW, self.window_size * self.window_size, C)))
            
            y = (y.reshape((B, nH, self.window_size, nW, self.window_size, C))
                 .transpose((0, 1, 3, 2, 4, 5)).reshape(
                     (B * nH * nW, self.window_size * self.window_size, C)))

            x, y = self.attn(x, y)
            # y = self.attn(y)
            # window reverse
            x = (x.reshape((B, nH, nW, self.window_size, self.window_size, C))
                 .transpose((0, 1, 3, 2, 4, 5)).reshape((B, pH, pW, C)))
            
            y = (y.reshape((B, nH, nW, self.window_size, self.window_size, C))
                 .transpose((0, 1, 3, 2, 4, 5)).reshape((B, pH, pW, C)))

            if padding:
                x = x[:, :H, :W]
                y = y[:, :H, :W]

            x = x.reshape((B, L, C))
            y = y.reshape((B, L, C))

        # x = res_x + self.drop_path(x)
        x = res_x + x
        y = res_y + y

        x = x.transpose((0, 2, 1)).reshape((B, C, H, W))
        x = self.local_conv(x)
        x = x.reshape((B, C, L)).transpose((0, 2, 1))

        y = y.transpose((0, 2, 1)).reshape((B, C, H, W))
        y = self.local_conv(y)
        y = y.reshape((B, C, L)).transpose((0, 2, 1))

        aux, auy = self.mlp(x, y)
        x = x + aux
        y = y + auy
        
        return x, y


class BasicLayer(nn.Layer):
    def __init__(
            self,
            dim,
            input_resolution,
            depth,
            num_heads,
            window_size,
            mlp_ratio=4.0,
            drop=0.0,
            drop_path=0.0,
            downsample=None,
            local_conv_size=3,
            activation=nn.GELU,
            rank=8,
            out_dim=None, ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.LayerList([
            BiTinyViTBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i]
                if isinstance(drop_path, list) else drop_path,
                local_conv_size=local_conv_size,
                activation=activation, 
                rank=rank) for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution,
                dim=dim,
                out_dim=out_dim,
                activation=activation)
        else:
            self.downsample = None

    def forward(self, x, y):
        for blk in self.blocks:
            x, y = blk(x, y)
        if self.downsample is not None:
            x = self.downsample(x)
            y = self.downsample(y)
        return x, y
