import numpy as np
import paddle
import paddle.nn as nn
import os
import paddleseg.models.layers as layers
import sys
# from ..utils import *


class conv2d(nn.Layer):
    def __init__(self,in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=1, bias_attr=False):
        super(conv2d, self).__init__()
        if type(kernel_size) is int:
            use_large_impl = kernel_size > 5
        else:
            assert len(kernel_size) == 2 and kernel_size[0] == kernel_size[1]
            use_large_impl = kernel_size[0] > 5
        has_large_impl = 'LARGE_KERNEL_CONV_IMPL' in os.environ

        if has_large_impl and in_channels == out_channels and out_channels == groups and use_large_impl and stride == 1 and padding == kernel_size // 2 and dilation == 1:
            self.conv = layers.DepthwiseConvBN(in_channels, in_channels, kernel_size, bias_attr=bias_attr)
        else:
            self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias_attr=bias_attr)

    def forward(self, x):
        return self.conv(x)


class convbn(nn.Layer):
    def __init__(self,in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=1, bias_attr=None):
        super(convbn, self).__init__()
        self.act = layers.ConvBN(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias_attr=bias_attr)
        if type(kernel_size) is int:
            use_large_impl = kernel_size > 5
        else:
            assert len(kernel_size) == 2 and kernel_size[0] == kernel_size[1]
            use_large_impl = kernel_size[0] > 5
        has_large_impl = 'LARGE_KERNEL_CONV_IMPL' in os.environ


        if has_large_impl and in_channels == out_channels and out_channels == groups and use_large_impl and stride == 1 and padding == kernel_size // 2 and dilation == 1:
            self.act.conv = layers.DepthwiseConvBN(in_channels, in_channels, kernel_size, bias_attr=bias_attr)
            self.act.bn = nn.BatchNorm2D(in_channels)

    def forward(self, x):
        return self.act(x)


class convbnrelu(nn.Layer):
    def __init__(self,in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=1, bias_attr=False):
        super(convbnrelu, self).__init__()
        self.act = layers.ConvBNReLU(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias_attr=bias_attr)
        if type(kernel_size) is int:
            use_large_impl = kernel_size > 5
        else:
            assert len(kernel_size) == 2 and kernel_size[0] == kernel_size[1]
            use_large_impl = kernel_size[0] > 5
        has_large_impl = 'LARGE_KERNEL_CONV_IMPL' in os.environ

        if has_large_impl and in_channels == out_channels and out_channels == groups and use_large_impl and stride == 1 and padding == kernel_size // 2 and dilation == 1:
            self.act.conv = layers.DepthwiseConvBN(in_channels, in_channels, kernel_size, bias_attr=bias_attr)
            self.act.bn = nn.BatchNorm2D(in_channels)

    def forward(self, x):
        return self.act(x)


def fuse_bn(conv:nn.Layer, bn:nn.Layer):
    kernel = conv.weight
    running_mean = bn._mean
    runing_var = bn._variance
    gamma = bn.weight
    beta = bn.bias
    eps = bn._epsilon
    std = (runing_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


class ReparamLargeKernelConv(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups,
                 small_kernel,
                 small_kernel_merged=True):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        padding = kernel_size//2
        if small_kernel_merged:
            self.lkb_reparam = conv2d(in_channels,out_channels, kernel_size, stride, padding,
                                    groups=groups, bias_attr=None)
            self.merge_kernel()
        else:
            self.lkb_origin = convbn(in_channels,out_channels, kernel_size, stride, padding,
                                    groups=groups)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = convbn(in_channels,out_channels, small_kernel, stride, small_kernel//2,
                                    groups=groups)

    def forward(self, x):
        if hasattr(self, 'lkb_reparam'):
            y = self.lkb_reparam(x)
        else:
            y = self.lkb_origin(x)
            if hasattr(self, 'small_conv'):
                y += self.small_conv(x)
        return y

    def get_equivalent_kernel_buas(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.act.conv, self.lkb_origin.act.bn)
        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.small_conv.act.conv, self.small_conv.act.bn)
            eq_b += small_b
            eq_k += nn.functional.pad(small_k, [(self.kernel_size - self.small_kernel) // 2] * 4)

        return eq_k, eq_b

    def merge_kernel(self):
        print("merge kernel")
        eq_k, eq_b = self.get_equivalent_kernel_buas()
        act = self.lkb_origin.act.conv
        self.lkb_reparam = conv2d(in_channels=act.in_channels,
                                out_channels=act.out_channels,
                                kernel_size=act.kernel_size,
                                stride=act.stride,
                                padding=act.padding,
                                dilation=act.dilation,
                                groups=act.groups,
                                bias_attr=True)

        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        # m = nn.Conv2D(act.in_channels, act.out_channels,act.kernel_size, act.stride,act.padding,act.dilation,act.groups,
        #               bias_attr=nn.initializer.Assign(eq_b),
        #               weight_attr=nn.initializer.Assign(eq_k))
        # self.lkb_reparam.weight = m.weight
        # self.lkb_reparam.bias = m.bias
        print("weight", self.lkb_reparam.weight)
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')


class ConvFFN(nn.Layer):
    def __init__(self, in_channels, internal_channels, out_channels, drop_path):
        super().__init__()
        # print(f"n_channels, internal_channels, out_channels, drop_path, {in_channels, internal_channels, out_channels, drop_path}")
        self.drop_path = nn.Dropout(drop_path)
        self.preffn_bn = nn.BatchNorm2D(in_channels)
        self.pw1 = convbn(in_channels, internal_channels, 1, 1)
        self.pw2 = convbn(internal_channels, out_channels, 1, 1)
        self.nonlinear = nn.GELU()

    def forward(self, x):
        y = self.preffn_bn(x)
        y = self.pw1(y)
        y = self.nonlinear(y)
        y = self.pw2(y)
        return x + self.drop_path(y)


class RepLKBlock(nn.Layer):
    def __init__(self, in_channels, dw_channels, block_lk_size, small_kernel, drop_path, small_kernel_merged=False):
        super(RepLKBlock, self).__init__()

        # self.inc = in_channels
        # self.dwc = dw_channels
        # self.ks = block_lk_size
        # self.dp = drop_path
        # self.sk = small_kernel

        self.pw1 = convbnrelu(in_channels, dw_channels, 1, 1)
        self.pw2 = convbn(dw_channels, in_channels, 1, 1)
        self.large_kernel = ReparamLargeKernelConv(dw_channels, dw_channels, block_lk_size,
                                                   1, dw_channels, small_kernel, small_kernel_merged)
        self.lk_nonlinear = nn.ReLU()
        self.prelkb_bn = nn.BatchNorm2D(in_channels)
        self.drop_path = nn.Dropout(drop_path)
        # print('drop path: ', self.drop_path.drop_path)

    def forward(self, x):
        # print(f"in_channels, dw_channels, block_lk_size, small_kernel, drop_path, {self.inc, self.dwc, self.ks, self.sk, self.dp}")
        y = self.prelkb_bn(x)
        y = self.pw1(y)
        y = self.large_kernel(y)
        y = self.lk_nonlinear(y)
        y = self.pw2(y)
        return x + self.drop_path(y)


class RepLKNetStage(nn.Layer):
    def __init__(self, channels, num_blocks, stage_lk_size, drop_path,
                 small_kernel, dw_ratio=1, ffn_ratio=4,
                 small_kernel_merged=True,
                 norm_intermediate_features=False):
        super().__init__()
        blks = []
        for i in range(num_blocks):
            block_drop_path = drop_path[i] if isinstance(drop_path, list) else drop_path
            #   Assume all RepLK Blocks within a stage share the same lk_size. You may tune it on your own model.
            replk_block = RepLKBlock(in_channels=channels, dw_channels=int(channels * dw_ratio), block_lk_size=stage_lk_size,
                                     small_kernel=small_kernel, drop_path=block_drop_path, small_kernel_merged=small_kernel_merged)
            convffn_block = ConvFFN(in_channels=channels, internal_channels=int(channels * ffn_ratio), out_channels=channels,
                                    drop_path=block_drop_path)
            blks.append(replk_block)
            blks.append(convffn_block)
        if norm_intermediate_features:
            blks.append(nn.BatchNorm2D(channels))
        #     self.norm = nn.BatchNorm2D(channels)    #   Only use this with RepLKNet-XL on downstream tasks
        # else:
        #     self.norm = nn.Identity()
        self.blocks = nn.LayerList(blks)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class RepLKNet(nn.Layer):
    def __init__(self, large_kernel_sizes, layers, channels, drop_path_rate, small_kernel,
                 dw_ratio=1, ffn_ratio=4, in_channels=3, num_classes=1000, out_indices=None,
                 small_kernel_merged=True,
                 norm_intermediate_features=False   
                 ):
        super().__init__()

        if num_classes is None and out_indices is None:
            raise ValueError('must specify one of num_classes (for pretraining) and out_indices (for downstream tasks)')
        elif num_classes is not None and out_indices is not None:
            raise ValueError('cannot specify both num_classes (for pretraining) and out_indices (for downstream tasks)')
        elif num_classes is not None and norm_intermediate_features:
            raise ValueError('for pretraining, no need to normalize the intermediate feature maps')
        self.out_indices = out_indices
    
        base_width = channels[0]
        self.norm_intermediate_features = norm_intermediate_features
        self.num_stages = len(layers)
        self.stem = nn.LayerList([
            convbnrelu(in_channels=in_channels, out_channels=base_width, kernel_size=3, stride=2, padding=1, groups=1),
            convbnrelu(in_channels=base_width, out_channels=base_width, kernel_size=3, stride=1, padding=1, groups=base_width),
            convbnrelu(in_channels=base_width, out_channels=base_width, kernel_size=1, stride=1, padding=0, groups=1),
            convbnrelu(in_channels=base_width, out_channels=base_width, kernel_size=3, stride=2, padding=1, groups=base_width)])
        # stochastic depth. We set block-wise drop-path rate. The higher level blocks are more likely to be dropped. This implementation follows Swin.
        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, sum(layers))]
        self.stages = nn.LayerList()
        self.transitions = nn.LayerList()
        for stage_idx in range(self.num_stages):
            layer = RepLKNetStage(channels=channels[stage_idx], num_blocks=layers[stage_idx],
                                  stage_lk_size=large_kernel_sizes[stage_idx],
                                  drop_path=dpr[sum(layers[:stage_idx]):sum(layers[:stage_idx + 1])],
                                  small_kernel=small_kernel, dw_ratio=dw_ratio, ffn_ratio=ffn_ratio,
                                  small_kernel_merged=small_kernel_merged,
                                  norm_intermediate_features=norm_intermediate_features)
            self.stages.append(layer)
            if stage_idx < len(layers) - 1:
                transition = nn.Sequential(
                    convbnrelu(channels[stage_idx], channels[stage_idx + 1], 1, 1, 0, groups=1),
                    convbnrelu(channels[stage_idx + 1], channels[stage_idx + 1], 3, stride=2, padding=1, groups=channels[stage_idx + 1]))
                self.transitions.append(transition)

        if num_classes is not None:
            self.norm = nn.BatchNorm2D(channels[-1])
            self.avgpool = nn.AdaptiveAvgPool2D(1)
            self.head = nn.Linear(channels[-1], num_classes)


    def forward_features(self, x):
        x = self.stem[0](x)
        for stem_layer in self.stem[1:]:
            x = stem_layer(x)
        print(x.shape)
        if self.out_indices is None:
            #   Just need the final output
            for stage_idx in range(self.num_stages):
                x = self.stages[stage_idx](x)
                print(x.shape, stage_idx)
                if stage_idx < self.num_stages - 1:
                    x = self.transitions[stage_idx](x)
            return x
        else:
            #   Need the intermediate feature maps
            outs = []
            for stage_idx in range(self.num_stages):
                x = self.stages[stage_idx](x)
                if stage_idx in self.out_indices:
                    outs.append(self.stages[stage_idx].norm(x))     # For RepLKNet-XL normalize the features before feeding them into the heads
                if stage_idx < self.num_stages - 1:
                    x = self.transitions[stage_idx](x)
            return outs

    def forward(self, x):
        x = self.forward_features(x)
        if self.out_indices:
            return x
        else:
            x = self.norm(x)
            x = self.avgpool(x)
            x = paddle.flatten(x, 1)
            x = self.head(x)
            return x

    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()

    #   If your framework cannot automatically fuse BN for inference, you may do it manually.
    #   The BNs after and before conv layers can be removed.
    #   No need to call this if your framework support automatic BN fusion.
    def deep_fuse_BN(self):
        for m in self.modules():
            if not isinstance(m, nn.Sequential):
                continue
            if not len(m) in [2, 3]:  # Only handle conv-BN or conv-BN-relu
                continue
            #   If you use a custom Conv2d impl, assume it also has 'kernel_size' and 'weight'
            if hasattr(m[0], 'kernel_size') and hasattr(m[0], 'weight') and isinstance(m[1], nn.BatchNorm2D):
                conv = m[0]
                bn = m[1]
                fused_kernel, fused_bias = fuse_bn(conv, bn)
                fused_conv = conv2d(conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size,
                                        stride=conv.stride,
                                        padding=conv.padding, dilation=conv.dilation, groups=conv.groups, bias_attr=True)
                fused_conv.weight.data = fused_kernel
                fused_conv.bias.data = fused_bias
                m[0] = fused_conv
                m[1] = nn.Identity()


class RepLKNet31B(nn.Layer):
    def __init__(self, channels=[64,128,256,512],large_kernel_sizes=[31,29,27,13], 
                 layers=[2,2,18,2], drop_path_rate=0.3, small_kernel=5,
                 dw_ratio=1, ffn_ratio=4, out_indices=[0,1,2,3],
                 small_kernel_merged=False,
                 norm_intermediate_features=False   
                 ):
        super().__init__()

        self.out_indices = out_indices
    
        self.norm_intermediate_features = norm_intermediate_features
        self.num_stages = len(layers)-1
        # stochastic depth. We set block-wise drop-path rate. The higher level blocks are more likely to be dropped. This implementation follows Swin.
        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, sum(layers))]
        self.stages = nn.LayerList()
        self.transitions = nn.LayerList()
        for stage_idx in range(self.num_stages):
            stage_idx += 1
            
            transition = nn.Sequential(
                convbnrelu(channels[stage_idx-1], channels[stage_idx], 1, 1, 0, groups=1),
                convbnrelu(channels[stage_idx], channels[stage_idx], 3, stride=2, padding=1, groups=channels[stage_idx]))
            self.transitions.append(transition)

            layer = RepLKNetStage(channels=channels[stage_idx], num_blocks=layers[stage_idx],
                                  stage_lk_size=large_kernel_sizes[stage_idx],
                                  drop_path=dpr[sum(layers[:stage_idx]):sum(layers[:stage_idx + 1])],
                                  small_kernel=small_kernel, dw_ratio=dw_ratio, ffn_ratio=ffn_ratio,
                                  small_kernel_merged=small_kernel_merged,
                                  norm_intermediate_features=norm_intermediate_features)
            self.stages.append(layer)
            

    def forward(self, x):
        outs = []
        for stage_idx in range(self.num_stages):
            x = self.transitions[stage_idx](x)
            x = self.stages[stage_idx](x)
            outs.append(x)     # For RepLKNet-XL normalize the features before feeding them into the heads 
        return outs
        

    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()

    #   If your framework cannot automatically fuse BN for inference, you may do it manually.
    #   The BNs after and before conv layers can be removed.
    #   No need to call this if your framework support automatic BN fusion.
    def deep_fuse_BN(self):
        for m in self.modules():
            if not isinstance(m, nn.Sequential):
                continue
            if not len(m) in [2, 3]:  # Only handle conv-BN or conv-BN-relu
                continue
            #   If you use a custom Conv2d impl, assume it also has 'kernel_size' and 'weight'
            if hasattr(m[0], 'kernel_size') and hasattr(m[0], 'weight') and isinstance(m[1], nn.BatchNorm2D):
                conv = m[0]
                bn = m[1]
                fused_kernel, fused_bias = fuse_bn(conv, bn)
                fused_conv = conv2d(conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size,
                                        stride=conv.stride,
                                        padding=conv.padding, dilation=conv.dilation, groups=conv.groups, bias_attr=True)
                fused_conv.weight.data = fused_kernel
                fused_conv.bias.data = fused_bias
                m[0] = fused_conv
                m[1] = nn.Identity()


def create_RepLKNet31B(in_channels = 3, drop_path_rate=0.3, num_classes=1000, small_kernel_merged=False):
    return RepLKNet(in_channels=in_channels, large_kernel_sizes=[31,29,27,13], layers=[2,2,18,2], 
                    drop_path_rate=drop_path_rate, small_kernel=5, num_classes=num_classes, channels=[128,256,512,1024],
                    small_kernel_merged=small_kernel_merged)

def create_RepLKNet31L(in_channels = 3,drop_path_rate=0.3, num_classes=1000, small_kernel_merged=False):
    return RepLKNet(in_channels=in_channels,large_kernel_sizes=[31,29,27,13], layers=[2,2,18,2], channels=[192,384,768,1536],
                    drop_path_rate=drop_path_rate, small_kernel=5, num_classes=num_classes,
                    small_kernel_merged=small_kernel_merged)

def create_RepLKNetXL(in_channels = 6,drop_path_rate=0.3, num_classes=1000, small_kernel_merged=False):
    return RepLKNet(in_channels=in_channels,large_kernel_sizes=[27,27,27,13], layers=[2,2,18,2], 
                    drop_path_rate=drop_path_rate, small_kernel=None, dw_ratio=1.5,
                    num_classes=num_classes, channels=[256,512,1024,2048],
                    small_kernel_merged=small_kernel_merged)




if __name__ == "__main__":
    print("replknet")
    from paddleseg.utils import op_flops_funs
    x = paddle.rand([1,6,16,16]).cuda()
    m = create_RepLKNet31B(6).to('gpu:0')
    y = m(x)
    print(y.shape)
    paddle.flops(
        m, [1, 6, 16, 16],
        custom_ops={paddle.nn.SyncBatchNorm: op_flops_funs.count_syncbn})

