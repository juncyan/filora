import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
        
class Reparams(nn.Layer):
    def __init__(self, **kwagrs):
        super().__init__()
        self.in_channels = 3
        self.lk = 7
        self.stride = 1

        # if kernel_size == 17:
        #     self.kernel_sizes = [5, 9, 3, 3, 3]
        #     self.dilates = [1, 2, 4, 5, 7]
        # elif kernel_size == 15:
        #     self.kernel_sizes = [5, 7, 3, 3, 3]
        #     self.dilates = [1, 2, 3, 5, 7]
        # elif kernel_size == 13:
        #     self.kernel_sizes = [5, 7, 3, 3, 3]
        #     self.dilates = [1, 2, 3, 4, 5]
        # elif kernel_size == 11:
        #     self.kernel_sizes = [5, 5, 3, 3, 3]
        #     self.dilates = [1, 2, 3, 4, 5]
        # elif kernel_size == 9:
        #     self.kernel_sizes = [5, 5, 3, 3]
        #     self.dilates = [1, 2, 3, 4]
        # elif kernel_size == 7:
        #     self.kernel_sizes = [5, 3, 3]
        #     self.dilates = [1, 2, 3]
        # elif kernel_size == 5:
        #     self.kernel_sizes = [3, 3]
        #     self.dilates = [1, 2]

    def _fuse_conv_bn(self, branch):
        if isinstance(branch, nn.Conv2D):
            weight = branch.weight
            d = branch._dilation[0]
            bias = branch.bias
            weight = self._pad_tensor_seize_to_k(weight, self.lk, d)
            return weight, bias
        
        assert isinstance(branch, nn.Layer)
        conv = branch._conv
        kernel = conv.weight
        d = conv._dilation[0]
        bias = conv.bias
        if not hasattr(branch, "_batch_norm"):
            fuse_weight = self._pad_tensor_seize_to_k(fuse_weight, self.lk, d)
            return fuse_weight, bias
        
        bn = branch._batch_norm
        running_mean = bn._mean
        runing_var = bn._variance
        gamma = bn.weight
        beta = bn.bias
        eps = bn._epsilon
        std = (runing_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        fuse_weight = kernel * t
        fuse_weight = self._pad_tensor_seize_to_k(fuse_weight, self.lk, d)
        return fuse_weight, beta - running_mean * gamma / std
        return fuse_weight, beta + (bias- running_mean) * gamma / std
    
    def _fuse_conv(self, m):
        if m is None:
            return 0, 0
        if isinstance(m, nn.Conv2D):
            w = m.weight          
            b = m.bias
            d = m._dilation[0]
            kw = self._pad_tensor_seize_to_k(w, self.lk, d)
        return kw, b
 
    def _pad_tensor_seize_to_k(self, kernel, k, dilation=1):
        if kernel is None:
            return 0
        if dilation == 1:
            smallk = kernel.shape[-1]
            if smallk == k:
                return kernel
            pk = (k - smallk) // 2
            return nn.functional.pad(kernel, [pk,pk,pk,pk])
        if dilation > 1:
            equivalent_kernel = self.convert_dilated_to_nondilated(kernel, dilation)
            smallk = equivalent_kernel.shape[-1]
            if smallk == k:
                return equivalent_kernel
            pk = (k - smallk) // 2
            dilated =  F.pad(equivalent_kernel, [pk,pk,pk,pk])
            return dilated
    
    def convert_dilated_to_nondilated(self, kernel, dilation):
        identity_kernel = paddle.zeros((1, 1, 1, 1))
        c = kernel.shape[1]
        if c == 1:
            #   This is a DW kernel
            dilated = F.conv2d_transpose(kernel, identity_kernel, stride=dilation)
            return dilated
        else:
            #   This is a dense or group-wise (but not DW) kernel
            slices = []
            for i in range(c):
                dilated = F.conv2d_transpose(kernel[:,i:i+1,:,:], identity_kernel, stride=dilation)
                # print(dilated.shape)
                slices.append(dilated)
        return paddle.concat(slices, axis=1)
    
    def get_equivalent_kernel_bias(self):
        return 0, 0
    
    def eval(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        if not hasattr(self, 'repc'):
            self.repc = nn.Conv2D(self.in_channels, self.in_channels, self.lk, stride=self.stride, padding=self.lk//2, groups=self.in_channels, bias_attr=bias is not None)
        self.training = False
        self.repc.weight.set_value(kernel)
        
        if bias is not None:
            self.repc.bias.set_value(bias)

        for layer in self.sublayers():
            layer.eval()
    
    # def train(self):
    #     if hasattr(self, "repc"):
    #         delattr(self, "repc")
    #     return super().train()

    def repparams_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return kernel.numpy(), bias.numpy()


class RepConvBn(nn.Layer):
    def __init__(self):
        super().__init__()
        self.lk = 3

    def _repparams(self, conv:nn.Conv2D=None, bn:nn.BatchNorm2D=None):
        weight = conv.weight
        bias = conv.bias
        d = conv._dilation[0]
        if bn == None:
            fuse_weight = self._pad_tensor_seize_to_k(weight,d)
            return weight, bias
        running_mean = bn._mean
        runing_var = bn._variance
        gamma = bn.weight
        beta = bn.bias
        eps = bn._epsilon
        std = (runing_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        fuse_weight = weight * t
        return fuse_weight, beta + (bias- running_mean) * gamma / std

    def _pad_tensor_seize_to_k(self, kernel, dilation=1):
        if kernel is None:
            return 0
        if dilation == 1:
            smallk = kernel.shape[-1]
            if smallk == self.lk:
                return kernel
            pk = (self.lk - smallk) // 2
            return F.pad(kernel, [pk,pk,pk,pk])
        if dilation > 1:
            equivalent_kernel = self.convert_dilated_to_nondilated(kernel, dilation)
            smallk = equivalent_kernel.shape[-1]
            if smallk == self.lk:
                return equivalent_kernel
            pk = (self.lk - smallk) // 2
            dilated =  F.pad(equivalent_kernel, [pk,pk,pk,pk])
            return dilated
    
    def convert_dilated_to_nondilated(self, kernel, dilation):
        identity_kernel = paddle.zeros((1, 1, 1, 1))
        c = kernel.shape[1]
        if c == 1:
            dilated = F.conv2d_transpose(kernel, identity_kernel, stride=dilation)
            return dilated
        else:
            slices = []
            for i in range(c):
                dilated = F.conv2d_transpose(kernel[:,i:i+1,:,:], identity_kernel, stride=dilation)
                slices.append(dilated)
        return paddle.concat(slices, axis=1)
    
    def get_equivalent_kernel_bias(self):
        return 0, None
    
    def eval(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        if not hasattr(self, 'repc'):
            self.repc = nn.Conv2D(self.in_channels, self.in_channels, self.lk, padding=self.lk//2, groups=self.in_channels, bias_attr=bias is not None)
        self.training = False
        self.repc.weight.set_value(kernel)
        
        if bias is not None:
            self.repc.bias.set_value(bias)

        for layer in self.sublayers():
            layer.eval()


        