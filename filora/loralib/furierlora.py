#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math
from einops import rearrange
from .layers import LoRALayer


class BiTemporalFrequencyFeatureFusion(nn.Layer):
    # BiTemporalFurierFeatureFusion
    def __init__(self, in_features: int, 
                    out_features: int=None, 
                    r: int = 8, 
                    lora_alpha: int = 1, 
                    lora_dropout: float = 0.,
                    merge_weights: bool = True,
                    **kwargs):
        super().__init__()
        # LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        #                    merge_weights=merge_weights)
        # dim = 2*in_features
        # self.lora_A = Linear(2*in_features, in_features, r=r)
        # self.ln = Linear(in_features, in_features, r=r, bias_attr=False)
        # nn.Conv1D(data_format='NLC')
        self.ln = nn.Conv1D(in_channels=in_features, out_channels=in_features, kernel_size=3, padding=1, 
                         groups = in_features, bias_attr=False, data_format='NLC')
        
        self.wx = self.create_parameter(shape=[1], dtype='float32', 
                    default_initializer=nn.initializer.Constant(0.3))
        self.wy = self.create_parameter(shape=[1], dtype='float32', 
                    default_initializer=nn.initializer.Constant(0.3))
        
    def furier_feature_transfer(self, x):
        b, l, c = x.shape
        # (batch, c, l//2 + 1, 2)
        ffted = paddle.fft.rfft(x, axis=1, norm='ortho')
        
        x_fft_real = paddle.unsqueeze(ffted.real(), axis=2)
        x_fft_imag = paddle.unsqueeze(ffted.imag(), axis=2)
       
        ffted = paddle.concat((x_fft_real, x_fft_imag), axis=2)
        ffted = ffted.reshape([b, -1, c*2])
        ffted = self.ln(ffted)
        ffted = ffted.reshape([b, -1, 2, c])
        ffted = paddle.complex(ffted[..., 0, :], ffted[..., 1, :])
        
        ffted = paddle.fft.irfft(ffted, n=l, axis=1, norm='ortho')
        return ffted
    
    def forward(self, x1: paddle.Tensor, x2: paddle.Tensor):
        aux = self.furier_feature_transfer(x1+x2)
        
        x1 = x1 + self.wx * aux
        x2 = x2 + self.wy * aux
        return x1, x2
    

class SVDLinear(nn.Linear, LoRALayer):
    # SVD-based adaptation implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, 
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = paddle.create_parameter(
                shape=[r, in_features],
                dtype=self.weight.dtype,
                default_initializer=paddle.nn.initializer.Constant(0.0)
            )
            self.lora_E = paddle.create_parameter(
                shape=[r, 1],
                dtype=self.weight.dtype,
                default_initializer=paddle.nn.initializer.Constant(0.0)
            ) 
            self.lora_B = paddle.create_parameter(
                shape=[out_features, r],
                dtype=self.weight.dtype,
                default_initializer=paddle.nn.initializer.Constant(0.0)
            )
            self.rank = self.r
            self.scaling = self.lora_alpha if self.lora_alpha > 0 else float(self.r)   
            # Freezing the pre-trained weight matrix
            self.weight.stop_gradient = True
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.set_value(self.weight.transpose([1, 0]))

    def reset_parameters(self):
        nn.initializer.Normal(mean=0.0, std=0.02)(self.weight)
        if hasattr(self, 'lora_A'): 
            nn.initializer.Normal(mean=0.0, std=0.02)(self.lora_E)
            nn.initializer.Normal(mean=0.0, std=0.02)(self.lora_A)
            nn.initializer.Normal(mean=0.0, std=0.02)(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose([1, 0]) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if self.merge_weights and getattr(self, 'merged', False):
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.set_value(self.weight - T(
                    paddle.matmul(self.lora_B, self.lora_A * self.lora_E)
                ) * self.scaling / (self.rank + 1e-5))
            self.merged = False
    
    def eval(self):
        def T(w):
            return w.transpose([1, 0]) if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not getattr(self, 'merged', False):
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.set_value(self.weight + T(
                    paddle.matmul(self.lora_B, self.lora_A * self.lora_E)
                ) * self.scaling / (self.rank + 1e-5))
            self.merged = True

    def forward(self, x: paddle.Tensor):
        def T(w):
            return w.transpose([1, 0]) if self.fan_in_fan_out else w
        if self.r > 0 and not getattr(self, 'merged', False):
            result = F.linear(x, T(self.weight), self.bias)
            if self.r > 0:
                result += (
                    self.lora_dropout(x) @ 
                    (self.lora_A * self.lora_E).transpose([1, 0]) @ self.lora_B.transpose([1, 0])
                ) * self.scaling / (self.rank + 1e-5)
            return result
        else:
            return F.linear(x, T(self.weight), self.bias)


class Conv1d(nn.Conv1D, LoRALayer):
    def __init__(self, in_channels, out_channels, kernel_size=1, r=4, lora_alpha=1, lora_dropout=0., merge_weights=True, **kwargs):
        nn.Conv1D.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert isinstance(kernel_size, int)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = self.create_parameter(
                shape=[r * kernel_size, in_channels * kernel_size], 
                dtype=self.weight.dtype,
                default_initializer=nn.initializer.Constant(0.0)
            )
            self.lora_B = self.create_parameter(
                shape=[out_channels // self._groups, r * kernel_size],
                dtype=self.weight.dtype,
                default_initializer=nn.initializer.Normal(0., 0.02)
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.stop_gradient = True
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        nn.initializer.Normal(mean=0.0, std=0.02)(self.weight)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.initializer.KaimingUniform()(self.lora_A)
            nn.initializer.Constant(0.0)(self.lora_B)

    def train(self, mode=True):
        super(nn.Conv1D, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # Make sure that the weights are not merged
                    delta = paddle.reshape(self.lora_B @ self.lora_A, self.weight.shape) * self.scaling
                    self.weight.set_value(self.weight - delta)
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # Merge the weights and mark it
                    delta = paddle.reshape(self.lora_B @ self.lora_A, self.weight.shape) * self.scaling
                    self.weight.set_value(self.weight + delta)
                self.merged = True

    def forward(self, x):
        if self.r > 0 and not self.merged:
            original_weight = self.weight
            delta_weight = paddle.reshape(self.lora_B @ self.lora_A, original_weight.shape) * self.scaling
            new_weight = original_weight + delta_weight
           
            # old_weight = self.weight
            # self.weight.set_value(new_weight)
            result = F.conv1d(x, new_weight, self.bias, self._stride,
                              self._padding, self._dilation, self._groups,
                              self._data_format)
            # self.weight.set_value(old_weight)
            return result
        return F.conv1d(x, self.weight, self.bias,self._stride,
                        self._padding, self._dilation, self._groups,
                        self._data_format)


class BiLinear(nn.Linear, LoRALayer):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_Ax = self.create_parameter(
                shape=[r, in_features], 
                dtype=self.weight.dtype,
                default_initializer=nn.initializer.Constant(0.0)
            )
            self.lora_Bx = self.create_parameter(
                shape=[out_features, r], 
                dtype=self.weight.dtype,
                default_initializer=nn.initializer.Constant(0.0)
            )
            self.lora_Ay = self.create_parameter(
                shape=[r, in_features], 
                dtype=self.weight.dtype,
                default_initializer=nn.initializer.Constant(0.0)
            )
            self.lora_By = self.create_parameter(
                shape=[out_features, r], 
                dtype=self.weight.dtype,
                default_initializer=nn.initializer.Constant(0.0)
            )
            self.lora_F = paddle.create_parameter(
                shape=[out_features * 2],
                dtype=self.weight.dtype,
                default_initializer=paddle.nn.initializer.Constant(1.0)
            )
            self.scaling = self.create_parameter([1], dtype=self.weight.dtype,
                                default_initializer=nn.initializer.Constant(self.lora_alpha / self.r))
            # Freezing the pre-trained weight matrix
            self.weight.stop_gradient = True
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.set_value(paddle.transpose(self.weight, [1, 0]))

    def reset_parameters(self):
        nn.initializer.Normal(mean=0.0, std=0.02)(self.weight)
        if hasattr(self, 'lora_Ax'):
            # initialize B the same way as the default for nn.Linear and A to zero
            nn.initializer.KaimingUniform()(self.lora_Ax)
            nn.initializer.Constant(0.0)(self.lora_Bx)
            nn.initializer.KaimingUniform()(self.lora_Ay)
            nn.initializer.Constant(0.0)(self.lora_By)
            nn.initializer.Constant(1.0)(self.lora_F)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)     

    def eval(self, mode: bool = True):
        nn.Linear.eval(self, mode)    

    def forward(self, x: paddle.Tensor, y:paddle.Tensor):
        def T(w):
            return paddle.transpose(w, [1, 0]) 
        resultx = F.linear(x, self.weight, self.bias) 
        resulty = F.linear(y, self.weight, self.bias)

        if self.r > 0: 
            aux = self.lora_dropout(x) @ T(self.lora_Ax) @ T(self.lora_Bx) + self.lora_dropout(y) @ T(self.lora_Ay) @ T(self.lora_By) 

            b, l, c = aux.shape
            # (batch, c, l//2 + 1, 2)
            ffted = paddle.fft.rfft(aux, axis=1, norm='ortho')
            
            x_fft_real = paddle.unsqueeze(ffted.real(), axis=2)
            x_fft_imag = paddle.unsqueeze(ffted.imag(), axis=2)
        
            ffted = paddle.concat((x_fft_real, x_fft_imag), axis=2)
            ffted = ffted.reshape([b, -1, c*2])
            ffted = self.lora_F * ffted
            ffted = ffted.reshape([b, -1, 2, c])
            ffted = paddle.complex(ffted[..., 0, :], ffted[..., 1, :])
            aux = paddle.fft.irfft(ffted, n=l, axis=1, norm='ortho')
         
            aux = self.scaling * aux   

            resultx = resultx + aux
            resulty = resulty + aux
      
        return resultx, resulty
