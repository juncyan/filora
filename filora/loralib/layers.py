#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import math
from typing import Optional, List

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Embedding(nn.Embedding, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0,
                           merge_weights=merge_weights)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = paddle.create_parameter(
                shape=[r, num_embeddings], 
                dtype=self.weight.dtype,
                default_initializer=nn.initializer.Constant(0.0)
            )
            self.lora_B = paddle.create_parameter(
                shape=[embedding_dim, r], 
                dtype=self.weight.dtype,
                default_initializer=nn.initializer.Normal()
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.stop_gradient = True
        self.reset_parameters()

    def reset_parameters(self):
        # nn.Embedding.reset_parameters(self)
        nn.initializer.Normal(mean=0.0, std=0.02)(self.weight)
        # self.weight.set_value(nn.initializer.Normal(mean=0.0, std=0.02))
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.initializer.Constant(0.0)(self.lora_A)
            nn.initializer.Normal()(self.lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.set_value(self.weight - paddle.transpose(self.lora_B @ self.lora_A, [1, 0]) * self.scaling)
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.set_value(self.weight + paddle.transpose(self.lora_B @ self.lora_A, [1, 0]) * self.scaling)
                self.merged = True
        
    def forward(self, x: paddle.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            after_A = F.embedding(
                x, paddle.transpose(self.lora_A, [1, 0]), self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse
            )
            result += (after_A @ paddle.transpose(self.lora_B, [1, 0])) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)
            

class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
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
            self.lora_A = paddle.create_parameter(
                shape=[r, in_features], 
                dtype=self.weight.dtype,
                default_initializer=nn.initializer.Constant(0.0)
            )
            self.lora_B = paddle.create_parameter(
                shape=[out_features, r], 
                dtype=self.weight.dtype,
                default_initializer=nn.initializer.Constant(0.0)
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.stop_gradient = True
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.set_value(paddle.transpose(self.weight, [1, 0]))

    def reset_parameters(self):
        nn.initializer.Normal(mean=0.0, std=0.02)(self.weight)
        if hasattr(self, 'lora_A'):
            # initialize B the same way as the default for nn.Linear and A to zero
            nn.initializer.KaimingUniform()(self.lora_A)
            nn.initializer.Constant(0.0)(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return paddle.transpose(w, [1, 0]) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.set_value(self.weight - T(self.lora_B @ self.lora_A) * self.scaling)
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.set_value(self.weight + T(self.lora_B @ self.lora_A) * self.scaling)
                self.merged = True       

    def forward(self, x: paddle.Tensor):
        def T(w):
            return paddle.transpose(w, [1, 0]) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), self.bias)            
            result += (self.lora_dropout(x) @ paddle.transpose(self.lora_A, [1, 0]) @ paddle.transpose(self.lora_B, [1, 0])) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), self.bias)


class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = paddle.create_parameter(
                shape=[r * sum(enable_lora), in_features], 
                dtype=self.weight.dtype,
                default_initializer=nn.initializer.Constant(0.0)
            )
            self.lora_B = paddle.create_parameter(
                shape=[out_features // len(enable_lora) * sum(enable_lora), r],
                dtype=self.weight.dtype,
                default_initializer=nn.initializer.Constant(0.0)
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.stop_gradient = True
            # Compute the indices
            self.lora_ind = paddle.zeros([out_features], dtype='bool')
            self.lora_ind = paddle.reshape(self.lora_ind, [len(enable_lora), -1])
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = paddle.reshape(self.lora_ind, [-1])
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.set_value(paddle.transpose(self.weight, [1, 0]))

    def reset_parameters(self):
        # nn.Linear.reset_parameters(self)
        nn.initializer.Normal(mean=0.0, std=0.02)(self.weight)
        # self.weight.set_value(nn.initializer.Normal(mean=0.0, std=0.02))
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.initializer.KaimingUniform()(self.lora_A)
            nn.initializer.Constant(0.0)(self.lora_B)

    def zero_pad(self, x):
        result = paddle.zeros([len(self.lora_ind)] + list(x.shape[1:]), dtype=x.dtype)
        result = paddle.scatter(result, paddle.arange(len(self.lora_ind)), x.expand([len(self.lora_ind)] + list(x.shape[1:])))
        return result

    def merge_AB(self):
        def T(w):
            return paddle.transpose(w, [1, 0]) if self.fan_in_fan_out else w
        delta_w = F.conv1d(
            paddle.unsqueeze(self.lora_A, 0), 
            paddle.unsqueeze(self.lora_B, -1), 
            groups=sum(self.enable_lora)
        ).squeeze(0)
        return T(self.zero_pad(delta_w))

    def train(self, mode: bool = True):
        def T(w):
            return paddle.transpose(w, [1, 0]) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):
                    self.weight.set_value(self.weight - self.merge_AB() * self.scaling)
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    self.weight.set_value(self.weight + self.merge_AB() * self.scaling)
                self.merged = True        

    def forward(self, x: paddle.Tensor):
        def T(w):
            return paddle.transpose(w, [1, 0]) if self.fan_in_fan_out else w
        if self.merged:
            return F.linear(x, T(self.weight), self.bias)
        else:
            result = F.linear(x, T(self.weight), self.bias)
            if self.r > 0:
                result += self.lora_dropout(x) @ paddle.transpose(T(self.merge_AB()), [1, 0]) * self.scaling
            return result

class ConvLoRA(nn.Layer, LoRALayer):
    def __init__(self, conv_module, in_channels, out_channels, kernel_size=1, r=0, lora_alpha=1, lora_dropout=0., merge_weights=True, **kwargs):
        super(ConvLoRA, self).__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        # Register parameters manually since we can't use reset_parameters in Paddle like in PyTorch
        for name, param in self.conv.named_parameters():
            self.add_parameter(name, param)
            self._parameters[name].stop_gradient = param.stop_gradient
        
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert isinstance(kernel_size, int)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = paddle.create_parameter(
                shape=[r * kernel_size, in_channels * kernel_size], 
                dtype=self.conv.weight.dtype,
                default_initializer=nn.initializer.Constant(0.0)
            )
            self.lora_B = paddle.create_parameter(
                shape=[out_channels // self.conv._groups, r * kernel_size],
                dtype=self.conv.weight.dtype,
                default_initializer=nn.initializer.Normal(0., 0.02)
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.conv.weight.stop_gradient = True
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        # self.conv.weight.set_value(nn.initializer.Normal(mean=0.0, std=0.02))
        nn.initializer.Normal(mean=0.0, std=0.02)(self.conv.weight)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.initializer.KaimingUniform()(self.lora_A)
            nn.initializer.Constant(0.0)(self.lora_B)

    def train(self, mode=True):
        super(ConvLoRA, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # Make sure that the weights are not merged
                    delta = paddle.reshape(self.lora_B @ self.lora_A, self.conv.weight.shape) * self.scaling
                    self.conv.weight.set_value(self.conv.weight - delta)
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # Merge the weights and mark it
                    delta = paddle.reshape(self.lora_B @ self.lora_A, self.conv.weight.shape) * self.scaling
                    self.conv.weight.set_value(self.conv.weight + delta)
                self.merged = True

    def forward(self, x):
        if self.r > 0 and not self.merged:
            original_weight = self.conv.weight
            t = self.lora_B @ self.lora_A
            print(t.shape, original_weight.shape)
            delta_weight = paddle.reshape(self.lora_B @ self.lora_A, original_weight.shape) * self.scaling
            new_weight = original_weight + delta_weight
            # Paddle's conv2d doesn't support direct weight passing like PyTorch, so we need to hack it
            # This is a workaround by temporarily setting the weight and then restoring it
            old_weight = self.conv.weight
            self.conv.weight.set_value(new_weight)
            result = self.conv(x)
            self.conv.weight.set_value(old_weight)
            return result
        return self.conv(x)

class Conv2d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(nn.Conv2D, *args, **kwargs)

class Conv1d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(nn.Conv1D, *args, **kwargs)

class Conv3d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv3d, self).__init__(nn.Conv3D, *args, **kwargs)