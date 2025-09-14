import re
import time
import math
import numpy as np
from numpy.random import RandomState
from functools import partial
from typing import Optional, Union, Type, List, Tuple, Callable, Dict, Sequence

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.models import layers
from paddleseg.utils import load_entire_model
from einops import rearrange

from .utils import MLPBlock
from ssm import MambaLayer

class BitemporalChangeInformationFusion(nn.Layer):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.fc1 = nn.Linear(2 * in_dims, in_dims)
        self.mamba = MambaLayer(in_dims)

        self.fc2 = nn.Linear(in_dims, in_dims)

        self.mlp = MLPBlock(in_dims, 2 * in_dims)
        self.fc3 = nn.Linear(in_dims, out_dims)

    def forward(self, x, y):
        c = paddle.concat([x, y], axis=-1)
        c = self.fc1(c)
        c = self.mamba(c)
        c = self.fc2(c)
        # y = self.fc2(y)

        y = c + y
        y = self.mlp(y) 
        y = self.fc3(y)
        return y
    
    
class Local_Feature_Gather(nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dim_sp = dim // 2

        self.CDilated_1 = nn.Conv2D(self.dim_sp, self.dim_sp, 3, stride=1, padding=1, dilation=1, groups=self.dim_sp)
        self.CDilated_2 = nn.Conv2D(self.dim_sp, self.dim_sp, 3, stride=1, padding=2, dilation=2, groups=self.dim_sp)
        self.CK1 = nn.Conv2D(dim, dim, 1)

    def forward(self, x):
        x1, x2 = paddle.chunk(x, 2, axis=1)
        cd1 = self.CDilated_1(x1)
        cd2 = self.CDilated_2(x2)
        y = paddle.concat([cd1, cd2], axis=1)
        y = y + self.CK1(x)
        return y
    

class MultiScaleFeatureAggregation(nn.Layer):
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