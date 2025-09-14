import paddle
import paddle.distribution
import paddle.nn as nn
import paddle.nn.functional as F
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from paddlenlp.transformers.mamba.modeling import MambaMixer, MambaConfig, MambaRMSNorm, MambaCache
from einops import rearrange, repeat
from paddlenlp.utils.initializer import constant_, kaiming_uniform_, normal_, uniform_, zeros_
from paddlenlp.transformers.activations import ACT2FN


class Adapter(nn.Layer):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU):
        super().__init__()
        
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.adaptor_fc1 = nn.Linear(D_features, D_hidden_features)
        self.adaptor_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.adaptor_fc1(x)
        xs = self.act(xs)
        xs = self.adaptor_fc2(xs)
        
        x = x + xs
        
        return x

class SFIF(nn.Layer):
    def __init__(self, in_features, out_features):
        super(SFIF, self).__init__()
        self.adaptor_ln = nn.Linear(2*in_features, out_features)
        self.adaptor_mamba = MambaLayer(out_features)
        self.adaptor_x = self.create_parameter([1], default_initializer=nn.initializer.Constant(0.5))
        self.adaptor_y = self.create_parameter([1], default_initializer=nn.initializer.Constant(0.5))

        self.reset_parameters()

    def reset_parameters(self):
        nn.initializer.KaimingNormal()(self.adaptor_ln.weight)
        nn.initializer.Constant(0.5)(self.adaptor_x)
        nn.initializer.Constant(0.5)(self.adaptor_y)
        self.adaptor_mamba._init_weights(self.adaptor_mamba)
    
    def forward(self, x, y):
        a = paddle.concat([x, y], -1)
        a = self.adaptor_ln(a)
        a = self.adaptor_mamba(a)
      
        return self.adaptor_x * a, self.adaptor_y * a
    

class MambaLayer(nn.Layer):
    def __init__(
            self,
            embed_dims=768,
            layer_norm_epsilon=1e-5,
            layer_idx=0,
            residual_in_fp32=True,
            **kwargs,
    ):
        super(MambaLayer, self).__init__()
        self.embed_dims = embed_dims
        self.config = MambaConfig(hidden_size = self.embed_dims,
                                  state_size=16, 
                                  conv_kernel=4,
                                  vocab_size=256,
                                  hidden_act="silu",
                                  use_bias=False,
                                  time_step_rank=math.ceil(self.embed_dims / 16),
                                  residual_in_fp32=residual_in_fp32,
                                  use_conv_bias=True,
                                  kwargs=kwargs)
        
        self.residual_in_fp32 = residual_in_fp32
        self.norm = MambaRMSNorm(self.embed_dims, eps=layer_norm_epsilon)
        self.mamba_layer = MambaMixer(self.config, layer_idx=layer_idx)

        # self._init_weights(self)
    def forward(self, x):
        residual = x
        if self.residual_in_fp32:
            residual = residual.cast(paddle.float32)
    
        x = self.norm(x)
        x = self.mamba_layer(x)

        x = residual + x
        return x
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, MambaMixer):
            module.A_log._no_weight_decay = True
            module.D._no_weight_decay = True

            dt_init_std = self.config.time_step_rank**-0.5 * self.config.time_step_scale
            if self.config.time_step_init_scheme == "constant":
                constant_(module.dt_proj.weight, dt_init_std)
            elif self.config.time_step_init_scheme == "random":
                uniform_(module.dt_proj.weight, -dt_init_std, dt_init_std)

            dt = paddle.exp(
                paddle.rand((self.config.intermediate_size,), dtype="float32").cast(paddle.get_default_dtype())
                * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                + math.log(self.config.time_step_min)
            ).clip(min=self.config.time_step_floor)
            # # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + paddle.log(-paddle.expm1(-dt))
            with paddle.no_grad():
                module.dt_proj.bias.copy_(inv_dt, False)
            module.dt_proj.bias._no_reinit = True

        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            normal_(module.weight, std=self.config.initializer_range)

        if self.config.rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["out_proj.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    kaiming_uniform_(p, a=math.sqrt(5))
                    with paddle.no_grad():
                        p.copy_(p / math.sqrt(self.config.num_layers), False)

class MambaBlock(nn.Layer):
    def __init__(self, hidden_size, num_hidden_layers=16, vocab_size=128, residual_in_fp32=True):
        super().__init__()
        self.config = MambaConfig(vocab_size, hidden_size, num_hidden_layer=num_hidden_layers)
        self.residual_in_fp32 = residual_in_fp32
        
        self.layers = nn.LayerList()
        for idx in range(num_hidden_layers):
            self.layers.append(MambaRMSNorm(hidden_size))
            self.layers.append(MambaMixer(self.config, layer_idx=idx))
        
        self._init_weights(self)

    def forward(self, x):
        residual = x
        if self.residual_in_fp32:
            residual = residual.cast(paddle.float32)

        for layer in self.layers:
            x = layer(x)
            x = residual + x
        # x = residual + x
        return x

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, MambaMixer):
            module.A_log._no_weight_decay = True
            module.D._no_weight_decay = True

            dt_init_std = self.config.time_step_rank**-0.5 * self.config.time_step_scale
            if self.config.time_step_init_scheme == "constant":
                constant_(module.dt_proj.weight, dt_init_std)
            elif self.config.time_step_init_scheme == "random":
                uniform_(module.dt_proj.weight, -dt_init_std, dt_init_std)

            dt = paddle.exp(
                paddle.rand((self.config.intermediate_size,), dtype="float32").cast(paddle.get_default_dtype())
                * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                + math.log(self.config.time_step_min)
            ).clip(min=self.config.time_step_floor)
            # # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + paddle.log(-paddle.expm1(-dt))
            with paddle.no_grad():
                module.dt_proj.bias.copy_(inv_dt, False)
            module.dt_proj.bias._no_reinit = True

        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            normal_(module.weight, std=self.config.initializer_range)

        if self.config.rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["out_proj.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    kaiming_uniform_(p, a=math.sqrt(5))
                    with paddle.no_grad():
                        p.copy_(p / math.sqrt(self.config.num_layers), False)

