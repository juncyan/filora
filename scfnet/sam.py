import paddle
from paddle import nn
from paddle.nn import functional as F

from typing import Any, Dict, List, Union
from functools import partial

from paddleseg.utils import load_entire_model

from segment_anything.modeling.ad_tiny_vit import TinyViT
# from .segment_anything.modeling.bflora import BFLoraViT


class MSAM(nn.Layer):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(self, img_size):
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.img_size = [img_size, img_size]

        self.image_encoder = TinyViT(img_size=img_size, in_chans=3, num_classes=1000,
                embed_dims=[64, 128, 160, 320],
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 5, 10],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.,
                drop_rate=0.,
                drop_path_rate=0.0,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
                layer_lr_decay=0.8,
            )
    
      

