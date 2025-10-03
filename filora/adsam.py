import paddle
from paddle import nn
from paddle.nn import functional as F

from typing import Any, Dict, List, Union
from functools import partial

from paddleseg.utils import load_entire_model

# from .segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer, TinyViT
from .segment_anything.modeling.adtinyvit import TinyViT
from .segment_anything.modeling.ad_encoder import ImageEncoderViT



def build_sam_vit_h(img_size=1024, rank=4):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        img_size=img_size,
        checkpoint=r"/home/jq/Code/weights/vit_h.pdparams", 
        rank=rank)


build_sam = build_sam_vit_h


def build_sam_vit_l(img_size=1024, rank=4):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
         img_size=img_size,
        checkpoint=r"/home/jq/Code/weights/vit_l.pdparams", 
        rank=rank)


def build_sam_vit_b(img_size=1024, rank=4):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        img_size=img_size,
        checkpoint=r"/home/jq/Code/weights/vit_b.pdparams",
        rank=rank)

def build_sam_vit_t(img_size=1024, rank=4):
    prompt_embed_dim = 256
    image_size = img_size
    vit_patch_size = 16 
    image_embedding_size = image_size // vit_patch_size
    mobile_sam = Sam(
            image_encoder=TinyViT(img_size=img_size, in_chans=3, num_classes=1000,
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
                rank=rank
            ),
        )

    mobile_sam.eval()
    load_entire_model(mobile_sam, r"/home/jq/Code/weights/vit_t.pdparams")
    mobile_sam.image_encoder.build_abs()
    return mobile_sam


sam_model_registry = {
    "default": build_sam,
    "vit_h": build_sam,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
    "vit_t": build_sam_vit_t,
}

def _build_sam(
        encoder_embed_dim,
        encoder_depth,
        encoder_num_heads,
        encoder_global_attn_indexes,
        img_size=1024,
        checkpoint=None, 
        rank=4):
    prompt_embed_dim = 256
    image_size = img_size
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(
                paddle.nn.LayerNorm, epsilon=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
             rank=rank ) )
    sam.eval()
    load_entire_model(sam, checkpoint)
    return sam

class Sam(nn.Layer):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
            self,
            image_encoder: Union[ImageEncoderViT, TinyViT] ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
