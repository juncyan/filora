import paddle
from paddle import nn
from paddle.nn import functional as F

from typing import Any, Dict, List, Union
from functools import partial

from paddleseg.utils import load_entire_model

from cd_models.segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer, TinyViT

class MobileSAM(nn.Layer):
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

        prompt_embed_dim = 256
        self.img_size = img_size
        vit_patch_size = 16 
        image_embedding_size = self.img_size // vit_patch_size

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
                layer_lr_decay=0.8
            )
        self.prompt_encoder = PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(img_size, img_size),
            mask_in_chans=16,
            )
        
        self.mask_decoder = MaskDecoder(
                    num_multimask_outputs=3,
                    transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
            )
        
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
        self.register_buffer(
            "pixel_mean",
            paddle.to_tensor(pixel_mean).reshape([-1, 1, 1]),
            persistable=False)
        self.register_buffer(
            "pixel_std",
            paddle.to_tensor(pixel_std).reshape([-1, 1, 1]),
            persistable=False)

    @property
    def device(self) -> Any:
        if paddle.is_compiled_with_cuda():
            return 'gpu'
        else:
            return 'cpu'

    # @paddle.no_grad()
    def forward(self, x, multimask_output: bool = False):
        image_embeddings = self.image_encoder(x)
        outputs = []
        # for curr_embedding in image_embeddings:
        for cur_img in image_embeddings:    
          sparse_embeddings, dense_embeddings = self.prompt_encoder(None,None,None,)
         
          low_res_masks, iou_predictions = self.mask_decoder(
              image_embeddings=cur_img.unsqueeze(0),
              image_pe=self.prompt_encoder.get_dense_pe(),
              sparse_prompt_embeddings=sparse_embeddings,
              dense_prompt_embeddings=dense_embeddings,
              multimask_output=multimask_output, )
          masks = self.postprocess_masks(
              low_res_masks,
              input_size=x.shape[-2:],
              original_size=[self.img_size,self.img_size], )
          masks = masks > self.mask_threshold
          
          outputs.append(masks)
          # {
          #     "masks": masks,
          #     "iou_predictions": iou_predictions,
          #     "low_res_logits": low_res_masks,
          # })
        return paddle.concat(outputs, axis=0)

    def postprocess_masks(
            self,
            masks,
            input_size,
            original_size):
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (paddle.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (paddle.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False, )

        masks = masks[..., :input_size[0], :input_size[1]]
        masks = F.interpolate(
            masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: paddle.tensor) -> paddle.tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
