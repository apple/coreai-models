# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""SAM3 Lite model for iOS.

Assembles the ANE-targeted image encoder, FPN neck, text encoder, DETR
encoder/decoder, mask decoder, and dot-product scoring into a single
forward pass that mirrors the IO of HF ``Sam3Model``.

Usage:
    model = Sam3Lite.from_pretrained("facebook/sam3", image_size=336)
    pred_masks, pred_boxes, pred_logits, presence_logits, semantic_seg =
     model(pixel_values, input_ids)
"""

import torch
import torch.nn as nn

from coreai_models.models.ios.sam3.detr import (
    DETRDecoderReauthored,
    DETREncoderReauthored,
)
from coreai_models.models.ios.sam3.fpn import FPNNeckReauthored
from coreai_models.models.ios.sam3.image_encoder import ImageEncoderBackbone
from coreai_models.models.ios.sam3.mask_decoder import (
    DotProductScoringReauthored,
    MaskDecoderReauthored,
)
from coreai_models.models.ios.sam3.text_encoder import TextEncoderReauthored


def _linear_to_conv2d(linear: nn.Linear) -> nn.Conv2d:
    in_f = linear.in_features
    out_f = linear.out_features
    has_bias = linear.bias is not None
    conv = nn.Conv2d(in_f, out_f, 1, bias=has_bias)
    conv.weight.data = linear.weight.data.reshape(out_f, in_f, 1, 1)
    if has_bias:
        conv.bias.data = linear.bias.data
    return conv


class Sam3Lite(nn.Module):
    """SAM3 Lite model for iOS (palettized).

    The default ``image_size=336`` keeps the global-attention sequence
    short. Pass ``image_size=1008`` to match HF's default.

    Inputs:
        pixel_values: (B, 3, image_size, image_size) float tensor.
        input_ids: (B, seq_len) int32 token IDs.

    Outputs:
        Tuple ``(pred_masks, pred_boxes, pred_logits, presence_logits, semantic_seg)``.
    """

    def __init__(self, image_size: int = 336) -> None:
        super().__init__()
        self.image_size = image_size
        self.grid_size = image_size // 14  # patch_size = 14

        self.image_encoder: ImageEncoderBackbone | None = None
        self.fpn: FPNNeckReauthored | None = None
        self.text_encoder: TextEncoderReauthored | None = None
        self.text_projection: nn.Conv2d | None = None
        self.detr_encoder: DETREncoderReauthored | None = None
        self.detr_decoder: DETRDecoderReauthored | None = None
        self.mask_decoder: MaskDecoderReauthored | None = None
        self.scoring: DotProductScoringReauthored | None = None

        # Stored as a buffer so torch.export doesn't trace tensor creation
        # inside forward().
        self.register_buffer(
            "spatial_shapes",
            torch.tensor([[self.grid_size, self.grid_size]], dtype=torch.long),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        backbone_features = self.image_encoder(pixel_values)

        fpn_hidden_states, fpn_position_encoding = self.fpn(backbone_features)
        fpn_hidden_states_trimmed = fpn_hidden_states[:-1]
        fpn_position_encoding_trimmed = fpn_position_encoding[:-1]

        text_hidden_states = self.text_encoder(input_ids)
        text_features = self.text_projection(text_hidden_states)

        vision_level2 = fpn_hidden_states_trimmed[-1]
        B = vision_level2.shape[0]
        vision_bc1s = vision_level2.reshape(B, 256, 1, -1)

        pos_level2 = fpn_position_encoding_trimmed[-1]
        pos_bc1s = pos_level2.reshape(1, 256, 1, -1).expand(B, -1, -1, -1)

        encoder_output = self.detr_encoder(
            vision_feats=vision_bc1s,
            text_feats=text_features,
            vision_pos=pos_bc1s,
        )

        final_hidden_states, pred_boxes, presence_logits = self.detr_decoder(
            vision_features=encoder_output,
            text_features=text_features,
            vision_pos=pos_bc1s,
            spatial_shapes=self.spatial_shapes,
        )

        pred_logits = (
            self.scoring(
                decoder_hidden_states=final_hidden_states.unsqueeze(0),
                text_features=text_features,
            )
            .squeeze(-1)
            .squeeze(0)
        )

        mask_outputs = self.mask_decoder(
            decoder_queries=final_hidden_states,
            backbone_features=list(fpn_hidden_states_trimmed),
            encoder_hidden_states=encoder_output,
            prompt_features=text_features,
        )
        pred_masks = mask_outputs["pred_masks"]
        semantic_seg = mask_outputs["semantic_seg"]

        return pred_masks, pred_boxes, pred_logits, presence_logits, semantic_seg

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "facebook/sam3",
        image_size: int = 336,
    ) -> "Sam3Lite":
        import transformers

        hf_model = transformers.Sam3Model.from_pretrained(model_id)

        grid_size = image_size // 14
        lite_model = cls(image_size=image_size)

        lite_model.image_encoder = ImageEncoderBackbone.from_hf_backbone(
            hf_model.vision_encoder.backbone,
            image_size=image_size,
        )
        lite_model.fpn = FPNNeckReauthored.from_hf_fpn(
            hf_model.vision_encoder.neck,
            grid_h=grid_size,
            grid_w=grid_size,
        )
        lite_model.text_encoder = TextEncoderReauthored.from_hf_text_encoder(hf_model.text_encoder)
        lite_model.text_projection = _linear_to_conv2d(hf_model.text_projection)
        lite_model.detr_encoder = DETREncoderReauthored.from_hf_encoder(hf_model.detr_encoder)
        lite_model.detr_decoder = DETRDecoderReauthored.from_hf_decoder(
            hf_model.detr_decoder,
            spatial_h=grid_size,
            spatial_w=grid_size,
        )
        lite_model.mask_decoder = MaskDecoderReauthored.from_hf_mask_decoder(hf_model.mask_decoder)
        lite_model.scoring = DotProductScoringReauthored.from_hf_scoring(
            hf_model.dot_product_scoring
        )

        del hf_model
        return lite_model
