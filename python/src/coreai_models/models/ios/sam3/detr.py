# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Re-authored DETR encoder + decoder for SAM3.

DETR encoder: 6 layers of self-attention with vision -> text cross-attention.
DETR decoder: 6 layers of self / text-cross / vision-cross attention with
relative position bias, plus box refinement and presence scoring.

All tensors in BC1S layout. ``nn.Linear`` becomes ``nn.Conv2d(1x1)`` and
all f32 literals are stored as f16 buffers.

HF reference: ``Sam3DetrEncoder``, ``Sam3DetrDecoder`` in
``transformers/models/sam3/modeling_sam3.py``.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from coreai_models.primitives.ios.bidirectional_sdpa import BidirectionalSDPA
from coreai_models.primitives.ios.layer_norm import LayerNormReauthored

_HIDDEN_SIZE = 256
_NUM_HEADS = 8
_HEAD_DIM = _HIDDEN_SIZE // _NUM_HEADS  # 32
_INTERMEDIATE_SIZE = 2048
_NUM_ENCODER_LAYERS = 6
_NUM_DECODER_LAYERS = 6
_NUM_QUERIES = 200
_LAYER_NORM_EPS = 1e-5


def _linear_to_conv2d(linear: nn.Linear) -> nn.Conv2d:
    in_f = linear.in_features
    out_f = linear.out_features
    has_bias = linear.bias is not None
    conv = nn.Conv2d(in_f, out_f, 1, bias=has_bias)
    conv.weight.data = linear.weight.data.reshape(out_f, in_f, 1, 1)
    if has_bias:
        conv.bias.data = linear.bias.data
    return conv


class DecoderMLPReauthored(nn.Module):
    """2- or 3-layer Conv2d(1x1) MLP with ReLU between hidden layers."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.layer1 = nn.Conv2d(input_dim, hidden_dim, 1, bias=True)
        self.layer2 = nn.Conv2d(
            hidden_dim, output_dim if num_layers == 2 else hidden_dim, 1, bias=True
        )
        self.layer3: nn.Conv2d | None = None
        if num_layers == 3:
            self.layer3 = nn.Conv2d(hidden_dim, output_dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        if self.layer3 is not None:
            x = F.relu(self.layer2(x))
            x = self.layer3(x)
        else:
            x = self.layer2(x)
        return x

    @classmethod
    def from_hf_mlp(cls, hf_mlp: nn.Module) -> "DecoderMLPReauthored":
        has_layer3 = hf_mlp.layer3 is not None
        num_layers = 3 if has_layer3 else 2
        input_dim = hf_mlp.layer1.in_features
        hidden_dim = hf_mlp.layer1.out_features
        output_dim = hf_mlp.layer3.out_features if has_layer3 else hf_mlp.layer2.out_features

        ane_mlp = cls(input_dim, hidden_dim, output_dim, num_layers)
        ane_mlp.layer1 = _linear_to_conv2d(hf_mlp.layer1)
        ane_mlp.layer2 = _linear_to_conv2d(hf_mlp.layer2)
        if has_layer3:
            ane_mlp.layer3 = _linear_to_conv2d(hf_mlp.layer3)
        return ane_mlp


class SinePositionEmbeddingReauthored(nn.Module):
    """Sine position embedding for DETR decoder box encoding."""

    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.register_buffer("_temperature", torch.tensor(float(temperature), dtype=torch.float16))
        self.register_buffer("_scale", torch.tensor(2.0 * math.pi, dtype=torch.float16))

    def encode_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Encode (cx, cy, w, h) boxes to (B, num_queries, num_pos_feats * 4)."""
        scale = self._scale.to(dtype=boxes.dtype)
        temperature = self._temperature.to(dtype=boxes.dtype)

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.int64, device=boxes.device).to(
            boxes.dtype
        )
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        x_embed = boxes[:, :, 0] * scale
        y_embed = boxes[:, :, 1] * scale
        w_embed = boxes[:, :, 2] * scale
        h_embed = boxes[:, :, 3] * scale

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_w = w_embed[:, :, None] / dim_t
        pos_h = h_embed[:, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        return torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)


class DETRAttentionReauthored(nn.Module):
    """Multi-head attention for DETR:
    separate Q/K inputs for cross-attn + position-injected Q/K."""

    def __init__(
        self,
        hidden_size: int = _HIDDEN_SIZE,
        num_heads: int = _NUM_HEADS,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Conv2d(hidden_size, hidden_size, 1, bias=True)
        self.k_proj = nn.Conv2d(hidden_size, hidden_size, 1, bias=True)
        self.v_proj = nn.Conv2d(hidden_size, hidden_size, 1, bias=True)
        self.o_proj = nn.Conv2d(hidden_size, hidden_size, 1, bias=True)

        self.sdpa = BidirectionalSDPA(num_heads=num_heads, head_dim=self.head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        out = self.sdpa(q, k, v, attention_mask=attention_mask)
        return self.o_proj(out)

    @classmethod
    def from_hf_attention(cls, hf_attn: nn.Module) -> "DETRAttentionReauthored":
        hidden_size = hf_attn.hidden_size
        num_heads = hf_attn.num_attention_heads
        ane_attn = cls(hidden_size=hidden_size, num_heads=num_heads)
        ane_attn.q_proj = _linear_to_conv2d(hf_attn.q_proj)
        ane_attn.k_proj = _linear_to_conv2d(hf_attn.k_proj)
        ane_attn.v_proj = _linear_to_conv2d(hf_attn.v_proj)
        ane_attn.o_proj = _linear_to_conv2d(hf_attn.o_proj)
        return ane_attn


class DETREncoderLayerReauthored(nn.Module):
    """Pre-norm DETR encoder layer with self-attn + text-cross-attn + MLP."""

    def __init__(
        self,
        hidden_size: int = _HIDDEN_SIZE,
        num_heads: int = _NUM_HEADS,
        intermediate_size: int = _INTERMEDIATE_SIZE,
        layer_norm_eps: float = _LAYER_NORM_EPS,
    ) -> None:
        super().__init__()
        self.layer_norm1 = LayerNormReauthored(hidden_size, eps=layer_norm_eps)
        self.self_attn = DETRAttentionReauthored(hidden_size, num_heads)

        self.layer_norm2 = LayerNormReauthored(hidden_size, eps=layer_norm_eps)
        self.cross_attn = DETRAttentionReauthored(hidden_size, num_heads)

        self.layer_norm3 = LayerNormReauthored(hidden_size, eps=layer_norm_eps)
        self.fc1 = nn.Conv2d(hidden_size, intermediate_size, 1, bias=True)
        self.fc2 = nn.Conv2d(intermediate_size, hidden_size, 1, bias=True)

    def forward(
        self,
        vision_feats: torch.Tensor,
        text_feats: torch.Tensor,
        vision_pos: torch.Tensor,
        text_cross_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = vision_feats
        h = self.layer_norm1(vision_feats)
        h_with_pos = h + vision_pos
        h = self.self_attn(query=h_with_pos, key=h_with_pos, value=h)
        vision_feats = residual + h

        residual = vision_feats
        h = self.layer_norm2(vision_feats)
        h = self.cross_attn(
            query=h,
            key=text_feats,
            value=text_feats,
            attention_mask=text_cross_attn_mask,
        )
        vision_feats = residual + h

        residual = vision_feats
        h = self.layer_norm3(vision_feats)
        h = F.relu(self.fc1(h))
        h = self.fc2(h)
        return residual + h

    @classmethod
    def from_hf_layer(cls, hf_layer: nn.Module) -> "DETREncoderLayerReauthored":
        hidden_size = hf_layer.layer_norm1.normalized_shape[0]
        num_heads = hf_layer.self_attn.num_attention_heads
        intermediate_size = hf_layer.mlp.fc1.out_features
        eps = hf_layer.layer_norm1.eps

        ane_layer = cls(
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            layer_norm_eps=eps,
        )

        ane_layer.layer_norm1 = LayerNormReauthored.from_torch_layer_norm(hf_layer.layer_norm1)
        ane_layer.self_attn = DETRAttentionReauthored.from_hf_attention(hf_layer.self_attn)

        ane_layer.layer_norm2 = LayerNormReauthored.from_torch_layer_norm(hf_layer.layer_norm2)
        ane_layer.cross_attn = DETRAttentionReauthored.from_hf_attention(hf_layer.cross_attn)

        ane_layer.layer_norm3 = LayerNormReauthored.from_torch_layer_norm(hf_layer.layer_norm3)
        ane_layer.fc1 = _linear_to_conv2d(hf_layer.mlp.fc1)
        ane_layer.fc2 = _linear_to_conv2d(hf_layer.mlp.fc2)
        return ane_layer


class DETREncoderReauthored(nn.Module):
    """6 DETR encoder layers — vision features with text cross-attention."""

    def __init__(
        self,
        hidden_size: int = _HIDDEN_SIZE,
        num_heads: int = _NUM_HEADS,
        intermediate_size: int = _INTERMEDIATE_SIZE,
        num_layers: int = _NUM_ENCODER_LAYERS,
        layer_norm_eps: float = _LAYER_NORM_EPS,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DETREncoderLayerReauthored(
                    hidden_size, num_heads, intermediate_size, layer_norm_eps
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        vision_feats: torch.Tensor,
        text_feats: torch.Tensor,
        vision_pos: torch.Tensor,
        text_cross_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = vision_feats
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                text_feats,
                vision_pos,
                text_cross_attn_mask=text_cross_attn_mask,
            )
        return hidden_states

    @classmethod
    def from_hf_encoder(cls, hf_encoder: nn.Module) -> "DETREncoderReauthored":
        config = hf_encoder.config
        ane_enc = cls(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            num_layers=config.num_layers,
            layer_norm_eps=1e-5,
        )
        for i, hf_layer in enumerate(hf_encoder.layers):
            ane_enc.layers[i] = DETREncoderLayerReauthored.from_hf_layer(hf_layer)
        return ane_enc


class DETRDecoderLayerReauthored(nn.Module):
    """Post-norm DETR decoder layer: self-attn + text-cross-attn + vision-cross-attn (RPB) + MLP."""

    def __init__(
        self,
        hidden_size: int = _HIDDEN_SIZE,
        num_heads: int = _NUM_HEADS,
        intermediate_size: int = _INTERMEDIATE_SIZE,
        layer_norm_eps: float = _LAYER_NORM_EPS,
    ) -> None:
        super().__init__()
        self.self_attn = DETRAttentionReauthored(hidden_size, num_heads)
        self.self_attn_layer_norm = LayerNormReauthored(hidden_size, eps=layer_norm_eps)

        self.text_cross_attn = DETRAttentionReauthored(hidden_size, num_heads)
        self.text_cross_attn_layer_norm = LayerNormReauthored(hidden_size, eps=layer_norm_eps)

        self.vision_cross_attn = DETRAttentionReauthored(hidden_size, num_heads)
        self.vision_cross_attn_layer_norm = LayerNormReauthored(hidden_size, eps=layer_norm_eps)

        self.fc1 = nn.Conv2d(hidden_size, intermediate_size, 1, bias=True)
        self.fc2 = nn.Conv2d(intermediate_size, hidden_size, 1, bias=True)
        self.mlp_layer_norm = LayerNormReauthored(hidden_size, eps=layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        query_pos: torch.Tensor,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
        vision_pos: torch.Tensor,
        text_cross_attn_mask: torch.Tensor | None = None,
        vision_cross_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        q_with_pos = hidden_states + query_pos
        h = self.self_attn(query=q_with_pos, key=q_with_pos, value=hidden_states)
        hidden_states = self.self_attn_layer_norm(residual + h)

        residual = hidden_states
        q_with_pos = hidden_states + query_pos
        h = self.text_cross_attn(
            query=q_with_pos,
            key=text_features,
            value=text_features,
            attention_mask=text_cross_attn_mask,
        )
        hidden_states = self.text_cross_attn_layer_norm(residual + h)

        residual = hidden_states
        q_with_pos = hidden_states + query_pos
        k_with_pos = vision_features + vision_pos
        h = self.vision_cross_attn(
            query=q_with_pos,
            key=k_with_pos,
            value=vision_features,
            attention_mask=vision_cross_attn_mask,
        )
        hidden_states = self.vision_cross_attn_layer_norm(residual + h)

        residual = hidden_states
        h = F.relu(self.fc1(hidden_states))
        h = self.fc2(h)
        return self.mlp_layer_norm(residual + h)

    @classmethod
    def from_hf_layer(cls, hf_layer: nn.Module) -> "DETRDecoderLayerReauthored":
        hidden_size = hf_layer.self_attn_layer_norm.normalized_shape[0]
        num_heads = hf_layer.self_attn.num_attention_heads
        intermediate_size = hf_layer.mlp.fc1.out_features
        eps = hf_layer.self_attn_layer_norm.eps

        ane_layer = cls(
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            layer_norm_eps=eps,
        )

        ane_layer.self_attn = DETRAttentionReauthored.from_hf_attention(hf_layer.self_attn)
        ane_layer.self_attn_layer_norm = LayerNormReauthored.from_torch_layer_norm(
            hf_layer.self_attn_layer_norm
        )

        ane_layer.text_cross_attn = DETRAttentionReauthored.from_hf_attention(
            hf_layer.text_cross_attn
        )
        ane_layer.text_cross_attn_layer_norm = LayerNormReauthored.from_torch_layer_norm(
            hf_layer.text_cross_attn_layer_norm
        )

        ane_layer.vision_cross_attn = DETRAttentionReauthored.from_hf_attention(
            hf_layer.vision_cross_attn
        )
        ane_layer.vision_cross_attn_layer_norm = LayerNormReauthored.from_torch_layer_norm(
            hf_layer.vision_cross_attn_layer_norm
        )

        ane_layer.fc1 = _linear_to_conv2d(hf_layer.mlp.fc1)
        ane_layer.fc2 = _linear_to_conv2d(hf_layer.mlp.fc2)
        ane_layer.mlp_layer_norm = LayerNormReauthored.from_torch_layer_norm(
            hf_layer.mlp_layer_norm
        )
        return ane_layer


def _box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """(cx, cy, w, h) -> (x1, y1, x2, y2). Tensor 0.5 keeps f32 literals out of the graph."""
    half = torch.tensor(0.5, dtype=x.dtype, device=x.device)
    x_c, y_c, w, h = x.unbind(-1)
    return torch.stack([x_c - w * half, y_c - h * half, x_c + w * half, y_c + h * half], dim=-1)


def _inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    eps = torch.tensor(1e-3, dtype=x.dtype, device=x.device)
    one = torch.tensor(1.0, dtype=x.dtype, device=x.device)
    zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
    x = x.clamp(min=zero, max=one)
    x1 = torch.max(x, eps)
    x2 = torch.max(one - x, eps)
    return torch.log(x1 / x2)


class DETRDecoderReauthored(nn.Module):
    """6 DETR decoder layers + box refinement + presence scoring.

    Returns ONLY the final layer's outputs — boxes already in xyxy — to
    keep the decoder region's number of escaping tensors minimal.
    """

    def __init__(
        self,
        hidden_size: int = _HIDDEN_SIZE,
        num_heads: int = _NUM_HEADS,
        intermediate_size: int = _INTERMEDIATE_SIZE,
        num_layers: int = _NUM_DECODER_LAYERS,
        num_queries: int = _NUM_QUERIES,
        layer_norm_eps: float = _LAYER_NORM_EPS,
        spatial_h: int = 72,
        spatial_w: int = 72,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w

        self.layers = nn.ModuleList(
            [
                DETRDecoderLayerReauthored(
                    hidden_size, num_heads, intermediate_size, layer_norm_eps
                )
                for _ in range(num_layers)
            ]
        )

        self.output_layer_norm = LayerNormReauthored(hidden_size, eps=layer_norm_eps)

        self.box_head = DecoderMLPReauthored(hidden_size, hidden_size, 4, num_layers=3)

        self.query_embed = nn.Embedding(num_queries, hidden_size)
        self.reference_points = nn.Embedding(num_queries, 4)

        self.presence_token = nn.Embedding(1, hidden_size)
        self.presence_head = DecoderMLPReauthored(hidden_size, hidden_size, 1, num_layers=3)
        self.presence_layer_norm = LayerNormReauthored(hidden_size, eps=layer_norm_eps)
        self.register_buffer("_clamp_val", torch.tensor(10.0, dtype=torch.float16))

        self.ref_point_head = DecoderMLPReauthored(
            2 * hidden_size, hidden_size, hidden_size, num_layers=2
        )

        self.box_rpb_embed_x = DecoderMLPReauthored(2, hidden_size, num_heads, num_layers=2)
        self.box_rpb_embed_y = DecoderMLPReauthored(2, hidden_size, num_heads, num_layers=2)

        self.position_encoding = SinePositionEmbeddingReauthored(num_pos_feats=hidden_size // 2)

        self.register_buffer(
            "_rpb_log_scale", torch.tensor(1.0 / math.log2(8), dtype=torch.float16)
        )
        self.register_buffer("_rpb_mult", torch.tensor(8.0, dtype=torch.float16))
        self.register_buffer("_rpb_one", torch.tensor(1.0, dtype=torch.float16))

    def _get_rpb_matrix(
        self,
        reference_boxes: torch.Tensor,
        spatial_h: int,
        spatial_w: int,
    ) -> torch.Tensor:
        """Box relative position bias in GPU SDPA layout (B, num_heads, Q, H*W)."""
        dtype = reference_boxes.dtype
        device = reference_boxes.device
        rpb_mult = self._rpb_mult.to(dtype=dtype)
        rpb_log_scale = self._rpb_log_scale.to(dtype=dtype)
        one = self._rpb_one.to(dtype=dtype)

        boxes_xyxy = _box_cxcywh_to_xyxy(reference_boxes)
        batch_size, num_queries, _ = boxes_xyxy.shape

        coords_h = torch.arange(0, spatial_h, device=device, dtype=dtype) / spatial_h
        coords_w = torch.arange(0, spatial_w, device=device, dtype=dtype) / spatial_w

        boxes_flat = boxes_xyxy.reshape(-1, 1, 4)

        deltas_y = coords_h.view(1, -1, 1) - boxes_flat[:, :, 1:4:2]
        deltas_y = deltas_y.view(batch_size, num_queries, -1, 2)

        deltas_x = coords_w.view(1, -1, 1) - boxes_flat[:, :, 0:3:2]
        deltas_x = deltas_x.view(batch_size, num_queries, -1, 2)

        deltas_x_log = deltas_x * rpb_mult
        deltas_x_log = (
            torch.sign(deltas_x_log) * torch.log2(torch.abs(deltas_x_log) + one) * rpb_log_scale
        )
        deltas_y_log = deltas_y * rpb_mult
        deltas_y_log = (
            torch.sign(deltas_y_log) * torch.log2(torch.abs(deltas_y_log) + one) * rpb_log_scale
        )

        dx_flat = deltas_x_log.reshape(batch_size * num_queries, -1, 2)
        dx_bc1s = dx_flat.permute(0, 2, 1).unsqueeze(2)
        dx_out = self.box_rpb_embed_x(dx_bc1s)

        dy_flat = deltas_y_log.reshape(batch_size * num_queries, -1, 2)
        dy_bc1s = dy_flat.permute(0, 2, 1).unsqueeze(2)
        dy_out = self.box_rpb_embed_y(dy_bc1s)

        dx = dx_out.squeeze(2).reshape(batch_size, num_queries, self.num_heads, -1)
        dx = dx.permute(0, 1, 3, 2)
        dy = dy_out.squeeze(2).reshape(batch_size, num_queries, self.num_heads, -1)
        dy = dy.permute(0, 1, 3, 2)

        rpb_matrix = dy.unsqueeze(3) + dx.unsqueeze(2)
        rpb_matrix = rpb_matrix.flatten(2, 3)
        rpb_matrix = rpb_matrix.permute(0, 3, 1, 2).contiguous()

        return rpb_matrix

    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        vision_pos: torch.Tensor,
        spatial_shapes: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = vision_features.shape[0]

        query_embeds = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        reference_boxes = (
            self.reference_points.weight.unsqueeze(0).expand(batch_size, -1, -1).sigmoid()
        )
        presence_token = self.presence_token.weight.unsqueeze(0).expand(batch_size, -1, -1)

        hidden_bsc = torch.cat([presence_token, query_embeds], dim=1)
        hidden_states = hidden_bsc.permute(0, 2, 1).unsqueeze(2)

        text_cross_attn_mask = None
        if text_mask is not None:
            inv_mask = (~text_mask).to(dtype=vision_features.dtype) * (-40000.0)
            text_cross_attn_mask = (
                inv_mask.unsqueeze(1)
                .unsqueeze(1)
                .expand(-1, self.num_heads, hidden_states.shape[3], -1)
            )

        normed_hs = None
        new_ref = None
        presence_logits = None

        for layer in self.layers:
            query_sine_embed = self.position_encoding.encode_boxes(reference_boxes)
            qse_bc1s = query_sine_embed.permute(0, 2, 1).unsqueeze(2)
            query_pos_bc1s = self.ref_point_head(qse_bc1s)

            # Prepend zeros for presence token (use cat, not pad — pad isn't supported on-device).
            presence_pad = torch.zeros(
                query_pos_bc1s.shape[0],
                query_pos_bc1s.shape[1],
                1,
                1,
                dtype=query_pos_bc1s.dtype,
                device=query_pos_bc1s.device,
            )
            query_pos_bc1s = torch.cat([presence_pad, query_pos_bc1s], dim=3)

            vision_cross_attn_mask = None
            if self.spatial_h > 0:
                rpb_matrix = self._get_rpb_matrix(reference_boxes, self.spatial_h, self.spatial_w)
                presence_rpb_pad = torch.zeros(
                    rpb_matrix.shape[0],
                    rpb_matrix.shape[1],
                    1,
                    rpb_matrix.shape[3],
                    dtype=rpb_matrix.dtype,
                    device=rpb_matrix.device,
                )
                vision_cross_attn_mask = torch.cat([presence_rpb_pad, rpb_matrix], dim=2)

            hidden_states = layer(
                hidden_states,
                query_pos=query_pos_bc1s,
                text_features=text_features,
                vision_features=vision_features,
                vision_pos=vision_pos,
                text_cross_attn_mask=text_cross_attn_mask,
                vision_cross_attn_mask=vision_cross_attn_mask,
            )

            query_hs = hidden_states[:, :, :, 1:]

            normed_hs = self.output_layer_norm(query_hs)
            ref_before_sigmoid = _inverse_sigmoid(reference_boxes)
            delta_boxes_bc1s = self.box_head(normed_hs)
            delta_boxes = delta_boxes_bc1s.squeeze(2).permute(0, 2, 1)
            new_ref = (delta_boxes + ref_before_sigmoid).sigmoid()
            reference_boxes = new_ref.detach()

            presence_hs = hidden_states[:, :, :, :1]
            presence_normed = self.presence_layer_norm(presence_hs)
            presence_logits_bc1s = self.presence_head(presence_normed)
            presence_logits = presence_logits_bc1s.squeeze(2).squeeze(1)
            clamp_val = self._clamp_val.to(dtype=presence_logits.dtype)
            presence_logits = presence_logits.clamp(min=-clamp_val, max=clamp_val)

        final_boxes_xyxy = _box_cxcywh_to_xyxy(new_ref)
        return normed_hs, final_boxes_xyxy, presence_logits

    @classmethod
    def from_hf_decoder(
        cls,
        hf_decoder: nn.Module,
        spatial_h: int = 72,
        spatial_w: int = 72,
    ) -> "DETRDecoderReauthored":
        config = hf_decoder.config
        ane_dec = cls(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            num_layers=config.num_layers,
            num_queries=config.num_queries,
            layer_norm_eps=1e-5,
            spatial_h=spatial_h,
            spatial_w=spatial_w,
        )

        for i, hf_layer in enumerate(hf_decoder.layers):
            ane_dec.layers[i] = DETRDecoderLayerReauthored.from_hf_layer(hf_layer)

        ane_dec.output_layer_norm = LayerNormReauthored.from_torch_layer_norm(
            hf_decoder.output_layer_norm
        )

        ane_dec.box_head = DecoderMLPReauthored.from_hf_mlp(hf_decoder.box_head)
        ane_dec.presence_head = DecoderMLPReauthored.from_hf_mlp(hf_decoder.presence_head)
        ane_dec.presence_layer_norm = LayerNormReauthored.from_torch_layer_norm(
            hf_decoder.presence_layer_norm
        )
        ane_dec.ref_point_head = DecoderMLPReauthored.from_hf_mlp(hf_decoder.ref_point_head)

        ane_dec.box_rpb_embed_x = DecoderMLPReauthored.from_hf_mlp(hf_decoder.box_rpb_embed_x)
        ane_dec.box_rpb_embed_y = DecoderMLPReauthored.from_hf_mlp(hf_decoder.box_rpb_embed_y)

        ane_dec.query_embed.weight.data = hf_decoder.query_embed.weight.data.clone()
        ane_dec.reference_points.weight.data = hf_decoder.reference_points.weight.data.clone()
        ane_dec.presence_token.weight.data = hf_decoder.presence_token.weight.data.clone()
        return ane_dec
