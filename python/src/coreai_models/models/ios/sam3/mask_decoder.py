# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Re-authored SAM3 mask decoder + dot-product scoring.

Mask decoder: pixel decoder (FPN-style), prompt cross-attention, mask
embedder MLP, instance / semantic projections, einsum-rewritten as
matmul.

Dot-product scoring: text MLP, mean pool, project, dot product with
scale + clamp.

HF reference: ``Sam3MaskDecoder``, ``Sam3DotProductScoring`` in
``transformers/models/sam3/modeling_sam3.py``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from coreai_models.primitives.ios.bidirectional_sdpa import BidirectionalSDPA
from coreai_models.primitives.ios.layer_norm import LayerNormReauthored


def _linear_to_conv2d(linear: nn.Linear) -> nn.Conv2d:
    in_f = linear.in_features
    out_f = linear.out_features
    has_bias = linear.bias is not None
    conv = nn.Conv2d(in_f, out_f, 1, bias=has_bias)
    conv.weight.data = linear.weight.data.reshape(out_f, in_f, 1, 1)
    if has_bias:
        conv.bias.data = linear.bias.data
    return conv


class GroupNormReauthored(nn.Module):
    """Manual GroupNorm in pure rank-4 ops.

    ``nn.GroupNorm`` is not supported on some accelerators (e.g. h16c);
    this implements the same math with reshape + mean + variance + scale
    + shift, all rank 4.
    """

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        assert num_channels % num_groups == 0
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.channels_per_group = num_channels // num_groups
        self._eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        G = self.num_groups
        CG = self.channels_per_group
        eps = torch.tensor(self._eps, dtype=x.dtype, device=x.device)

        x_grouped = x.reshape(B * G, CG, H, W)

        mean = x_grouped.mean(dim=(1, 2, 3), keepdim=True)
        x_centered = x_grouped - mean
        var = (x_centered * x_centered).mean(dim=(1, 2, 3), keepdim=True)
        inv_std = torch.rsqrt(var + eps)

        x_norm = x_centered * inv_std
        x_norm = x_norm.reshape(B, C, H, W)
        return x_norm * self.weight + self.bias

    @classmethod
    def from_torch_group_norm(cls, gn: nn.GroupNorm) -> "GroupNormReauthored":
        ane_gn = cls(
            num_groups=gn.num_groups,
            num_channels=gn.num_channels,
            eps=gn.eps,
        )
        if gn.weight is not None:
            ane_gn.weight.data = gn.weight.data.reshape(1, gn.num_channels, 1, 1)
        if gn.bias is not None:
            ane_gn.bias.data = gn.bias.data.reshape(1, gn.num_channels, 1, 1)
        return ane_gn


_HIDDEN_SIZE = 256
_NUM_HEADS = 8
_HEAD_DIM = _HIDDEN_SIZE // _NUM_HEADS  # 32
_NUM_UPSAMPLING_STAGES = 3
_INTERMEDIATE_SIZE = 2048


class PixelDecoderReauthored(nn.Module):
    """FPN-style pixel decoder operating on spatial ``(B, C, H, W)`` tensors.

    Uses ``GroupNormReauthored`` (rank-4 manual implementation) and
    ``repeat_interleave`` for 2x nearest upsample (instead of
    ``F.interpolate``, which produces f32 scale/offset constants the
    accelerator rejects).
    """

    def __init__(
        self,
        hidden_size: int = _HIDDEN_SIZE,
        num_stages: int = _NUM_UPSAMPLING_STAGES,
    ) -> None:
        super().__init__()
        self.conv_layers = nn.ModuleList(
            [nn.Conv2d(hidden_size, hidden_size, 3, padding=1) for _ in range(num_stages)]
        )
        self.norms = nn.ModuleList([GroupNormReauthored(8, hidden_size) for _ in range(num_stages)])

    @staticmethod
    def _nearest_upsample_2x(x: torch.Tensor) -> torch.Tensor:
        return x.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)

    def forward(self, backbone_features: list[torch.Tensor]) -> torch.Tensor:
        prev_fpn = backbone_features[-1]
        for layer_idx, backbone_feat in enumerate(reversed(backbone_features[:-1])):
            prev_fpn = self._nearest_upsample_2x(prev_fpn)
            prev_fpn = prev_fpn + backbone_feat
            prev_fpn = self.conv_layers[layer_idx](prev_fpn)
            prev_fpn = self.norms[layer_idx](prev_fpn)
            prev_fpn = F.relu(prev_fpn)
        return prev_fpn

    @classmethod
    def from_hf_pixel_decoder(cls, hf_pd: nn.Module) -> "PixelDecoderReauthored":
        num_stages = len(hf_pd.conv_layers)
        hidden_size = hf_pd.conv_layers[0].out_channels
        ane_pd = cls(hidden_size=hidden_size, num_stages=num_stages)
        for i in range(num_stages):
            ane_pd.conv_layers[i].load_state_dict(hf_pd.conv_layers[i].state_dict())
            ane_pd.norms[i] = GroupNormReauthored.from_torch_group_norm(hf_pd.norms[i])
        return ane_pd


class MaskEmbedderReauthored(nn.Module):
    """3-layer Conv2d(1x1) MLP with ReLU between the first two layers."""

    def __init__(self, hidden_size: int = _HIDDEN_SIZE) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(hidden_size, hidden_size, 1, bias=True),
                nn.Conv2d(hidden_size, hidden_size, 1, bias=True),
                nn.Conv2d(hidden_size, hidden_size, 1, bias=True),
            ]
        )

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        h = queries
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = F.relu(h)
        return h

    @classmethod
    def from_hf_mask_embedder(cls, hf_me: nn.Module) -> "MaskEmbedderReauthored":
        hidden_size = hf_me.layers[0].in_features
        ane_me = cls(hidden_size=hidden_size)
        for i in range(3):
            ane_me.layers[i] = _linear_to_conv2d(hf_me.layers[i])
        return ane_me


class CrossAttentionReauthored(nn.Module):
    """Encoder hidden states attend to prompt features (BC1S)."""

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

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(query)
        k = self.k_proj(key_value)
        v = self.v_proj(key_value)
        return self.o_proj(self.sdpa(q, k, v))

    @classmethod
    def from_hf_attention(cls, hf_attn: nn.Module) -> "CrossAttentionReauthored":
        hidden_size = hf_attn.hidden_size
        num_heads = hf_attn.num_attention_heads
        ane_attn = cls(hidden_size=hidden_size, num_heads=num_heads)
        ane_attn.q_proj = _linear_to_conv2d(hf_attn.q_proj)
        ane_attn.k_proj = _linear_to_conv2d(hf_attn.k_proj)
        ane_attn.v_proj = _linear_to_conv2d(hf_attn.v_proj)
        ane_attn.o_proj = _linear_to_conv2d(hf_attn.o_proj)
        return ane_attn


class MaskDecoderReauthored(nn.Module):
    """Mask decoder: prompt cross-attention + pixel decoder + einsum-as-matmul."""

    def __init__(
        self,
        hidden_size: int = _HIDDEN_SIZE,
        num_heads: int = _NUM_HEADS,
        num_upsampling_stages: int = _NUM_UPSAMPLING_STAGES,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        self.pixel_decoder = PixelDecoderReauthored(hidden_size, num_upsampling_stages)
        self.mask_embedder = MaskEmbedderReauthored(hidden_size)

        self.instance_projection = nn.Conv2d(hidden_size, hidden_size, 1)
        self.semantic_projection = nn.Conv2d(hidden_size, 1, 1)

        self.prompt_cross_attn = CrossAttentionReauthored(hidden_size, num_heads)
        self.prompt_cross_attn_norm = LayerNormReauthored(hidden_size)

    def forward(
        self,
        decoder_queries: torch.Tensor,
        backbone_features: list[torch.Tensor],
        encoder_hidden_states: torch.Tensor,
        prompt_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if prompt_features is not None:
            residual = encoder_hidden_states
            normed = self.prompt_cross_attn_norm(encoder_hidden_states)
            attn_out = self.prompt_cross_attn(normed, prompt_features)
            encoder_hidden_states = residual + attn_out

        pixel_embed = self._embed_pixels(backbone_features, encoder_hidden_states)
        instance_embeds = self.instance_projection(pixel_embed)
        mask_embeddings = self.mask_embedder(decoder_queries)
        pred_masks = self._einsum_as_matmul(mask_embeddings, instance_embeds)
        semantic_seg = self.semantic_projection(pixel_embed)

        return {"pred_masks": pred_masks, "semantic_seg": semantic_seg}

    def _embed_pixels(
        self,
        backbone_features: list[torch.Tensor],
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        backbone_visual_feats = [feat.clone() for feat in backbone_features]

        H, W = backbone_features[-1].shape[-2:]
        spatial_dim = H * W

        B, C = encoder_hidden_states.shape[:2]
        encoder_visual_embed = encoder_hidden_states[:, :, :, :spatial_dim]
        encoder_visual_embed = encoder_visual_embed.reshape(B, C, H, W)

        backbone_visual_feats[-1] = encoder_visual_embed
        return self.pixel_decoder(backbone_visual_feats)

    @staticmethod
    def _einsum_as_matmul(
        mask_embeddings: torch.Tensor,
        instance_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Rewrite ``einsum('bqc,bchw->bqhw')`` as a single matmul + reshape."""
        B, C, H, W = instance_embeds.shape
        Q = mask_embeddings.shape[3]

        me = mask_embeddings.squeeze(2).permute(0, 2, 1)  # (B, Q, C)
        ie = instance_embeds.reshape(B, C, H * W)
        out = torch.matmul(me, ie)
        return out.reshape(B, Q, H, W)

    @classmethod
    def from_hf_mask_decoder(cls, hf_md: nn.Module) -> "MaskDecoderReauthored":
        hidden_size = hf_md.config.hidden_size
        num_heads = hf_md.config.num_attention_heads
        num_stages = hf_md.config.num_upsampling_stages

        ane_md = cls(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_upsampling_stages=num_stages,
        )

        ane_md.pixel_decoder = PixelDecoderReauthored.from_hf_pixel_decoder(hf_md.pixel_decoder)
        ane_md.mask_embedder = MaskEmbedderReauthored.from_hf_mask_embedder(hf_md.mask_embedder)

        ane_md.instance_projection.load_state_dict(hf_md.instance_projection.state_dict())
        ane_md.semantic_projection.load_state_dict(hf_md.semantic_projection.state_dict())

        ane_md.prompt_cross_attn = CrossAttentionReauthored.from_hf_attention(
            hf_md.prompt_cross_attn
        )
        ane_md.prompt_cross_attn_norm = LayerNormReauthored.from_torch_layer_norm(
            hf_md.prompt_cross_attn_norm
        )
        return ane_md


class DotProductScoringReauthored(nn.Module):
    """Text MLP + mean pool + projection + dot product with scale and clamp."""

    def __init__(
        self,
        hidden_size: int = _HIDDEN_SIZE,
        intermediate_size: int = _INTERMEDIATE_SIZE,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        self.text_mlp_layer1 = nn.Conv2d(hidden_size, intermediate_size, 1, bias=True)
        self.text_mlp_layer2 = nn.Conv2d(intermediate_size, hidden_size, 1, bias=True)
        self.text_mlp_out_norm = LayerNormReauthored(hidden_size)

        self.text_proj = nn.Conv2d(hidden_size, hidden_size, 1, bias=True)
        self.query_proj = nn.Conv2d(hidden_size, hidden_size, 1, bias=True)

        self.register_buffer("_scale", torch.tensor(1.0 / (hidden_size**0.5), dtype=torch.float16))
        self.register_buffer("_clamp_max", torch.tensor(12.0, dtype=torch.float16))

    def forward(
        self,
        decoder_hidden_states: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        orig_text = text_features
        h = self.text_mlp_layer1(text_features)
        h = F.relu(h)
        h = self.text_mlp_layer2(h)
        text_features = self.text_mlp_out_norm(h + orig_text)

        pooled_text = text_features.mean(dim=3, keepdim=True)
        proj_text = self.text_proj(pooled_text)

        scale = self._scale.to(dtype=decoder_hidden_states.dtype)
        clamp_max = self._clamp_max.to(dtype=decoder_hidden_states.dtype)

        num_layers = decoder_hidden_states.shape[0]
        results = []
        for layer_idx in range(num_layers):
            queries = decoder_hidden_states[layer_idx]
            proj_q = self.query_proj(queries)

            pq = proj_q.squeeze(2).permute(0, 2, 1)
            pt = proj_text.squeeze(2).squeeze(2).unsqueeze(2)

            scores = torch.matmul(pq, pt)
            scores = scores * scale
            scores = scores.clamp(min=-clamp_max, max=clamp_max)
            results.append(scores)

        return torch.stack(results, dim=0)

    @classmethod
    def from_hf_scoring(cls, hf_sc: nn.Module) -> "DotProductScoringReauthored":
        hidden_size = hf_sc.text_proj.in_features
        intermediate_size = hf_sc.text_mlp.layer1.out_features

        ane_sc = cls(hidden_size=hidden_size, intermediate_size=intermediate_size)

        ane_sc.text_mlp_layer1 = _linear_to_conv2d(hf_sc.text_mlp.layer1)
        ane_sc.text_mlp_layer2 = _linear_to_conv2d(hf_sc.text_mlp.layer2)

        ane_sc.text_mlp_out_norm = LayerNormReauthored.from_torch_layer_norm(
            hf_sc.text_mlp_out_norm
        )

        ane_sc.text_proj = _linear_to_conv2d(hf_sc.text_proj)
        ane_sc.query_proj = _linear_to_conv2d(hf_sc.query_proj)

        ane_sc._scale.fill_(hf_sc.scale)
        ane_sc._clamp_max.fill_(hf_sc.clamp_max_val)
        return ane_sc
