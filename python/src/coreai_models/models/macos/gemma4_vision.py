# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Gemma 4 vision encoder, reauthored for Core AI export.

Gemma 4 (as shipped in ``transformers==5.10.2``) uses a native ViT-style vision
tower (:class:`Gemma4VisionModel`: patch embedder + transformer encoder + 2-D
average pooler + optional standardization) followed by a soft-token multimodal
embedder (:class:`Gemma4MultimodalEmbedder`). This differs from both Qwen3-VL's
ViT+patchify tower and the earlier Gemma 3n MobileNet-v5 tower.

HF flow (``Gemma4Model.get_image_features`` -> ``vision_tower`` ->
``embed_vision``), for a single full 768x768 tile:

  1. Image is rescaled to ``[0, 1]``, resized to 768x768, and patchified into
     ``[num_patches, 3*patch_size**2]`` (num_patches = (768/16)**2 = 2304).
  2. ``pixel_position_ids`` are the ``(x, y)`` grid coords ``[num_patches, 2]``.
  3. Patch embedder: ``2*(x-0.5)`` -> ``input_proj`` (Linear) + 2-D position
     embeddings.
  4. Transformer encoder over the patches.
  5. Pooler: 2-D average-pool by a ``k x k`` grid (k = pooling_kernel_size = 3)
     down to ``num_soft_tokens = num_patches // k**2 = 256`` tokens, scaled by
     ``sqrt(hidden_size)``; optional standardization.
  6. ``embed_vision`` soft path: ``embedding_pre_projection_norm`` ->
     ``embedding_projection`` (Linear to text hidden). -> ``[1, 256, text_hidden]``.

For a single, un-padded tile all patches are valid, so the pooler's
data-dependent boolean masking collapses to a plain reshape and the whole thing
exports with fully static shapes -- the same trick Qwen3-VL's
``StaticVisionEncoder`` uses. This module bakes in the fixed grid / position ids
at init and accepts plain NCHW pixels (what the Swift ``ImagePreprocessor``
produces), patchifying internally so the runner needs only resize + rescale.

Input:  ``pixel_values``  float32 ``[1, 3, image_size, image_size]`` (rescaled to [0,1], NCHW)
Output: ``image_features`` float16 ``[1, num_soft_tokens, text_hidden]``
"""

import torch
import torch.nn as nn


class Gemma4StaticVisionEncoder(nn.Module):
    """Static single-tile Gemma 4 vision encoder (patchify + tower + soft-token embed)."""

    def __init__(
        self,
        vision_tower: nn.Module,
        embed_vision: nn.Module,
        *,
        image_size: int,
        patch_size: int,
        pooling_kernel_size: int,
    ) -> None:
        super().__init__()
        self.patch_embedder = vision_tower.patch_embedder
        self.encoder = vision_tower.encoder
        self.pooler = vision_tower.pooler
        self.standardize = bool(getattr(vision_tower.config, "standardize", False))
        if self.standardize:
            self.register_buffer("std_bias", vision_tower.std_bias.detach().clone())
            self.register_buffer("std_scale", vision_tower.std_scale.detach().clone())
        self.embed_vision = embed_vision

        self.image_size = image_size
        self.patch_size = patch_size
        self.channels = 3
        self.grid = image_size // patch_size  # patches per side
        self.num_patches = self.grid * self.grid
        self.output_length = self.num_patches // (pooling_kernel_size**2)
        self._k = pooling_kernel_size

        # Fixed (x, y) patch position ids for a full grid: meshgrid(arange(W), arange(H)).
        # Matches image_processing_gemma4: stacked_grid = stack(meshgrid(x, y)).reshape(N, 2).
        with torch.no_grad():
            gx, gy = torch.meshgrid(torch.arange(self.grid), torch.arange(self.grid), indexing="ij")
            pos = torch.stack([gx, gy], dim=-1).reshape(self.num_patches, 2)
            self.register_buffer("pixel_position_ids", pos.unsqueeze(0).long())  # [1, N, 2]
            # No padding for a single full tile.
            self.register_buffer(
                "padding_positions", torch.zeros(1, self.num_patches, dtype=torch.bool)
            )
            # Precompute the pooling weight matrix [N, output_length] so the
            # data-dependent one_hot / boolean-mask path in the HF pooler is
            # replaced by a static matmul. Mirrors _avg_pool_by_positions with a
            # full grid (all patches valid).
            k = pooling_kernel_size
            kernel = torch.div(pos, k, rounding_mode="floor")
            max_x = int(pos[:, 0].max()) + 1
            kernel_idx = kernel[:, 0] + (max_x // k) * kernel[:, 1]
            weights = torch.zeros(self.num_patches, self.output_length)
            weights[torch.arange(self.num_patches), kernel_idx.long()] = 1.0 / (k * k)
            self.register_buffer("pool_weights", weights)  # [N, output_length]

    def _patchify(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """NCHW pixels -> [1, num_patches, 3*patch_size**2] (Gemma4/SigLIP2 layout)."""
        c, p, g = self.channels, self.patch_size, self.grid
        x = pixel_values.reshape(c, g, p, g, p)
        x = x.permute(1, 3, 2, 4, 0)  # (gh, gw, ph, pw, c)
        return x.reshape(1, self.num_patches, p * p * c)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        patches = self._patchify(pixel_values)  # [1, N, 3*p*p]
        hidden = self.patch_embedder(patches, self.pixel_position_ids, self.padding_positions)
        enc = self.encoder(
            inputs_embeds=hidden,
            attention_mask=~self.padding_positions,
            pixel_position_ids=self.pixel_position_ids,
        )
        hs = enc.last_hidden_state  # [1, N, vision_hidden]

        # Static 2-D average pool -> [1, output_length, vision_hidden], scaled by sqrt(hidden).
        pooled = self.pool_weights.transpose(0, 1).to(hs.dtype) @ hs
        pooled = pooled.float() * self.pooler.root_hidden_size
        if self.standardize:
            pooled = (pooled - self.std_bias.float()) * self.std_scale.float()
        pooled = pooled.to(hs.dtype)

        # Soft-token embedder -> [1, output_length, text_hidden].
        return self.embed_vision(inputs_embeds=pooled)


class BatchedF16VisionEncoder(nn.Module):
    """Cast the encoder output to f16 with a leading batch dim (embed/main contract)."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.encoder(pixel_values)
        if isinstance(out, tuple):
            out = out[0]
        if out.dim() == 2:
            out = out.unsqueeze(0)
        return out.to(torch.float16)


def build_vision_encoder(
    hf_model: nn.Module,
    *,
    image_size: int,
) -> nn.Module:
    """Build the exportable Gemma 4 vision encoder from a loaded HF model.

    ``hf_model`` is a ``Gemma4ForConditionalGeneration``; the vision submodules
    live under ``hf_model.model.vision_tower`` and ``hf_model.model.embed_vision``.
    """
    inner = getattr(hf_model, "model", hf_model)
    vision_tower = inner.vision_tower
    embed_vision = inner.embed_vision
    vcfg = vision_tower.config
    encoder = Gemma4StaticVisionEncoder(
        vision_tower,
        embed_vision,
        image_size=image_size,
        patch_size=vcfg.patch_size,
        pooling_kernel_size=vcfg.pooling_kernel_size,
    )
    return BatchedF16VisionEncoder(encoder).eval()
