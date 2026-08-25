# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Wan 2.1 text-to-video model wrappers and dummy-input factories for export.
"""

from typing import Any

import torch

SPATIAL_COMPRESSION = 8
TEMPORAL_COMPRESSION = 4
DEFAULT_NUM_FRAMES = 81
DEFAULT_LATENT_FRAMES = (DEFAULT_NUM_FRAMES - 1) // TEMPORAL_COMPRESSION + 1  # 21
DEFAULT_TILE_SIZE = 32
DEFAULT_LATENT_HEIGHT = 60
DEFAULT_LATENT_WIDTH = 104
TEXT_SEQ_LEN = 226


class WanVAEDecoderWrapper(torch.nn.Module):
    def __init__(self, vae: torch.nn.Module) -> None:
        super().__init__()
        self.vae: Any = vae

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latent).sample


class WanTextEncoderWrapper(torch.nn.Module):
    def __init__(self, text_encoder: torch.nn.Module) -> None:
        super().__init__()
        self.model = text_encoder

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, attention_mask).last_hidden_state


class WanTransformerWrapper(torch.nn.Module):
    """Wrapper that lets the model compute RoPE internally from hidden_states shape."""

    def __init__(self, transformer: torch.nn.Module) -> None:
        super().__init__()
        self.model: Any = transformer
        self._patch_for_export()
        if hasattr(self.model, "fuse_qkv_projections"):
            self.model.fuse_qkv_projections()

    def _patch_for_export(self) -> None:
        """Replace FP32LayerNorm.forward to skip .float() upcast for clean bf16/fp16 export."""
        from diffusers.models.normalization import FP32LayerNorm

        for module in self.model.modules():
            if isinstance(module, FP32LayerNorm):
                module.forward = lambda x, _m=module: torch.nn.functional.layer_norm(
                    x,
                    _m.normalized_shape,
                    _m.weight.to(x.dtype) if _m.weight is not None else None,
                    _m.bias.to(x.dtype) if _m.bias is not None else None,
                    _m.eps,
                )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            return_dict=False,
        )[0]


# ---------------------------------------------------------------------------
# Dummy-input factories
# ---------------------------------------------------------------------------


def dummy_wan_vae_decoder(
    pipe: Any, batch_size: int = 2, *, tile_size: int | None = None
) -> tuple[torch.Tensor, ...]:
    """VAE at full resolution and full temporal extent.

    The Wan 3D VAE uses causal convolutions with internal frame-to-frame state
    propagation. Chunking temporally produces flash artifacts at boundaries.
    Must decode the full sequence in one pass (21 latent frames for 81 output frames).
    """
    dtype = next(pipe.vae.parameters()).dtype
    h = tile_size if tile_size is not None else DEFAULT_LATENT_HEIGHT
    w = tile_size if tile_size is not None else DEFAULT_LATENT_WIDTH
    return (torch.randn(1, 16, DEFAULT_LATENT_FRAMES, h, w, dtype=dtype),)


def dummy_wan_text_encoder(pipe: Any, batch_size: int = 2) -> tuple[torch.Tensor, ...]:
    return (
        torch.zeros(1, TEXT_SEQ_LEN, dtype=torch.long),
        torch.ones(1, TEXT_SEQ_LEN, dtype=torch.long),
    )


def dummy_wan_transformer(pipe: Any, batch_size: int = 2) -> tuple[torch.Tensor, ...]:
    """Transformer at default resolution (480x832) and frame count (81 frames = 21 latent)."""
    cfg = pipe.transformer.config
    # Some diffusers models keep a few submodules (e.g. norms, patch_embedding) in fp32
    # via `_skip_layerwise_casting_patterns`, so `next(parameters())` can return the wrong
    # dtype depending on iteration order. Use the patch_embedding's own dtype instead, since
    # that's the first layer the dummy hidden_states input actually feeds into.
    dtype = pipe.transformer.patch_embedding.weight.dtype

    latent_frames = DEFAULT_LATENT_FRAMES
    latent_h = DEFAULT_LATENT_HEIGHT
    latent_w = DEFAULT_LATENT_WIDTH

    return (
        torch.randn(1, cfg.in_channels, latent_frames, latent_h, latent_w, dtype=dtype),
        torch.randn(1, TEXT_SEQ_LEN, cfg.text_dim, dtype=dtype),
        torch.tensor([999.0], dtype=dtype),
    )


def wan_transformer_dynamic_shapes() -> tuple[dict[int, "torch.export.Dim"] | None, ...]:
    """Dynamic shape specs for Wan transformer — temporal dim is flexible."""
    temporal_dim = torch.export.Dim("latent_frames", min=2, max=21)
    return (
        {2: temporal_dim},  # hidden_states: [1, 16, T, 60, 104]
        None,  # encoder_hidden_states: [1, 226, 4096]
        None,  # timestep: [1]
    )
