# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
LTX Video component specifications and torch wrappers for Core AI export.

LTX Video is a video diffusion transformer from Lightricks that uses:
- T5 text encoder (frozen, last hidden state)
- N-block transformer with 3D RoPE (temporal + spatial)
- AutoencoderKLLTXVideo 3D VAE for encode/decode of video latents

Key design: like Flux2, the transformer uses pre-computed 3D RoPE embeddings
passed as model inputs rather than computed in-graph, to avoid Core AI graph
optimizer issues with RoPE frequency ops in deep transformers.
"""

from __future__ import annotations

import math
from typing import Any, cast

import torch


# ---------------------------------------------------------------------------
# 3D RoPE pre-computation (outside the exported graph)
# ---------------------------------------------------------------------------


def compute_ltx_video_rope(
    num_frames: int,
    height: int,
    width: int,
    dim: int,
    patch_size: int = 1,
    patch_size_t: int = 1,
    base_num_frames: int = 20,
    base_height: int = 2048,
    base_width: int = 2048,
    theta: float = 10000.0,
    rope_interpolation_scale: tuple[float, float, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute 3D RoPE (cos, sin) embeddings for LTX Video transformer.

    Replicates LTXVideoRotaryPosEmbed.forward() logic:
      - Build a 3D grid of (frame, height, width) coordinates
      - Scale each axis by interpolation scale and base resolution
      - Compute frequency bands and apply to grid
      - Return (cos, sin) each of shape [1, seq_len, dim]

    Args:
        num_frames: Number of temporal frames (after patchification).
        height: Spatial height (after patchification).
        width: Spatial width (after patchification).
        dim: Inner dimension of the transformer (num_heads * head_dim).
        patch_size: Spatial patch size.
        patch_size_t: Temporal patch size.
        base_num_frames: Base number of frames for interpolation scaling.
        base_height: Base height for interpolation scaling.
        base_width: Base width for interpolation scaling.
        theta: RoPE theta parameter.
        rope_interpolation_scale: Optional (t_scale, h_scale, w_scale) tuple.

    Returns:
        (rotary_emb_cos, rotary_emb_sin) each of shape [1, seq_len, dim].
    """
    # Build 3D coordinate grid
    grid_f = torch.arange(num_frames, dtype=torch.float32)
    grid_h = torch.arange(height, dtype=torch.float32)
    grid_w = torch.arange(width, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(grid_f, grid_h, grid_w, indexing="ij"), dim=0)
    grid = grid.unsqueeze(0)  # [1, 3, F, H, W]

    # Apply interpolation scaling
    if rope_interpolation_scale is not None:
        t_scale, h_scale, w_scale = rope_interpolation_scale
    else:
        t_scale, h_scale, w_scale = 1.0, 1.0, 1.0

    grid[:, 0:1] = grid[:, 0:1] * t_scale * patch_size_t / base_num_frames
    grid[:, 1:2] = grid[:, 1:2] * h_scale * patch_size / base_height
    grid[:, 2:3] = grid[:, 2:3] * w_scale * patch_size / base_width

    # Flatten spatial dims: [1, 3, F*H*W] -> transpose -> [1, F*H*W, 3]
    grid = grid.flatten(2, 4).transpose(1, 2)

    # Compute frequency bands
    start = 1.0
    end = theta
    freqs = theta ** torch.linspace(
        math.log(start, theta),
        math.log(end, theta),
        dim // 6,
        dtype=torch.float32,
    )
    freqs = freqs * math.pi / 2.0

    # Apply frequencies to grid coordinates
    # grid: [1, S, 3], freqs: [dim//6]
    # -> [1, S, 3, dim//6] -> transpose -> [1, S, dim//6, 3] -> flatten -> [1, S, dim//2]
    freqs = freqs * (grid.unsqueeze(-1) * 2 - 1)
    freqs = freqs.transpose(-1, -2).flatten(2)

    cos_freqs = freqs.cos().repeat_interleave(2, dim=-1)
    sin_freqs = freqs.sin().repeat_interleave(2, dim=-1)

    # Handle non-divisible dimensions with padding
    if dim % 6 != 0:
        cos_padding = torch.ones_like(cos_freqs[:, :, : dim % 6])
        sin_padding = torch.zeros_like(cos_freqs[:, :, : dim % 6])
        cos_freqs = torch.cat([cos_padding, cos_freqs], dim=-1)
        sin_freqs = torch.cat([sin_padding, sin_freqs], dim=-1)

    return cos_freqs, sin_freqs


# ---------------------------------------------------------------------------
# Torch wrappers
# ---------------------------------------------------------------------------


class LTXVideoTransformerWrapper(torch.nn.Module):
    """Wraps LTXVideoTransformer3DModel for export with pre-computed 3D RoPE.

    Instead of computing RoPE internally via self.rope(), this wrapper accepts
    (rotary_emb_cos, rotary_emb_sin) directly, removing all RoPE frequency
    computation from the traced graph.
    """

    def __init__(self, transformer: torch.nn.Module) -> None:
        super().__init__()
        self.model: Any = transformer

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        rotary_emb_cos: torch.Tensor,
        rotary_emb_sin: torch.Tensor,
    ) -> torch.Tensor:
        model = self.model

        image_rotary_emb = (rotary_emb_cos, rotary_emb_sin)

        # Convert encoder_attention_mask to a bias
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(hidden_states.dtype)
            ) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        batch_size = hidden_states.size(0)
        hidden_states = model.proj_in(hidden_states)

        temb, embedded_timestep = model.time_embed(
            timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )

        temb = temb.view(batch_size, -1, temb.size(-1))
        embedded_timestep = embedded_timestep.view(
            batch_size, -1, embedded_timestep.size(-1)
        )

        encoder_hidden_states = model.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(
            batch_size, -1, hidden_states.size(-1)
        )

        for block in model.transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                encoder_attention_mask=encoder_attention_mask,
            )

        scale_shift_values = (
            model.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        hidden_states = model.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        return model.proj_out(hidden_states)


class LTXVideoTextEncoderWrapper(torch.nn.Module):
    """Wraps T5EncoderModel for LTX Video: (input_ids, attention_mask) -> hidden_states."""

    def __init__(self, text_encoder: torch.nn.Module) -> None:
        super().__init__()
        self.model = text_encoder

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self.model(input_ids=input_ids, attention_mask=attention_mask)[0],
        )


class LTXVideoVAEDecoderWrapper(torch.nn.Module):
    """Wraps AutoencoderKLLTXVideo.decode: (latent) -> (video frames).

    The 3D VAE decodes latents of shape [B, C, T, H, W] to pixel space.
    """

    def __init__(self, vae: torch.nn.Module) -> None:
        super().__init__()
        self.vae: Any = vae

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.vae.decode(z).sample)


class LTXVideoVAEEncoderWrapper(torch.nn.Module):
    """Wraps AutoencoderKLLTXVideo.encode: (video frames) -> (latent).

    For image-to-video conditioning, encodes pixel-space video to latent space.
    """

    def __init__(self, vae: torch.nn.Module) -> None:
        super().__init__()
        self.vae: Any = vae

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.vae.encode(x).latent_dist.mode())


# ---------------------------------------------------------------------------
# Dummy-input factories
# ---------------------------------------------------------------------------


def _dummy_ltx_video_transformer_impl(
    pipe: Any,
    num_frames: int = 9,
    height: int = 64,
    width: int = 64,
) -> tuple[torch.Tensor, ...]:
    """Build dummy inputs for the LTX Video transformer at a given resolution.

    Args:
        pipe: The loaded HuggingFace LTXPipeline.
        num_frames: Number of latent frames (after temporal compression).
        height: Latent spatial height (after spatial compression).
        width: Latent spatial width (after spatial compression).
    """
    cfg = pipe.transformer.config
    dtype = next(pipe.transformer.parameters()).dtype
    num_heads = cfg.num_attention_heads
    head_dim = cfg.attention_head_dim
    inner_dim = num_heads * head_dim
    in_channels = cfg.in_channels
    caption_channels = cfg.caption_channels
    text_seq_len = 128

    video_seq_len = num_frames * height * width

    # Pre-compute 3D RoPE
    rope_cos, rope_sin = compute_ltx_video_rope(
        num_frames=num_frames,
        height=height,
        width=width,
        dim=inner_dim,
        patch_size=getattr(cfg, "patch_size", 1),
        patch_size_t=getattr(cfg, "patch_size_t", 1),
    )

    return (
        torch.randn(1, video_seq_len, in_channels, dtype=dtype),  # hidden_states
        torch.randn(1, text_seq_len, caption_channels, dtype=dtype),  # encoder_hidden_states
        torch.tensor([0.5], dtype=dtype),  # timestep
        torch.ones(1, text_seq_len, dtype=dtype),  # encoder_attention_mask
        rope_cos.to(dtype),  # rotary_emb_cos
        rope_sin.to(dtype),  # rotary_emb_sin
    )


def dummy_ltx_video_transformer(pipe: Any) -> tuple[torch.Tensor, ...]:
    """Default resolution: 9 latent frames, 64x64 spatial (512x320 output approx)."""
    return _dummy_ltx_video_transformer_impl(pipe, num_frames=9, height=40, width=64)


def dummy_ltx_video_text_encoder(pipe: Any) -> tuple[torch.Tensor, ...]:
    text_seq_len = 128
    return (
        torch.zeros(1, text_seq_len, dtype=torch.long),  # input_ids
        torch.ones(1, text_seq_len, dtype=torch.long),  # attention_mask
    )


def dummy_ltx_video_vae_decoder(pipe: Any) -> tuple[torch.Tensor, ...]:
    latent_channels = pipe.vae.config.latent_channels
    dtype = next(pipe.vae.parameters()).dtype
    # 3D VAE: [B, C, T, H, W]
    return (torch.randn(1, latent_channels, 9, 40, 64, dtype=dtype),)


def dummy_ltx_video_vae_encoder(pipe: Any) -> tuple[torch.Tensor, ...]:
    dtype = next(pipe.vae.parameters()).dtype
    # Pixel space video: [B, C, T, H, W]
    return (torch.randn(1, 3, 49, 320, 512, dtype=dtype),)
