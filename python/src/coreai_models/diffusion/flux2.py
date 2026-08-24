# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
FLUX.2 component specifications and torch wrappers for Core AI export.

FLUX.2 Klein 4B is a DiT (Diffusion Transformer) that uses:
- Qwen3 text encoder (intermediate hidden states from layers 9, 18, 27)
- 25-block double-stream + single-stream transformer with 4D RoPE
- AutoencoderKLFlux2 VAE with batch normalization

The transformer computes RoPE in-graph from position IDs (img_ids, txt_ids),
matching upstream diffusers. Position IDs are cheap to build and depend only on
grid geometry, so the exported graph owns the frequency computation.
"""

from typing import Any, cast

import torch

# ---------------------------------------------------------------------------
# Torch wrappers
# ---------------------------------------------------------------------------


class Flux2TransformerWrapper(torch.nn.Module):
    """Wraps Flux2Transformer2DModel for export with in-graph RoPE.

    Takes position IDs and lets the model compute rotary embeddings internally via
    self.pos_embed(), so the exported graph matches upstream diffusers.
    """

    def __init__(self, transformer: torch.nn.Module) -> None:
        super().__init__()
        self.model = transformer

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
    ) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self.model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                guidance=guidance,
                img_ids=img_ids,
                txt_ids=txt_ids,
            ).sample,
        )


class Flux2TextEncoderWrapper(torch.nn.Module):
    """Wraps Qwen3ForCausalLM to extract and concatenate intermediate hidden states.

    FLUX.2 uses hidden states from 3 intermediate layers (default: 9, 18, 27),
    stacked and reshaped from [1, 3, seq_len, 2560] -> [1, seq_len, 7680].
    """

    def __init__(
        self, text_encoder: torch.nn.Module, hidden_states_layers: tuple[int, ...] = (9, 18, 27)
    ) -> None:
        super().__init__()
        self.model = text_encoder
        self.hidden_states_layers = hidden_states_layers

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        stacked = torch.stack([outputs.hidden_states[k] for k in self.hidden_states_layers], dim=1)
        batch_size, num_layers, seq_len, hidden_dim = stacked.shape
        return stacked.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_layers * hidden_dim)


class Flux2VAEDecoderWrapper(torch.nn.Module):
    """Wraps AutoencoderKLFlux2.decode: (latent) -> (image)."""

    def __init__(self, vae: torch.nn.Module) -> None:
        super().__init__()
        self.vae: Any = vae
        # Ensure all parameters + buffers (including BN running stats) share the same dtype
        self.vae = self.vae.to(next(vae.parameters()).dtype)
        from coreai_models.diffusion.components import _patch_nearest_upsample

        _patch_nearest_upsample(self.vae.decoder)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.vae.decode(z).sample)


class Flux2VAEEncoderWrapper(torch.nn.Module):
    """Wraps AutoencoderKLFlux2.encode: (image) -> (latent)."""

    def __init__(self, vae: torch.nn.Module) -> None:
        super().__init__()
        self.vae: Any = vae
        self.vae = self.vae.to(next(vae.parameters()).dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # diffusers encodes img2img reference images with
        # `retrieve_latents(..., sample_mode="argmax")` -> `latent_dist.mode()`,
        # i.e. the distribution MEAN (first `latent_channels` channels), not the
        # raw `parameters` tensor (which is mean concat logvar = 2x channels).
        # Returning `.parameters` would emit 64 channels where the pipeline
        # expects 32, corrupting the img2img latents. `.mode()` is deterministic,
        # so it is also the correct choice for a traced/exported graph.
        return cast(torch.Tensor, self.vae.encode(x).latent_dist.mode())


# ---------------------------------------------------------------------------
# Dummy-input factories
# ---------------------------------------------------------------------------


def _dummy_flux2_transformer_impl(pipe: Any, grid_size: int) -> tuple[torch.Tensor, ...]:
    cfg = pipe.transformer.config
    dtype = next(pipe.transformer.parameters()).dtype
    image_seq_len = grid_size * grid_size
    text_seq_len = 512
    axes_dim = list(cfg.axes_dims_rope)

    # Position IDs per token: [T, H, W, L]. Image tokens carry the spatial grid on
    # axes 1/2; text tokens carry the sequence index on the last axis.
    num_rope_axes = len(axes_dim)
    img_ids = torch.zeros(1, image_seq_len, num_rope_axes)
    for h in range(grid_size):
        for w in range(grid_size):
            idx = h * grid_size + w
            img_ids[0, idx, 1] = float(h)
            img_ids[0, idx, 2] = float(w)

    txt_ids = torch.zeros(1, text_seq_len, num_rope_axes)
    for i in range(text_seq_len):
        txt_ids[0, i, num_rope_axes - 1] = float(i)

    return (
        torch.randn(1, image_seq_len, cfg.in_channels, dtype=dtype),
        torch.randn(1, text_seq_len, cfg.joint_attention_dim, dtype=dtype),
        torch.tensor([0.5], dtype=dtype),
        torch.tensor([1.0], dtype=dtype),
        img_ids,
        txt_ids,
    )


def dummy_flux2_transformer(pipe: Any) -> tuple[torch.Tensor, ...]:
    """1024×1024 (grid=64, seqLen=4096)."""
    return _dummy_flux2_transformer_impl(pipe, grid_size=64)


def dummy_flux2_text_encoder(pipe: Any) -> tuple[torch.Tensor, ...]:
    text_seq_len = 512
    return (
        torch.zeros(1, text_seq_len, dtype=torch.long),  # input_ids
        torch.ones(1, text_seq_len, dtype=torch.long),  # attention_mask
    )


def dummy_flux2_vae_decoder(pipe: Any) -> tuple[torch.Tensor, ...]:
    latent_channels = pipe.vae.config.latent_channels
    sample_size = 128  # 1024 / 8
    dtype = next(pipe.vae.parameters()).dtype
    return (torch.randn(1, latent_channels, sample_size, sample_size, dtype=dtype),)


def dummy_flux2_vae_decoder_half(pipe: Any) -> tuple[torch.Tensor, ...]:
    latent_channels = pipe.vae.config.latent_channels
    sample_size = 64  # 512 / 8
    dtype = next(pipe.vae.parameters()).dtype
    return (torch.randn(1, latent_channels, sample_size, sample_size, dtype=dtype),)


def dummy_flux2_vae_encoder(pipe: Any) -> tuple[torch.Tensor, ...]:
    dtype = next(pipe.vae.parameters()).dtype
    return (torch.randn(1, 3, 1024, 1024, dtype=dtype),)


def dummy_flux2_vae_encoder_half(pipe: Any) -> tuple[torch.Tensor, ...]:
    dtype = next(pipe.vae.parameters()).dtype
    return (torch.randn(1, 3, 512, 512, dtype=dtype),)


def dummy_flux2_transformer_512(pipe: Any) -> tuple[torch.Tensor, ...]:
    """512×512 (grid=32, seqLen=1024)."""
    return _dummy_flux2_transformer_impl(pipe, grid_size=32)
