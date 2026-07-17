# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Diffusion component specifications and torch wrappers.

Each diffusion pipeline is made of independent components (text encoder, UNet,
VAE decoder, VAE encoder) that are exported separately.  A ComponentSpec
captures everything needed to export one component: its I/O names, a thin
torch.nn.Module wrapper that normalises the HF output, and a factory for
dummy inputs.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import torch

from coreai_models.diffusion.flux2 import (
    Flux2TextEncoderWrapper,
    Flux2TransformerPrecomputedRoPEWrapper,
    Flux2VAEDecoderWrapper,
    Flux2VAEEncoderWrapper,
    dummy_flux2_text_encoder,
    dummy_flux2_transformer,
    dummy_flux2_transformer_512,
    dummy_flux2_vae_decoder,
    dummy_flux2_vae_decoder_half,
    dummy_flux2_vae_encoder,
    dummy_flux2_vae_encoder_half,
)

# ---------------------------------------------------------------------------
# Torch wrappers — thin adapters that extract the tensor we need from the
# HuggingFace model's rich output objects.
# ---------------------------------------------------------------------------


class TextEncoderWrapper(torch.nn.Module):
    def __init__(self, text_encoder: torch.nn.Module) -> None:
        super().__init__()
        self.model = text_encoder

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.model(input_ids).last_hidden_state)


class TextEncoderWithPooledWrapper(torch.nn.Module):
    """Returns (last_hidden_state, pooled). Used by SD3 CLIP-L and CLIP-G."""

    def __init__(self, text_encoder: torch.nn.Module) -> None:
        super().__init__()
        self.model = text_encoder
        # CLIPTextModelWithProjection emits text_embeds; CLIPTextModel emits pooler_output.
        self._use_text_embeds = "WithProjection" in type(text_encoder).__name__

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(input_ids)
        pooled = out.text_embeds if self._use_text_embeds else out.pooler_output
        return out.last_hidden_state, pooled


class SD3TransformerWrapper(torch.nn.Module):
    def __init__(self, transformer: torch.nn.Module) -> None:
        super().__init__()
        self.model: Any = transformer

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
    ) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self.model(
                hidden_states=hidden_states.contiguous(),
                encoder_hidden_states=encoder_hidden_states.contiguous(),
                pooled_projections=pooled_projections.contiguous(),
                timestep=timestep,
            ).sample,
        )


class UNetWrapper(torch.nn.Module):
    def __init__(self, unet: torch.nn.Module) -> None:
        super().__init__()
        self.model = unet
        _patch_nearest_upsample(self.model)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return cast(torch.Tensor, self.model(sample, timestep, encoder_hidden_states).sample)


def _patch_nearest_upsample(module: torch.nn.Module) -> None:
    """Replace nearest-neighbor interpolate with repeat_interleave in Upsample2D.

    MPSGraph's segmenter rejects coreai.interpolate with nearest_neighbor mode,
    routing those ops to the BNNS (CPU) backend. This causes two problems:
    1. Mixed-backend execution is unsupported for this op.
    2. Performance: cross-backend data copies (GPU→CPU→GPU) at every upsample
       boundary. Full GPU residency avoids these copies entirely.

    repeat_interleave is mathematically equivalent to nearest-neighbor interpolation
    for integer scale factors and traces to ops that MPSGraph accepts.

    This patch should be kept even after the framework fix ships, because full MPSGraph
    residency is important for inference performance.
    """
    from diffusers.models.upsampling import Upsample2D

    for mod in module.modules():
        if isinstance(mod, Upsample2D):
            original_forward = mod.forward

            def _patched_forward(hidden_states, output_size=None, _orig=original_forward, _mod=mod):
                # Skip the interpolate call — do repeat_interleave instead
                if _mod.use_conv_transpose:
                    return _orig(hidden_states, output_size)

                # Only handles 2× upsample (all current diffusion models use this)
                scale = getattr(_mod, "scale_factor", 2)
                assert scale == 2, (
                    f"_patch_nearest_upsample only supports scale_factor=2, got {scale}"
                )

                dtype = hidden_states.dtype
                if dtype == torch.bfloat16:
                    hidden_states = hidden_states.to(torch.float32)

                if hidden_states.shape[0] >= 64:
                    hidden_states = hidden_states.contiguous()

                # Nearest-neighbor 2x upsample via repeat (only if interpolate is enabled)
                if getattr(_mod, "interpolate", True):
                    hidden_states = hidden_states.repeat_interleave(2, dim=-1).repeat_interleave(
                        2, dim=-2
                    )

                if dtype == torch.bfloat16:
                    hidden_states = hidden_states.to(dtype)

                if _mod.use_conv:
                    if getattr(_mod, "name", "conv") == "conv":
                        hidden_states = _mod.conv(hidden_states)
                    else:
                        hidden_states = _mod.Conv2d_0(hidden_states)

                return hidden_states

            mod.forward = _patched_forward


class VAEDecoderWrapper(torch.nn.Module):
    def __init__(self, vae: torch.nn.Module) -> None:
        super().__init__()
        self.vae: Any = vae
        _patch_nearest_upsample(self.vae.decoder)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.vae.decode(z).sample)


class VAEEncoderWrapper(torch.nn.Module):
    def __init__(self, vae: torch.nn.Module) -> None:
        super().__init__()
        self.vae: Any = vae

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.vae.encode(x).latent_dist.parameters)


# ---------------------------------------------------------------------------
# ComponentSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentSpec:
    asset_name: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    wrapper_fn: Callable
    dummy_fn: Callable
    quantizable: bool = False


# ---------------------------------------------------------------------------
# Dummy-input factories — build reference tensors for torch.export
# ---------------------------------------------------------------------------


def _model_dtype(pipe: Any) -> torch.dtype:
    """Infer the dtype from the pipeline's denoiser weights (UNet or transformer)."""
    denoiser = getattr(pipe, "unet", None) or pipe.transformer
    return cast(torch.dtype, next(denoiser.parameters()).dtype)


def _dummy_text_encoder(pipe: Any, batch_size: int = 2) -> tuple[torch.Tensor, ...]:
    return (torch.zeros(1, 77, dtype=torch.long),)


def _dummy_unet(pipe: Any, batch_size: int = 2) -> tuple[torch.Tensor, ...]:
    cfg = pipe.unet.config
    dtype = _model_dtype(pipe)
    return (
        torch.randn(batch_size, cfg.in_channels, cfg.sample_size, cfg.sample_size, dtype=dtype),
        torch.tensor([999.0] * batch_size, dtype=dtype),
        torch.randn(batch_size, 77, cfg.cross_attention_dim, dtype=dtype),
    )


def _dummy_vae_decoder(pipe: Any, batch_size: int = 2) -> tuple[torch.Tensor, ...]:
    latent_ch = pipe.vae.config.latent_channels
    size = (
        pipe.unet.config.sample_size
        if hasattr(pipe, "unet") and pipe.unet is not None
        else pipe.transformer.config.sample_size
    )
    dtype = next(pipe.vae.parameters()).dtype
    return (torch.randn(1, latent_ch, size, size, dtype=dtype),)


def _dummy_vae_encoder(pipe: Any, batch_size: int = 2) -> tuple[torch.Tensor, ...]:
    size = (
        pipe.unet.config.sample_size
        if hasattr(pipe, "unet") and pipe.unet is not None
        else pipe.transformer.config.sample_size
    )
    dtype = _model_dtype(pipe)
    return (torch.randn(1, 3, size * 8, size * 8, dtype=dtype),)


def _dummy_sd3_transformer(pipe: Any, batch_size: int = 2) -> tuple[torch.Tensor, ...]:
    cfg = pipe.transformer.config
    dtype = _model_dtype(pipe)
    return (
        torch.randn(batch_size, cfg.in_channels, cfg.sample_size, cfg.sample_size, dtype=dtype),
        torch.tensor([999.0] * batch_size, dtype=dtype),
        torch.randn(batch_size, 154, cfg.joint_attention_dim, dtype=dtype),
        torch.randn(batch_size, cfg.pooled_projection_dim, dtype=dtype),
    )


# ---------------------------------------------------------------------------
# Component registries
# ---------------------------------------------------------------------------

SD_COMPONENTS: dict[str, ComponentSpec] = {
    "text_encoder": ComponentSpec(
        asset_name="TextEncoder",
        input_names=("input_ids",),
        output_names=("last_hidden_state",),
        wrapper_fn=lambda p: TextEncoderWrapper(p.text_encoder),
        dummy_fn=_dummy_text_encoder,
        quantizable=True,
    ),
    "unet": ComponentSpec(
        asset_name="Unet",
        input_names=("sample", "timestep", "encoder_hidden_states"),
        output_names=("noise_pred",),
        wrapper_fn=lambda p: UNetWrapper(p.unet),
        dummy_fn=_dummy_unet,
        quantizable=True,
    ),
    "vae_decoder": ComponentSpec(
        asset_name="VAEDecoder",
        input_names=("z",),
        output_names=("image",),
        wrapper_fn=lambda p: VAEDecoderWrapper(p.vae),
        dummy_fn=_dummy_vae_decoder,
    ),
    "vae_encoder": ComponentSpec(
        asset_name="VAEEncoder",
        input_names=("image",),
        output_names=("latent_params",),
        wrapper_fn=lambda p: VAEEncoderWrapper(p.vae),
        dummy_fn=_dummy_vae_encoder,
    ),
}

ALL_SD_COMPONENTS: list[str] = list(SD_COMPONENTS.keys())

FLUX2_COMPONENTS: dict[str, ComponentSpec] = {
    "transformer": ComponentSpec(
        asset_name="Transformer",
        input_names=(
            "hidden_states",
            "encoder_hidden_states",
            "timestep",
            "guidance",
            "rotary_emb_cos",
            "rotary_emb_sin",
        ),
        output_names=("output",),
        wrapper_fn=lambda p: Flux2TransformerPrecomputedRoPEWrapper(p.transformer),
        dummy_fn=dummy_flux2_transformer,
        quantizable=True,
    ),
    "transformer_512": ComponentSpec(
        asset_name="Transformer_512",
        input_names=(
            "hidden_states",
            "encoder_hidden_states",
            "timestep",
            "guidance",
            "rotary_emb_cos",
            "rotary_emb_sin",
        ),
        output_names=("output",),
        wrapper_fn=lambda p: Flux2TransformerPrecomputedRoPEWrapper(p.transformer),
        dummy_fn=dummy_flux2_transformer_512,
        quantizable=True,
    ),
    "text_encoder": ComponentSpec(
        asset_name="TextEncoder",
        input_names=("input_ids", "attention_mask"),
        output_names=("hidden_states",),
        wrapper_fn=lambda p: Flux2TextEncoderWrapper(p.text_encoder),
        dummy_fn=dummy_flux2_text_encoder,
        quantizable=False,
    ),
    "vae_decoder": ComponentSpec(
        asset_name="VAEDecoder",
        input_names=("z",),
        output_names=("image",),
        wrapper_fn=lambda p: Flux2VAEDecoderWrapper(p.vae),
        dummy_fn=dummy_flux2_vae_decoder,
    ),
    "vae_decoder_half": ComponentSpec(
        asset_name="VAEDecoder_half",
        input_names=("z",),
        output_names=("image",),
        wrapper_fn=lambda p: Flux2VAEDecoderWrapper(p.vae),
        dummy_fn=dummy_flux2_vae_decoder_half,
    ),
    "vae_encoder": ComponentSpec(
        asset_name="VAEEncoder",
        input_names=("image",),
        output_names=("latent_params",),
        wrapper_fn=lambda p: Flux2VAEEncoderWrapper(p.vae),
        dummy_fn=dummy_flux2_vae_encoder,
    ),
    "vae_encoder_half": ComponentSpec(
        asset_name="VAEEncoder_half",
        input_names=("image",),
        output_names=("latent_params",),
        wrapper_fn=lambda p: Flux2VAEEncoderWrapper(p.vae),
        dummy_fn=dummy_flux2_vae_encoder_half,
    ),
}

ALL_FLUX2_COMPONENTS: list[str] = list(FLUX2_COMPONENTS.keys())


SD3_COMPONENTS: dict[str, ComponentSpec] = {
    "text_encoder": ComponentSpec(
        asset_name="TextEncoder",
        input_names=("input_ids",),
        output_names=("hidden_embeds", "pooled_outputs"),
        wrapper_fn=lambda p: TextEncoderWithPooledWrapper(p.text_encoder),
        dummy_fn=_dummy_text_encoder,
        quantizable=True,
    ),
    "text_encoder_2": ComponentSpec(
        asset_name="TextEncoder2",
        input_names=("input_ids",),
        output_names=("hidden_embeds", "pooled_outputs"),
        wrapper_fn=lambda p: TextEncoderWithPooledWrapper(p.text_encoder_2),
        dummy_fn=_dummy_text_encoder,
        quantizable=True,
    ),
    "transformer": ComponentSpec(
        asset_name="MMDiT",
        input_names=("sample", "timestep", "encoder_hidden_states", "pooled_projections"),
        output_names=("noise_pred",),
        wrapper_fn=lambda p: SD3TransformerWrapper(p.transformer),
        dummy_fn=_dummy_sd3_transformer,
        quantizable=True,
    ),
    "vae_decoder": ComponentSpec(
        asset_name="VAEDecoder",
        input_names=("z",),
        output_names=("image",),
        wrapper_fn=lambda p: VAEDecoderWrapper(p.vae),
        dummy_fn=_dummy_vae_decoder,
    ),
}

ALL_SD3_COMPONENTS: list[str] = list(SD3_COMPONENTS.keys())


def get_component_registry(
    hf_pipe: Any,
    pipeline_type: str = "sd",
) -> dict[str, ComponentSpec]:
    """Return the component registry for the given pipeline type.

    Args:
        hf_pipe: The loaded HuggingFace pipeline (unused for routing, but
            available for future introspection).
        pipeline_type: One of "sd", "sd3", or "flux2".
    """
    if pipeline_type == "flux2":
        return FLUX2_COMPONENTS
    if pipeline_type == "sd3":
        return SD3_COMPONENTS
    return SD_COMPONENTS


def get_valid_components(pipeline_type: str) -> list[str]:
    """Return valid component names for a given pipeline type."""
    if pipeline_type == "flux2":
        return ALL_FLUX2_COMPONENTS
    if pipeline_type == "sd3":
        return ALL_SD3_COMPONENTS
    return ALL_SD_COMPONENTS
