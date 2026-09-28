# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Sana Sprint component specifications and torch wrappers for Core AI export.

Sana Sprint 0.6B is a few-step (1-4) distilled linear-attention DiT that uses:
- Gemma-2-2B text encoder (last hidden state), prompt prefixed with a fixed instruction
- SanaTransformer2DModel with guidance embedding and no positional embedding
- DC-AE (AutoencoderDC) with 32 latent channels at 32× spatial compression

Its TrigFlow/SCM sampler is flow matching under a change of variables: with
σ = sin t / (sin t + cos t) the transformer input is the flow latent and its output
is the flow velocity. The exported transformer is therefore driven like any other
flow-matching DiT (timestep = σ), and the runtime renoises with fresh noise per step.
"""

import inspect
from typing import Any, cast

import torch

# Diffusers keeps the first token (BOS) and the last `max_sequence_length - 1` of the
# instruction-prefixed prompt; see `SanaSprintPipeline.encode_prompt`.
TEXT_SEQUENCE_LENGTH = 300


def sana_prompt_prefix() -> str:
    """The instruction diffusers prepends to every prompt (its `complex_human_instruction`)."""
    from diffusers import SanaSprintPipeline

    lines = inspect.signature(SanaSprintPipeline.__call__).parameters["complex_human_instruction"]
    return "\n".join(lines.default)


def sana_text_input_length(pipe: Any) -> int:
    """Tokenized length the text encoder is traced at, matching diffusers' `max_length_all`."""
    return len(pipe.tokenizer.encode(sana_prompt_prefix())) + TEXT_SEQUENCE_LENGTH - 2


# ---------------------------------------------------------------------------
# Torch wrappers
# ---------------------------------------------------------------------------


class SanaTransformerWrapper(torch.nn.Module):
    """Wraps SanaTransformer2DModel: (latent, text, mask, σ, guidance) -> velocity.

    The mask is a float input so the runtime can bind it alongside the float tensors.
    Guidance is scaled by `guidance_embeds_scale` here so callers pass the raw scale.
    """

    def __init__(self, transformer: torch.nn.Module) -> None:
        super().__init__()
        self.model = transformer
        self.guidance_embeds_scale = float(transformer.config.guidance_embeds_scale)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        timestep: torch.Tensor,
        guidance: torch.Tensor,
    ) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self.model(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                guidance=guidance * self.guidance_embeds_scale,
                return_dict=False,
            )[0],
        )


class SanaTextEncoderWrapper(torch.nn.Module):
    """Wraps Gemma2Model: (input_ids, attention_mask) [1, L] -> hidden_states [1, 300, D].

    Applies diffusers' token selection in-graph: BOS plus the last 299 positions.
    """

    def __init__(self, text_encoder: torch.nn.Module) -> None:
        super().__init__()
        self.model = text_encoder

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).last_hidden_state
        return torch.cat([hidden[:, :1], hidden[:, -(TEXT_SEQUENCE_LENGTH - 1) :]], dim=1)


class SanaVAEDecoderWrapper(torch.nn.Module):
    """Wraps AutoencoderDC.decode: (z) -> (image). The runtime divides by scaling_factor."""

    def __init__(self, vae: torch.nn.Module) -> None:
        super().__init__()
        self.vae: Any = vae

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.vae.decode(z).sample)


# ---------------------------------------------------------------------------
# Dummy-input factories
# ---------------------------------------------------------------------------


def _dummy_sana_transformer_impl(pipe: Any, grid_size: int) -> tuple[torch.Tensor, ...]:
    cfg = pipe.transformer.config
    dtype = next(pipe.transformer.parameters()).dtype
    return (
        torch.randn(1, cfg.in_channels, grid_size, grid_size, dtype=dtype),
        torch.randn(1, TEXT_SEQUENCE_LENGTH, cfg.caption_channels, dtype=dtype),
        torch.ones(1, TEXT_SEQUENCE_LENGTH, dtype=dtype),
        torch.tensor([0.5], dtype=dtype),
        torch.tensor([4.5], dtype=dtype),
    )


def dummy_sana_transformer(pipe: Any) -> tuple[torch.Tensor, ...]:
    """1024×1024: 32×32 latent grid (1024 tokens)."""
    return _dummy_sana_transformer_impl(pipe, grid_size=pipe.transformer.config.sample_size)


def dummy_sana_transformer_quant_trace(pipe: Any) -> tuple[torch.Tensor, ...]:
    """Small trace for the weight quantizer; there is no positional embedding to crop."""
    return _dummy_sana_transformer_impl(pipe, grid_size=8)


def dummy_sana_text_encoder(pipe: Any) -> tuple[torch.Tensor, ...]:
    seq_len = sana_text_input_length(pipe)
    return (
        torch.zeros(1, seq_len, dtype=torch.long),
        torch.ones(1, seq_len, dtype=torch.long),
    )


def dummy_sana_text_encoder_quant_trace(pipe: Any) -> tuple[torch.Tensor, ...]:
    return (torch.zeros(1, 16, dtype=torch.long), torch.ones(1, 16, dtype=torch.long))


def dummy_sana_vae_decoder(pipe: Any) -> tuple[torch.Tensor, ...]:
    cfg = pipe.vae.config
    dtype = next(pipe.vae.parameters()).dtype
    size = pipe.transformer.config.sample_size
    return (torch.randn(1, cfg.latent_channels, size, size, dtype=dtype),)
