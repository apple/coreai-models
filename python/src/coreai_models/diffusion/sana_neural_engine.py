# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Sana Sprint components re-authored for the Neural Engine (``--compute-unit neural-engine``).

Each has the same asset name, input and output names and positional order as the GPU component
in ``components.py``, so the runtime binds them the same way; only the declared dtypes differ.
"""

from typing import Any

import torch

from coreai_models.diffusion.components import SANA_SPRINT_COMPONENTS, NeuralEngineComponentSpec
from coreai_models.diffusion.sana import TEXT_SEQUENCE_LENGTH, sana_text_input_length


def build_transformer(pipe: Any) -> torch.nn.Module:
    from coreai_models.models.ios.sana_transformer import SanaTransformer

    return SanaTransformer.from_source_model(pipe.transformer).to_neural_engine_dtypes()


def dummy_transformer(pipe: Any) -> tuple[torch.Tensor, ...]:
    """float16 latent, text and mask; float32 σ and raw guidance scale.

    See ``SanaTransformer.to_neural_engine_dtypes``.
    """
    cfg = pipe.transformer.config
    size = cfg.sample_size
    return (
        torch.randn(1, cfg.in_channels, size, size, dtype=torch.float16),
        torch.randn(1, TEXT_SEQUENCE_LENGTH, cfg.caption_channels, dtype=torch.float16),
        torch.ones(1, TEXT_SEQUENCE_LENGTH, dtype=torch.float16),
        torch.tensor([0.5], dtype=torch.float32),
        torch.tensor([4.5], dtype=torch.float32),
    )


def build_text_encoder(pipe: Any) -> torch.nn.Module:
    from coreai_models.models.ios.sana_gemma2_encoder import SanaGemma2Encoder

    length = sana_text_input_length(pipe)
    return SanaGemma2Encoder.from_source_model(pipe.text_encoder, length).half()


def dummy_text_encoder(pipe: Any) -> tuple[torch.Tensor, ...]:
    length = sana_text_input_length(pipe)
    return (torch.zeros(1, length, dtype=torch.long), torch.ones(1, length, dtype=torch.long))


def build_vae_decoder(pipe: Any) -> torch.nn.Module:
    from coreai_models.models.ios.sana_dc_ae_decoder import SanaDCAEDecoder

    size = pipe.transformer.config.sample_size
    return SanaDCAEDecoder.from_source_model(pipe.vae.decoder, latent_size=size).half()


def dummy_vae_decoder(pipe: Any) -> tuple[torch.Tensor, ...]:
    size = pipe.transformer.config.sample_size
    return (torch.randn(1, pipe.vae.config.latent_channels, size, size, dtype=torch.float16),)


def _spec(name: str, build_fn: Any, dummy_fn: Any) -> NeuralEngineComponentSpec:
    gpu = SANA_SPRINT_COMPONENTS[name]
    return NeuralEngineComponentSpec(
        asset_name=gpu.asset_name,
        input_names=gpu.input_names,
        output_names=gpu.output_names,
        build_fn=build_fn,
        dummy_fn=dummy_fn,
    )


SANA_SPRINT_NEURAL_ENGINE_COMPONENTS: dict[str, NeuralEngineComponentSpec] = {
    "transformer": _spec("transformer", build_transformer, dummy_transformer),
    "text_encoder": _spec("text_encoder", build_text_encoder, dummy_text_encoder),
    "vae_decoder": _spec("vae_decoder", build_vae_decoder, dummy_vae_decoder),
}
