# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Model registry mapping HuggingFace model_type to model classes."""

from dataclasses import dataclass
from functools import lru_cache

import torch.nn as nn


def _register_novel_configs() -> None:
    """Register model types not in our transformers version with AutoConfig."""
    try:
        from transformers import AutoConfig, PretrainedConfig
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

        if "muse_glimmer" not in CONFIG_MAPPING_NAMES:

            class _MuseGlimmerTextConfig(PretrainedConfig):
                model_type = "muse_glimmer_text"

                def __init__(self, **kwargs):
                    kwargs.setdefault("hidden_size", 64)
                    kwargs.setdefault("num_attention_heads", 4)
                    kwargs.setdefault("num_key_value_heads", 2)
                    kwargs.setdefault("intermediate_size", 128)
                    kwargs.setdefault("vocab_size", 200)
                    kwargs.setdefault("max_position_embeddings", 512)
                    kwargs.setdefault("head_dim", 16)
                    kwargs.setdefault("rms_norm_eps", 1e-5)
                    kwargs.setdefault("sliding_window", 8)
                    kwargs.setdefault("output_multiplier", 0.196)
                    kwargs.setdefault("qk_scale_factor", 3.87)
                    kwargs.setdefault("final_logit_softcapping", 20.0)
                    kwargs.setdefault("tie_word_embeddings", False)
                    kwargs.setdefault("post_norm_eps", 1e-8)
                    n_layers = kwargs.setdefault("num_hidden_layers", 4)
                    # Ensure layer_types/layer_rope_theta match num_hidden_layers
                    pattern = ["sliding_attention"] * 3 + ["full_attention"]
                    theta_pattern = [500000.0, 500000.0, 500000.0, 0]
                    kwargs.setdefault("layer_types", (pattern * ((n_layers // 4) + 1))[:n_layers])
                    kwargs.setdefault(
                        "layer_rope_theta", (theta_pattern * ((n_layers // 4) + 1))[:n_layers]
                    )
                    super().__init__(**kwargs)

            class _MuseGlimmerConfig(PretrainedConfig):
                model_type = "muse_glimmer"

                def __init__(self, **kwargs):
                    tc = kwargs.pop("text_config", None)
                    super().__init__(**kwargs)
                    if isinstance(tc, dict):
                        self.text_config = _MuseGlimmerTextConfig(**tc)
                    elif tc is not None:
                        self.text_config = tc

            AutoConfig.register("muse_glimmer", _MuseGlimmerConfig)
            AutoConfig.register("muse_glimmer_text", _MuseGlimmerTextConfig)
    except Exception:
        pass


_register_novel_configs()


@dataclass
class ModelEntry:
    """Registry entry for a model family."""

    macos_class: type[nn.Module] | None = None
    ios_class: type[nn.Module] | None = None
    # Multimodal-checkpoint hooks consumed by `from_hf_memory_efficient` and the
    # `text_config` unwrap in the export pipeline.
    # `hf_config_attr`: attribute on the top-level HF config holding the
    #     per-modality sub-config (e.g. "text_config" for Gemma-3).
    # `hf_state_dict_prefix`: prefix on safetensors keys for this modality
    #     (e.g. "language_model." for Gemma-3). Stripped before assignment.
    hf_config_attr: str | None = None
    hf_state_dict_prefix: str = ""


@lru_cache(maxsize=1)
def _get_registry() -> dict[str, ModelEntry]:
    """Build the model registry (cached singleton). Lazy imports to avoid circular deps."""
    from coreai_models.models.ios.mistral import MistralForCausalLMForiOS
    from coreai_models.models.ios.qwen2 import Qwen2ForCausalLMForiOS
    from coreai_models.models.ios.qwen3 import Qwen3ForCausalLMForiOS
    from coreai_models.models.macos.gemma3_text import Gemma3ForCausalLM
    from coreai_models.models.macos.gpt_oss import GptOssForCausalLM
    from coreai_models.models.macos.mistral import MistralForCausalLM
    from coreai_models.models.macos.mixtral import MixtralForCausalLM
    from coreai_models.models.macos.muse_glimmer import MuseGlimmerForCausalLM
    from coreai_models.models.macos.qwen2 import Qwen2ForCausalLM
    from coreai_models.models.macos.qwen3 import Qwen3ForCausalLM
    from coreai_models.models.macos.qwen3_moe import Qwen3MoeForCausalLM
    from coreai_models.models.macos.qwen3_vl import (
        Qwen3VLForCausalLM,
    )

    return {
        "gemma3_text": ModelEntry(
            macos_class=Gemma3ForCausalLM,
            hf_config_attr="text_config",
            hf_state_dict_prefix="language_model.",
        ),
        "gpt_oss": ModelEntry(
            macos_class=GptOssForCausalLM,
        ),
        "mistral": ModelEntry(
            macos_class=MistralForCausalLM,
            ios_class=MistralForCausalLMForiOS,
        ),
        "mixtral": ModelEntry(
            macos_class=MixtralForCausalLM,
        ),
        "muse_glimmer_text": ModelEntry(
            macos_class=MuseGlimmerForCausalLM,
            hf_config_attr="text_config",
            hf_state_dict_prefix="model.language_model.",
        ),
        "qwen2": ModelEntry(
            macos_class=Qwen2ForCausalLM,
            ios_class=Qwen2ForCausalLMForiOS,
        ),
        "qwen3": ModelEntry(
            macos_class=Qwen3ForCausalLM,
            ios_class=Qwen3ForCausalLMForiOS,
        ),
        "qwen3_moe": ModelEntry(
            macos_class=Qwen3MoeForCausalLM,
        ),
        # Qwen3-VL: vision-language model.
        # macos_class = standard text decoder (input_ids, for text-only use).
        # Qwen3VLForCausalLMEmbeddings is used by the VLM export script (takes inputs_embeds).
        "qwen3_vl": ModelEntry(
            macos_class=Qwen3VLForCausalLM,
            hf_config_attr="text_config",
            hf_state_dict_prefix="model.language_model.",
        ),
    }


# Type alias for the remapping dict
MODEL_TYPE_REMAPPING: dict[str, str] = {
    "gemma3": "gemma3_text",
    "muse_glimmer": "muse_glimmer_text",
    "qwen2_5": "qwen2",
}


def get_model_entry(model_type: str) -> ModelEntry:
    """Look up a model by HuggingFace model_type."""
    registry = _get_registry()
    remapped = MODEL_TYPE_REMAPPING.get(model_type, model_type)
    if remapped not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise KeyError(f"Unknown model type '{model_type}'. Available: {available}")
    return registry[remapped]


def list_models() -> list[str]:
    """List all supported model types."""
    return sorted(_get_registry().keys())
