"""Configuration dataclasses for the DiffusionGemma model.

DiffusionGemma-26B-A4B is a block-diffusion model whose transformer backbone is
the Gemma 4 26B-A4B text architecture used in an encoder/decoder configuration
with shared weights. The configuration is loaded through transformers'
``AutoConfig`` (the ``diffusion_gemma`` model type is registered in
transformers >= 5.11) and mapped into these dataclasses, which add the
export-oriented helpers — unified KV-cache sizing, per-layer attention type, and
per-rope-type parameters — that the Core AI model builder relies on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


def _default_layer_types() -> list[str]:
    # 30-layer pattern from the 26B checkpoint: 5 sliding + 1 full, repeated 5x.
    return (["sliding_attention"] * 5 + ["full_attention"]) * 5


@dataclass
class DiffusionGemmaTextConfig:
    hidden_size: int = 2816
    num_hidden_layers: int = 30
    num_attention_heads: int = 16
    head_dim: int = 256
    global_head_dim: int = 512
    num_key_value_heads: int = 8
    num_global_key_value_heads: int = 2
    intermediate_size: int = 2112
    moe_intermediate_size: int = 704
    num_experts: int = 128
    top_k_experts: int = 8
    vocab_size: int = 262144
    max_position_embeddings: int = 262144
    sliding_window: int = 1024
    rms_norm_eps: float = 1e-6
    final_logit_softcapping: float = 30.0
    hidden_activation: str = "gelu_pytorch_tanh"
    attention_bias: bool = False
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    bos_token_id: int = 2
    eos_token_id: int = 1
    pad_token_id: int = 0
    tie_word_embeddings: bool = True
    use_bidirectional_attention: str = "vision"
    layer_types: list[str] = field(default_factory=_default_layer_types)
    rope_parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0,
                "rope_type": "proportional",
            },
            "sliding_attention": {
                "rope_theta": 10000.0,
                "rope_type": "default",
            },
        }
    )
    dtype: str = "bfloat16"
    model_type: str = "diffusion_gemma_text"

    def is_full_attention_layer(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "full_attention"

    @property
    def cache_head_dim(self) -> int:
        """Max head_dim across layer types — used to size the unified KV cache."""
        return max(self.head_dim, self.global_head_dim)

    @property
    def cache_num_key_value_heads(self) -> int:
        """Max KV heads across layer types — used to size the unified KV cache."""
        return max(self.num_key_value_heads, self.num_global_key_value_heads)

    @property
    def rope_sliding(self) -> dict[str, Any]:
        return self.rope_parameters.get("sliding_attention", {})

    @property
    def rope_full(self) -> dict[str, Any]:
        return self.rope_parameters.get("full_attention", {})

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DiffusionGemmaTextConfig:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class DiffusionGemmaConfig:
    text_config: DiffusionGemmaTextConfig = field(default_factory=DiffusionGemmaTextConfig)
    canvas_length: int = 256
    vision_soft_tokens_per_image: int = 280
    image_token_id: int = 258880
    boi_token_id: int = 255999
    eoi_token_id: int = 258882
    eos_token_id: list[int] = field(default_factory=lambda: [1, 106])
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
    dtype: str = "bfloat16"
    model_type: str = "diffusion_gemma"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DiffusionGemmaConfig:
        text_cfg = DiffusionGemmaTextConfig.from_dict(d.get("text_config", {}))
        top_level_known = {
            "canvas_length",
            "vision_soft_tokens_per_image",
            "image_token_id",
            "boi_token_id",
            "eoi_token_id",
            "eos_token_id",
            "tie_word_embeddings",
            "initializer_range",
            "dtype",
            "model_type",
        }
        top_kwargs = {k: v for k, v in d.items() if k in top_level_known}
        return cls(text_config=text_cfg, **top_kwargs)

    @classmethod
    def from_local(cls, config_path: str) -> DiffusionGemmaConfig:
        with open(config_path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    @classmethod
    def from_pretrained(cls, model_id: str, cache_dir: str | None = None) -> DiffusionGemmaConfig:
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
        return cls.from_dict(hf_config.to_dict())


@dataclass
class DiffusionGemmaGenerationConfig:
    """Hyperparameters governing the block-diffusion denoising loop."""

    max_denoising_steps: int = 48
    canvas_length: int = 256
    confidence_threshold: float = 0.005
    stability_threshold: int = 1
    t_max: float = 0.8
    t_min: float = 0.4
    entropy_bound: float = 0.1
    # image_token_id doubles as the MASK token during diffusion.
    mask_token_id: int = 258880

    @classmethod
    def from_pretrained(
        cls, model_id: str, cache_dir: str | None = None
    ) -> DiffusionGemmaGenerationConfig:
        from huggingface_hub import hf_hub_download

        try:
            path = hf_hub_download(model_id, "generation_config.json", cache_dir=cache_dir)
        except Exception:  # noqa: BLE001 — fall back to defaults if absent.
            return cls()
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        sampler = d.get("sampler_config", {}) or {}
        return cls(
            max_denoising_steps=d.get("max_denoising_steps", 48),
            canvas_length=d.get("canvas_length", 256),
            confidence_threshold=d.get("confidence_threshold", 0.005),
            stability_threshold=d.get("stability_threshold", 1),
            t_max=d.get("t_max", 0.8),
            t_min=d.get("t_min", 0.4),
            entropy_bound=sampler.get("entropy_bound", 0.1),
        )
