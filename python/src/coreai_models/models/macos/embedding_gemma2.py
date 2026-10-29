# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""EmbeddingGemma 2 (EG2) text encoder for Core AI export.

Bidirectional encoder with per-layer inputs and a sentence-embedding head: masked mean over the
valid tokens (prompt included), bias-free projection to 768, then L2 normalization. Masks derive
from the static sequence length and ``attention_mask``, so the graph has no data-dependent control
flow. The GPU path runs in float32; the float16 variant lives in the iOS modules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from coreai_models.models.macos.diffusion_gemma import (
    DiffusionGemmaDenseMLP as EmbeddingGemma2MLP,
)
from coreai_models.models.macos.diffusion_gemma import (
    DiffusionGemmaRMSNorm as EmbeddingGemma2RMSNorm,
)
from coreai_models.models.macos.diffusion_gemma import (
    ScaledEmbedding,
    apply_rotary_pos_emb,
)
from coreai_models.primitives.macos.sdpa import SDPA

HF_TEXT_PREFIX = "language_model."


@dataclass
class EmbeddingGemma2TextConfig:
    """Text-tower config parsed from the HF ``text_config`` block (needs no transformers>=5.19)."""

    vocab_size: int = 262_144
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_hidden_layers: int = 24
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 256
    hidden_activation: str = "gelu_pytorch_tanh"
    rms_norm_eps: float = 1e-6
    pad_token_id: int = 0
    attention_bias: bool = False
    sliding_window: int = 512
    layer_types: list[str] = field(default_factory=list)
    hidden_size_per_layer_input: int = 512
    embedding_dim: int = 768
    rope_theta_sliding: float = 10_000.0
    rope_theta_full: float = 1_000_000.0
    # Per-layer overrides {layer_idx: {"head_dim": int, "num_key_value_heads": int}}.
    per_layer_config: dict[int, dict[str, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries, "
                f"expected {self.num_hidden_layers}"
            )
        if self.hidden_activation != "gelu_pytorch_tanh":
            raise ValueError(f"Unsupported hidden_activation {self.hidden_activation!r}")
        self.per_layer_config = {int(k): dict(v) for k, v in self.per_layer_config.items()}

    def layer_head_dim(self, layer_idx: int) -> int:
        return int(self.per_layer_config.get(layer_idx, {}).get("head_dim", self.head_dim))

    def layer_num_key_value_heads(self, layer_idx: int) -> int:
        override = self.per_layer_config.get(layer_idx, {})
        return int(override.get("num_key_value_heads", self.num_key_value_heads))

    def is_sliding(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "sliding_attention"

    @classmethod
    def from_dict(cls, text_config: dict[str, Any]) -> EmbeddingGemma2TextConfig:
        """Build from an HF ``text_config`` dict (``config.json["text_config"]``).

        ``layer_types``, ``per_layer_config`` and both ``rope_parameters`` entries define the layer
        layout, so they are required; the other fields default to EmbeddingGemma 2's values.
        """
        missing = [
            key
            for key in ("layer_types", "per_layer_config", "rope_parameters")
            if not text_config.get(key)
        ]
        if missing:
            raise ValueError(f"text_config is missing {missing}")
        rope = text_config["rope_parameters"]
        thetas = {}
        for kind in ("sliding_attention", "full_attention"):
            params = rope.get(kind) or {}
            if "rope_theta" not in params:
                raise ValueError(f"rope_parameters[{kind!r}] has no rope_theta")
            rope_type = params.get("rope_type", "default")
            if rope_type != "default":
                raise ValueError(f"Unsupported rope_type {rope_type!r} for {kind}")
            thetas[kind] = float(params["rope_theta"])
        names = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in text_config.items() if k in names and v is not None}
        kwargs["rope_theta_sliding"] = thetas["sliding_attention"]
        kwargs["rope_theta_full"] = thetas["full_attention"]
        per_layer = text_config["per_layer_config"]
        kwargs["per_layer_config"] = {
            int(k): {kk: int(vv) for kk, vv in dict(v).items() if vv is not None}
            for k, v in per_layer.items()
        }
        return cls(**kwargs)

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> EmbeddingGemma2TextConfig:
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
        return cls.from_dict(cfg.get("text_config", cfg))


def build_attention_masks(
    attention_mask: torch.Tensor, sliding_window: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return boolean ``(full, sliding)`` masks of shape ``[B, 1, L, L]`` (True = attend).

    ``full``    = key_padding
    ``sliding`` = ``|i - j| <= sliding_window`` AND key_padding

    Matches ``create_bidirectional_mask`` / ``create_bidirectional_sliding_window_mask`` in
    transformers (``masking_utils.sliding_window_bidirectional_overlay`` +
    ``padding_mask_function``). Rows of padded queries may be fully masked; the SDPA composite
    converts the boolean mask into a finite additive bias, so those rows stay finite and they are
    excluded from pooling anyway.
    """
    batch, seq_len = attention_mask.shape
    key_valid = (attention_mask != 0).reshape(batch, 1, 1, seq_len)
    positions = torch.arange(seq_len, device=attention_mask.device, dtype=torch.int32)
    distance = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs()
    band = (distance <= sliding_window).reshape(1, 1, seq_len, seq_len)
    full = key_valid.expand(batch, 1, seq_len, seq_len)
    sliding = torch.logical_and(band, key_valid)
    return full, sliding


class EmbeddingGemma2Attention(nn.Module):
    """Bidirectional attention; q/k RMSNorm + RoPE, scale-free v_norm, scaling 1.0."""

    def __init__(self, config: EmbeddingGemma2TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.layer_num_key_value_heads(layer_idx)
        self.head_dim = config.layer_head_dim(layer_idx)
        dim, hd = config.hidden_size, self.head_dim
        bias = config.attention_bias

        self.q_proj = nn.Linear(dim, self.n_heads * hd, bias=bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * hd, bias=bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * hd, bias=bias)
        self.o_proj = nn.Linear(self.n_heads * hd, dim, bias=bias)
        self.q_norm = EmbeddingGemma2RMSNorm(hd, eps=config.rms_norm_eps)
        self.k_norm = EmbeddingGemma2RMSNorm(hd, eps=config.rms_norm_eps)
        self.v_norm = EmbeddingGemma2RMSNorm(hd, eps=config.rms_norm_eps, with_scale=False)
        # The mask carries both the window and the key padding; the composite itself stays
        # windowless and non-causal.
        self.sdpa = SDPA(scale=1.0, window_size=0, is_causal=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        n_h, n_kv, hd = self.n_heads, self.n_kv_heads, self.head_dim

        query = self.q_norm(self.q_proj(x).reshape(batch, seq_len, n_h, hd))
        query = apply_rotary_pos_emb(query, cos, sin, unsqueeze_dim=2).transpose(1, 2)
        key = self.k_norm(self.k_proj(x).reshape(batch, seq_len, n_kv, hd))
        key = apply_rotary_pos_emb(key, cos, sin, unsqueeze_dim=2).transpose(1, 2)
        value = self.v_norm(self.v_proj(x).reshape(batch, seq_len, n_kv, hd)).transpose(1, 2)

        out = self.sdpa(query=query, key=key, value=value, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(batch, seq_len, n_h * hd)
        return self.o_proj(out)


class EmbeddingGemma2PLE(nn.Module):
    """Projection-only per-layer embeddings: inputs_embeds -> [B, L, num_layers, ple_dim]."""

    def __init__(self, config: EmbeddingGemma2TextConfig) -> None:
        super().__init__()
        self.num_layers = config.num_hidden_layers
        self.ple_dim = config.hidden_size_per_layer_input
        self.per_layer_model_projection = nn.Linear(
            config.hidden_size, self.num_layers * self.ple_dim, bias=False
        )
        self.scale = config.hidden_size**-0.5
        self.per_layer_projection_norm = EmbeddingGemma2RMSNorm(
            self.ple_dim, eps=config.rms_norm_eps
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = inputs_embeds.shape
        proj = self.per_layer_model_projection(inputs_embeds) * self.scale
        proj = proj.reshape(batch, seq_len, self.num_layers, self.ple_dim)
        return self.per_layer_projection_norm(proj)


class EmbeddingGemma2PLEBlock(nn.Module):
    """Third residual sub-block: gate -> GELU -> * per_layer_input -> project -> norm -> add."""

    def __init__(self, config: EmbeddingGemma2TextConfig) -> None:
        super().__init__()
        ple_dim = config.hidden_size_per_layer_input
        self.per_layer_input_gate = nn.Linear(config.hidden_size, ple_dim, bias=False)
        self.per_layer_projection = nn.Linear(ple_dim, config.hidden_size, bias=False)
        self.post_per_layer_input_norm = EmbeddingGemma2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, x: torch.Tensor, per_layer_input: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.per_layer_input_gate(x), approximate="tanh") * per_layer_input
        return x + self.post_per_layer_input_norm(self.per_layer_projection(h))


class EmbeddingGemma2EncoderLayer(nn.Module):
    def __init__(self, config: EmbeddingGemma2TextConfig, layer_idx: int) -> None:
        super().__init__()
        eps = config.rms_norm_eps
        self.is_sliding = config.is_sliding(layer_idx)
        self.self_attn = EmbeddingGemma2Attention(config, layer_idx)
        self.mlp = EmbeddingGemma2MLP(config)
        self.input_layernorm = EmbeddingGemma2RMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = EmbeddingGemma2RMSNorm(config.hidden_size, eps=eps)
        self.pre_feedforward_layernorm = EmbeddingGemma2RMSNorm(config.hidden_size, eps=eps)
        self.post_feedforward_layernorm = EmbeddingGemma2RMSNorm(config.hidden_size, eps=eps)
        self.ple_block = EmbeddingGemma2PLEBlock(config)
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(
        self,
        x: torch.Tensor,
        per_layer_input: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.self_attn(self.input_layernorm(x), cos, sin, attn_mask)
        x = x + self.post_attention_layernorm(h)
        h = self.mlp(self.pre_feedforward_layernorm(x))
        x = x + self.post_feedforward_layernorm(h)
        x = self.ple_block(x, per_layer_input)
        return x * self.layer_scalar


class EmbeddingGemma2TextEncoder(nn.Module):
    """EG2 text tower + sentence-embedding head (masked mean -> projection -> L2).

    Parameter names mirror the HF ``language_model.*`` namespace (minus the prefix), so a
    checkpoint loads with a prefix strip and no renames.
    """

    def __init__(self, config: EmbeddingGemma2TextConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = ScaledEmbedding(
            config.vocab_size, config.hidden_size, embed_scale=config.hidden_size**0.5
        )
        self.ple = EmbeddingGemma2PLE(config)
        self.layers = nn.ModuleList(
            [EmbeddingGemma2EncoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = EmbeddingGemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embedding_projection = nn.Linear(config.hidden_size, config.embedding_dim, bias=False)

        sliding_dims = {
            config.layer_head_dim(i) for i in range(len(self.layers)) if config.is_sliding(i)
        }
        full_dims = {
            config.layer_head_dim(i) for i in range(len(self.layers)) if not config.is_sliding(i)
        }
        if len(sliding_dims) > 1 or len(full_dims) > 1:
            raise ValueError("RoPE tables assume one head_dim per layer type")
        sliding_hd = next(iter(sliding_dims), config.head_dim)
        full_hd = next(iter(full_dims), config.head_dim)
        self.register_buffer(
            "sliding_inv_freq",
            self._inv_freq(sliding_hd, config.rope_theta_sliding),
            persistent=False,
        )
        self.register_buffer(
            "full_inv_freq", self._inv_freq(full_hd, config.rope_theta_full), persistent=False
        )

    @staticmethod
    def _inv_freq(head_dim: int, base: float, device: torch.device | None = None) -> torch.Tensor:
        return 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
        )

    @staticmethod
    def _cos_sin(
        inv_freq: torch.Tensor, seq_len: int, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Half-dim cos/sin ``[1, L, head_dim/2]`` for positions ``0..L-1`` (fp32 math)."""
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = positions.unsqueeze(1) * inv_freq.float().to(device).unsqueeze(0)
        freqs = freqs.unsqueeze(0)
        return freqs.cos().to(dtype), freqs.sin().to(dtype)

    def encode_tokens(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the tower; returns final-normed hidden states ``[B, L, hidden_size]``."""
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Pass exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        batch, seq_len, _ = inputs_embeds.shape
        if attention_mask is None:
            attention_mask = torch.ones(
                batch, seq_len, dtype=torch.int32, device=inputs_embeds.device
            )

        per_layer_inputs = self.ple(inputs_embeds)
        full_mask, sliding_mask = build_attention_masks(attention_mask, self.config.sliding_window)
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        sliding_cs = self._cos_sin(self.sliding_inv_freq, seq_len, dtype, device)
        full_cs = self._cos_sin(self.full_inv_freq, seq_len, dtype, device)

        h = inputs_embeds
        for i, layer in enumerate(self.layers):
            cos, sin = sliding_cs if layer.is_sliding else full_cs
            mask = sliding_mask if layer.is_sliding else full_mask
            h = layer(h, per_layer_inputs[:, :, i, :], cos, sin, mask)
        return self.norm(h)

    def pool(self, hidden: torch.Tensor, pooling_mask: torch.Tensor) -> torch.Tensor:
        """Masked mean (fp32) -> projection -> L2 normalize. Returns ``[B, embedding_dim]``."""
        mask = pooling_mask.to(torch.float32).unsqueeze(-1)
        summed = (hidden.to(torch.float32) * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1e-9)
        mean = summed / count
        projected = F.linear(mean, self.embedding_projection.weight.to(torch.float32))
        sq_norm = (projected * projected).sum(dim=-1, keepdim=True)
        return projected * torch.rsqrt(sq_norm.clamp(min=1e-24))

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pooling_mask: torch.Tensor | None = None,
        return_token_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return ``sentence_embedding`` ``[B, embedding_dim]`` (unit L2 norm, fp32).

        ``pooling_mask`` defaults to ``attention_mask`` (include_prompt=True); pass a narrower
        mask to exclude e.g. prompt tokens from the mean. With ``return_token_states`` the
        per-token projected states ``[B, L, embedding_dim]`` (HF ``last_hidden_state``
        equivalent) are returned as a second output for parity checks.
        """
        hidden = self.encode_tokens(input_ids, attention_mask, inputs_embeds)
        if attention_mask is None:
            attention_mask = torch.ones(hidden.shape[:2], dtype=torch.int32, device=hidden.device)
        if pooling_mask is None:
            pooling_mask = attention_mask
        sentence_embedding = self.pool(hidden, pooling_mask)
        if return_token_states:
            return sentence_embedding, self.embedding_projection(hidden)
        return sentence_embedding


def extract_text_state_dict(
    state_dict: dict[str, torch.Tensor], prefix: str = HF_TEXT_PREFIX
) -> dict[str, torch.Tensor]:
    """Keep only ``prefix*`` keys (vision/audio dropped) and strip the prefix."""
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}


def load_text_state_dict(
    model: EmbeddingGemma2TextEncoder, state_dict: dict[str, torch.Tensor]
) -> None:
    """Load an (already prefix-stripped) text state dict; fail closed on any missing/extra key."""
    info = model.load_state_dict(state_dict, strict=False)
    if info.missing_keys or info.unexpected_keys:
        raise RuntimeError(
            f"EG2 text load mismatch: missing={info.missing_keys[:10]} "
            f"unexpected={info.unexpected_keys[:10]}"
        )


def _resolve_model_dir(model_dir_or_id: str | Path) -> Path:
    path = Path(model_dir_or_id).expanduser()
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            str(model_dir_or_id), allow_patterns=["config.json", "*.safetensors", "*.json"]
        )
    )


def load_embedding_gemma2_text_encoder(
    model_dir_or_id: str | Path = "google/embeddinggemma-2",
    dtype: torch.dtype = torch.float32,
) -> EmbeddingGemma2TextEncoder:
    """Build the text encoder from a local snapshot dir or HF id, loading ``language_model.*``."""
    from safetensors import safe_open

    model_dir = _resolve_model_dir(model_dir_or_id)
    config = EmbeddingGemma2TextConfig.from_pretrained(model_dir)
    model = EmbeddingGemma2TextEncoder(config)

    state_dict: dict[str, torch.Tensor] = {}
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors in {model_dir}")
    for file in files:
        with safe_open(str(file), framework="pt") as handle:
            for key in handle.keys():  # noqa: SIM118 - safe_open is not a Mapping
                if key.startswith(HF_TEXT_PREFIX):
                    state_dict[key[len(HF_TEXT_PREFIX) :]] = handle.get_tensor(key).to(dtype)
    load_text_state_dict(model, state_dict)
    return model.to(dtype).eval()


__all__ = [
    "EmbeddingGemma2TextConfig",
    "EmbeddingGemma2TextEncoder",
    "build_attention_masks",
    "extract_text_state_dict",
    "load_embedding_gemma2_text_encoder",
    "load_text_state_dict",
]
