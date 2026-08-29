# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Gemma 3n text decoder for CoreAI model export.

Novel features vs standard decoder-only transformers:
- AltUp: 4-copy hidden state with predict/correct routing per layer
- Per-layer input embeddings with gated injection
- LAUREL: low-rank residual branch
- KV sharing: last N layers reuse KV from earlier layers
- Gaussian TopK activation sparsity in early MLP layers
- Dual RoPE: local (θ=10K) + global (θ=1M)
- QKV norms with scaling=1.0
"""

import math

import torch
import torch.nn as nn
from transformers.models.gemma3n.configuration_gemma3n import Gemma3nTextConfig
from transformers.models.gemma3n.modeling_gemma3n import (
    Gemma3nForCausalLM as HFGemma3nForCausalLM,
)
from typing_extensions import Self, override

from coreai_models.models.base import BaseForCausalLM
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.rope import initialize_rope
from coreai_models.primitives.macos.sdpa import SDPA


class AltUp(nn.Module):
    def __init__(self, config: Gemma3nTextConfig) -> None:
        super().__init__()
        self.num_inputs = config.altup_num_inputs
        self.active_idx = config.altup_active_idx
        hidden = config.hidden_size

        self.correct_output_scale = nn.Parameter(torch.zeros(hidden))
        self.correction_coefs = nn.Linear(self.num_inputs, self.num_inputs, bias=False)
        self.prediction_coefs = nn.Linear(self.num_inputs, self.num_inputs**2, bias=False)
        self.modality_router = nn.Linear(hidden, self.num_inputs, bias=False)
        self.router_norm = RMSNorm(hidden, eps=config.rms_norm_eps)
        self.router_input_scale = hidden**-1.0

    def _compute_modalities(self, x: torch.Tensor) -> torch.Tensor:
        router_inputs = self.router_norm(x) * self.router_input_scale
        return torch.tanh(self.modality_router(router_inputs).float()).to(x.dtype)

    def predict(self, copies: list[torch.Tensor]) -> list[torch.Tensor]:
        N = self.num_inputs
        modalities = self._compute_modalities(copies[self.active_idx])
        # all_coefs: [B, S, N_in, N_out] after transpose on static NxN dims
        all_coefs = (
            self.prediction_coefs(modalities)
            .reshape(*modalities.shape[:-1], N, N)
            .transpose(-2, -1)
        )
        # Stack inputs on last dim: [B, S, H, N_in] (static N, dynamic S untouched)
        stacked = torch.stack(copies, dim=-1)
        # Batched matmul: [B, S, H, N_in] @ [B, S, N_in, N_out] -> [B, S, H, N_out]
        predictions_stacked = torch.matmul(stacked, all_coefs)
        # Unstack + residual
        return [predictions_stacked[..., i] + copies[i] for i in range(N)]

    def correct(
        self, predictions: list[torch.Tensor], activated: torch.Tensor
    ) -> list[torch.Tensor]:
        N = self.num_inputs
        modalities = self._compute_modalities(activated)
        innovation = activated - predictions[self.active_idx]  # [B, S, H]
        # correction_coefs: [B, S, N], add 1.0 bias
        coefs = self.correction_coefs(modalities) + 1.0  # [B, S, N]
        # For each copy i: corrected[i] = coefs[..., i:i+1] * innovation + predictions[i]
        return [coefs[..., i : i + 1] * innovation + predictions[i] for i in range(N)]

    def scale_output(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.correct_output_scale


class Laurel(nn.Module):
    def __init__(self, config: Gemma3nTextConfig) -> None:
        super().__init__()
        self.linear_left = nn.Linear(config.hidden_size, config.laurel_rank, bias=False)
        self.linear_right = nn.Linear(config.laurel_rank, config.hidden_size, bias=False)
        self.post_laurel_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.post_laurel_norm(self.linear_right(self.linear_left(x)))


class Attention(nn.Module):
    def __init__(self, config: Gemma3nTextConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.head_dim = head_dim = config.head_dim

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)

        layer_types = config.layer_types or [
            "full_attention" if (i + 1) % 5 == 0 else "sliding_attention"
            for i in range(config.num_hidden_layers)
        ]
        self.is_sliding = layer_types[layer_idx] == "sliding_attention"

        first_kv_shared = config.num_hidden_layers - config.num_kv_shared_layers
        self.is_kv_shared = layer_idx >= first_kv_shared > 0

        if not self.is_kv_shared:
            self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
            self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
            self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)

        rope_params = getattr(config, "rope_parameters", None) or {}
        if self.is_sliding:
            self.sdpa = SDPA(is_causal=True, scale=1.0, window_size=config.sliding_window)
            local_theta = rope_params.get("sliding_attention", {}).get(
                "rope_theta", getattr(config, "rope_local_base_freq", 10000.0)
            )
            self.rope = initialize_rope(base=local_theta)
        else:
            self.sdpa = SDPA(is_causal=True, scale=1.0)
            global_theta = rope_params.get("full_attention", {}).get(
                "rope_theta", getattr(config, "rope_theta", 1000000.0)
            )
            self.rope = initialize_rope(base=global_theta)

        self._v_norm_eps = config.rms_norm_eps

        # Cache slot index: shared layers map to their source layer's slot.
        # Built by Gemma3nModel and assigned after construction.
        self.cache_slot: int = layer_idx

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        batch_size, query_len, _ = x.shape
        n_heads, n_kv_heads = self.n_heads, self.n_kv_heads

        query = (
            self.q_proj(x)
            .reshape(batch_size, query_len, n_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        query = self.q_norm(query)

        seq_len = position_ids.shape[-1]
        torch._check_is_size(query_len)
        torch._check_is_size(seq_len)
        offset = seq_len - query_len
        torch._check_is_size(offset)
        rope_positions = position_ids.narrow(-1, offset, query_len)

        query = self.rope(query, position_ids=rope_positions)

        if self.is_kv_shared and cache is not None:
            # Fetch K/V from source layer's cache slot via the custom op (no-op
            # write with a zero-length update keeps the graph MPS-traceable).
            empty_k = x.new_empty(batch_size, n_kv_heads, 0, self.head_dim)
            empty_v = x.new_empty(batch_size, n_kv_heads, 0, self.head_dim)
            key, value = cache.update_and_fetch(
                self.cache_slot, offset, empty_k, empty_v, seq_len=seq_len, query_len=0
            )
        else:
            key = (
                self.k_proj(x)
                .reshape(batch_size, query_len, n_kv_heads, self.head_dim)
                .permute(0, 2, 1, 3)
            )
            key = self.k_norm(key)
            key = self.rope(key, position_ids=rope_positions)

            value = (
                self.v_proj(x)
                .reshape(batch_size, query_len, n_kv_heads, self.head_dim)
                .permute(0, 2, 1, 3)
            )
            value = value / torch.sqrt(value.pow(2).mean(-1, keepdim=True) + self._v_norm_eps)

            if cache is not None:
                key, value = cache.update_and_fetch(
                    self.cache_slot, offset, key, value, seq_len=seq_len, query_len=query_len
                )

        output = (
            self.sdpa(query, key, value)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, query_len, self.n_heads * self.head_dim)
        )
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, config: Gemma3nTextConfig, layer_idx: int) -> None:
        super().__init__()
        hidden = config.hidden_size
        intermediate = config.intermediate_size[layer_idx]
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
        self.act_fn = nn.GELU(approximate="tanh")
        self.activation_sparsity = config.activation_sparsity_pattern[layer_idx]
        if self.activation_sparsity > 0.0:
            from scipy.special import erfinv as _erfinv

            self._std_mult = float(_erfinv(2.0 * self.activation_sparsity - 1.0) * math.sqrt(2.0))

    def _gaussian_topk(self, x: torch.Tensor) -> torch.Tensor:
        # Compute mean/std in float32 to avoid fp16 overflow in variance
        x_f32 = x.float()
        mu = x_f32.mean(dim=-1, keepdim=True)
        variance = (x_f32 - mu).pow(2).mean(dim=-1, keepdim=True)
        std = variance.sqrt()
        cutoff = (mu + std * self._std_mult).to(x.dtype)
        return nn.functional.relu(x - cutoff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        if self.activation_sparsity > 0.0:
            gate = self._gaussian_topk(gate)
        return self.down_proj(self.act_fn(gate) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: Gemma3nTextConfig, layer_idx: int) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.layer_idx = layer_idx
        self.active_idx = config.altup_active_idx
        self.hidden_size_per_layer = config.hidden_size_per_layer_input

        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config, layer_idx)
        self.altup = AltUp(config)
        self.laurel = Laurel(config)

        self.input_layernorm = RMSNorm(hidden, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(hidden, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(hidden, eps=config.rms_norm_eps)

        pli_dim = config.hidden_size_per_layer_input
        self.per_layer_input_gate = nn.Linear(hidden, pli_dim, bias=False)
        self.per_layer_projection = nn.Linear(pli_dim, hidden, bias=False)
        self.post_per_layer_input_norm = RMSNorm(hidden, eps=config.rms_norm_eps)

        self.altup_correct_scale = getattr(config, "altup_correct_scale", True)
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(
        self,
        hidden_states: list[torch.Tensor],
        per_layer_input: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
    ) -> list[torch.Tensor]:
        predictions = self.altup.predict(hidden_states)
        active = predictions[self.active_idx]

        active_normed = self.input_layernorm(active)
        laurel_output = self.laurel(active_normed)

        attn = self.self_attn(active_normed, position_ids, cache)
        attn = self.post_attention_layernorm(attn)

        attn_gated = active + attn
        attn_laurel = (attn_gated + laurel_output) / math.sqrt(2)

        ffw = self.mlp(self.pre_feedforward_layernorm(attn_laurel))
        ffw = self.post_feedforward_layernorm(ffw)
        ffw_out = attn_laurel + ffw

        corrected = self.altup.correct(predictions, ffw_out)

        first = corrected[self.active_idx].clone()
        if self.altup_correct_scale:
            first = self.altup.scale_output(first)

        first = self.act_fn(self.per_layer_input_gate(first))
        first = torch.multiply(first, per_layer_input)
        first = self.post_per_layer_input_norm(self.per_layer_projection(first))
        for i in range(1, len(corrected)):
            corrected[i] = corrected[i] + first

        return corrected


class Gemma3nModel(nn.Module):
    def __init__(self, config: Gemma3nTextConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        self.per_layer_projection_scale = hidden**-0.5
        self.per_layer_input_scale = 2.0**-0.5

        self.embed_tokens = nn.Embedding(config.vocab_size, hidden)
        self.embed_tokens_per_layer = nn.Embedding(
            config.vocab_size_per_layer_input,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
        )
        self.embed_scale = hidden**0.5
        self.per_layer_embed_scale = config.hidden_size_per_layer_input**0.5
        self.per_layer_model_projection = nn.Linear(
            hidden, config.num_hidden_layers * config.hidden_size_per_layer_input, bias=False
        )
        self.per_layer_projection_norm = RMSNorm(
            config.hidden_size_per_layer_input, eps=config.rms_norm_eps
        )

        self.altup_projections = nn.ModuleList(
            [nn.Linear(hidden, hidden, bias=False) for _ in range(config.altup_num_inputs - 1)]
        )
        self.altup_unembed_projections = nn.ModuleList(
            [nn.Linear(hidden, hidden, bias=False) for _ in range(config.altup_num_inputs - 1)]
        )

        self.layers = nn.ModuleList(
            [TransformerBlock(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(hidden, eps=config.rms_norm_eps)

        layer_types = config.layer_types or [
            "full_attention" if (i + 1) % 5 == 0 else "sliding_attention"
            for i in range(config.num_hidden_layers)
        ]
        first_kv_shared = max(0, config.num_hidden_layers - config.num_kv_shared_layers)
        self._kv_share_map: dict[int, int] = {}
        if first_kv_shared > 0 and config.num_kv_shared_layers > 0:
            non_shared_types = layer_types[:first_kv_shared]
            for i in range(first_kv_shared, config.num_hidden_layers):
                lt = layer_types[i]
                if lt in non_shared_types:
                    src = len(non_shared_types) - 1 - non_shared_types[::-1].index(lt)
                    self._kv_share_map[i] = src

        # Build compact cache slot mapping: non-shared layers get slots 0..N-1,
        # shared layers point to their source layer's slot.
        slot_for_layer: dict[int, int] = {}
        next_slot = 0
        for i in range(config.num_hidden_layers):
            if i not in self._kv_share_map:
                slot_for_layer[i] = next_slot
                next_slot += 1
        for i, src in self._kv_share_map.items():
            slot_for_layer[i] = slot_for_layer[src]
        self.num_cache_slots = next_slot

        for layer in self.layers:
            layer.self_attn.cache_slot = slot_for_layer[layer.layer_idx]

    def _get_per_layer_inputs(
        self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        per_layer = self.embed_tokens_per_layer(input_ids) * self.per_layer_embed_scale
        pli_dim = self.config.hidden_size_per_layer_input
        per_layer = per_layer.reshape(
            *input_ids.shape, self.config.num_hidden_layers, pli_dim
        ).contiguous()
        proj = self.per_layer_model_projection(inputs_embeds) * self.per_layer_projection_scale
        proj = proj.reshape(
            *inputs_embeds.shape[:-1], self.config.num_hidden_layers, pli_dim
        ).contiguous()
        proj = self.per_layer_projection_norm(proj)
        return (proj + per_layer) * self.per_layer_input_scale

    def _altup_expand(self, h: torch.Tensor) -> list[torch.Tensor]:
        target_mag = torch.mean(h**2, dim=-1, keepdim=True) ** 0.5
        eps = torch.tensor(1e-5, device=h.device, dtype=h.dtype)
        copies = [h]
        for proj in self.altup_projections:
            p = proj(h).to(h.dtype)
            mag = torch.sqrt(torch.maximum(torch.mean(p**2, dim=-1, keepdim=True), eps))
            copies.append(p * target_mag / mag)
        return copies

    def _altup_collapse(self, hidden_states: list[torch.Tensor]) -> torch.Tensor:
        target_mag = torch.mean(hidden_states[0] ** 2, dim=-1, keepdim=True) ** 0.5
        eps = torch.tensor(1e-5, device=hidden_states[0].device, dtype=hidden_states[0].dtype)
        copies = [hidden_states[0]]
        for i, proj in enumerate(self.altup_unembed_projections):
            p = proj(hidden_states[i + 1]).to(hidden_states[0].dtype)
            mag = torch.sqrt(torch.maximum(torch.mean(p**2, dim=-1, keepdim=True), eps))
            copies.append(p * target_mag / mag)
        return torch.mean(torch.stack(copies), dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        per_layer_inputs = self._get_per_layer_inputs(input_ids, inputs_embeds)

        hidden_states = self._altup_expand(inputs_embeds)

        for layer in self.layers:
            idx = layer.layer_idx
            per_layer_input = per_layer_inputs[:, :, idx, :]
            hidden_states = layer(hidden_states, per_layer_input, position_ids, cache)

        hidden_states = self._altup_collapse(hidden_states)
        return self.norm(hidden_states)


class Gemma3nForCausalLM(BaseForCausalLM):
    _HF_MODEL_CLASS = HFGemma3nForCausalLM

    @classmethod
    @override
    def from_hf_memory_efficient(
        cls,
        huggingface_model_id: str,
        max_context_length: int | None = None,
        target_dtype: torch.dtype = torch.float16,
        mmap_path: str | None = None,
        num_layers: int | None = None,
        hf_config_attr: str | None = "text_config",
        hf_state_dict_prefix: str = "model.language_model.",
        disable_embedding_quantization: bool = False,
    ):
        import gc
        import json
        import os
        import re
        from types import SimpleNamespace

        from huggingface_hub import snapshot_download
        from safetensors import safe_open

        from coreai_models.models.base import _load_tensors_for_keys, _resolve_safetensors_files

        model_dir = snapshot_download(
            huggingface_model_id,
            allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"],
        )

        with open(os.path.join(model_dir, "config.json")) as f:
            raw = json.load(f)
        cfg_dict = raw.get(hf_config_attr, raw) if hf_config_attr else raw
        hf_config = SimpleNamespace(**cfg_dict) if isinstance(cfg_dict, dict) else cfg_dict

        config = cls._get_reauthored_config(hf_config, max_context_length, num_layers=num_layers)
        model = cls(config=config, model_device="meta")
        model.to(dtype=target_dtype)

        safetensors_files = _resolve_safetensors_files(model_dir)

        prefix = hf_state_dict_prefix
        layer_pattern = re.compile(re.escape(prefix) + r"layers\.(\d+)\.")

        per_layer: dict[int, dict[str, str]] = {}
        shared: dict[str, str] = {}
        for path in safetensors_files:
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():  # noqa: SIM118
                    if not key.startswith(prefix) and key != "lm_head.weight":
                        continue
                    match = layer_pattern.match(key)
                    if match:
                        layer_idx = int(match.group(1))
                        if num_layers is not None and layer_idx >= num_layers:
                            continue
                        per_layer.setdefault(layer_idx, {})[key] = path
                    else:
                        shared[key] = path

        shared_dict = _load_tensors_for_keys(shared, target_dtype)
        normalized: dict[str, torch.Tensor] = {}
        for k, v in shared_dict.items():
            if k.startswith(prefix):
                normalized["model." + k[len(prefix) :]] = v
            else:
                normalized[k] = v
        del shared_dict
        model._mutate_state_dict(normalized)
        model.load_state_dict(normalized, assign=True, strict=False)
        del normalized
        gc.collect()

        for layer_idx in sorted(per_layer.keys()):
            layer_key_to_file = per_layer.pop(layer_idx)
            layer_sd = _load_tensors_for_keys(layer_key_to_file, target_dtype)
            del layer_key_to_file
            remapped: dict[str, torch.Tensor] = {}
            for k, v in layer_sd.items():
                remapped["model." + k[len(prefix) :]] = v
            del layer_sd
            model._mutate_state_dict(remapped)
            model.load_state_dict(remapped, assign=True, strict=False)
            del remapped
            gc.collect()

        if getattr(config, "tie_word_embeddings", True):
            model.lm_head.weight = model.model.embed_tokens.weight

        meta_params = [n for n, p in model.named_parameters() if p.is_meta]
        if meta_params:
            raise RuntimeError(f"Parameters not loaded: {meta_params}")

        return model

    @classmethod
    @override
    def _get_reauthored_config(cls, hf_config, max_context_length=None, num_layers=None):
        text_config = hf_config.text_config if hasattr(hf_config, "text_config") else hf_config
        if max_context_length is not None:
            text_config.max_position_embeddings = max_context_length
        if num_layers is not None:
            text_config.num_hidden_layers = num_layers
        if text_config.rope_scaling is not None:
            text_config.rope_scaling = None
        return text_config

    @override
    def _init_model(self, config: Gemma3nTextConfig) -> None:
        self.model = Gemma3nModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    @classmethod
    def create_cache_tensors(
        cls, config, dtype: torch.dtype = torch.float32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create KV cache with compact slots (shared layers reuse source slots)."""
        n_kv_heads = config.num_key_value_heads
        max_seq_len = config.max_position_embeddings
        head_dim = config.head_dim

        first_kv_shared = config.num_hidden_layers - config.num_kv_shared_layers
        n_slots = first_kv_shared if config.num_kv_shared_layers > 0 else config.num_hidden_layers

        k_cache = torch.zeros(n_slots, 1, n_kv_heads, max_seq_len, head_dim, dtype=dtype)
        v_cache = torch.zeros(n_slots, 1, n_kv_heads, max_seq_len, head_dim, dtype=dtype)
        return k_cache, v_cache

    @BaseForCausalLM.cast_logits_bfloat16_to_float16
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        cache = KVCache(k_cache, v_cache)
        out = self.model(input_ids, position_ids, cache)
        logits = self.lm_head(out)
        if hasattr(self.config, "final_logit_softcapping") and self.config.final_logit_softcapping:
            cap = self.config.final_logit_softcapping
            logits = logits / cap
            logits = torch.tanh(logits)
            logits = logits * cap
        return logits

    @override
    def _mutate_state_dict(self: Self, state_dict: dict[str, torch.Tensor]) -> None:
        # Keys arrive in one of two forms:
        # (a) Raw: "model.language_model.layers.0.self_attn.q_proj.weight"
        # (b) Already-stripped by from_hf_memory_efficient: "layers.0.self_attn.q_proj.weight"
        # Normalize all to "model.layers.N.*" / "model.embed_tokens.*" / "lm_head.*"
        prefix = "model.language_model."
        keys = list(state_dict.keys())
        for key in keys:
            if key.startswith("model.visual.") or key.startswith("model.audio_tower."):
                del state_dict[key]
            elif key.startswith(prefix):
                state_dict["model." + key[len(prefix) :]] = state_dict.pop(key)
            elif key.startswith("language_model."):
                state_dict["model." + key[len("language_model.") :]] = state_dict.pop(key)
            elif (
                key.startswith("layers.")
                or key.startswith("norm.")
                or key.startswith("embed_tokens")
                or key.startswith("altup_")
                or key.startswith("per_layer_")
            ):
                state_dict["model." + key] = state_dict.pop(key)

        # Strip v_norm (scale-free norm, no learned weights in our impl)
        for key in list(state_dict.keys()):
            if "v_norm" in key:
                del state_dict[key]

        # Slice per-layer tensors when num_layers is truncated
        n_layers = self.config.num_hidden_layers
        pli_dim = self.config.hidden_size_per_layer_input
        expected_pli_total = n_layers * pli_dim

        pli_embed_key = "model.embed_tokens_per_layer.weight"
        if pli_embed_key in state_dict and state_dict[pli_embed_key].shape[-1] > expected_pli_total:
            state_dict[pli_embed_key] = state_dict[pli_embed_key][:, :expected_pli_total]

        pli_proj_key = "model.per_layer_model_projection.weight"
        if pli_proj_key in state_dict and state_dict[pli_proj_key].shape[0] > expected_pli_total:
            state_dict[pli_proj_key] = state_dict[pli_proj_key][:expected_pli_total, :]

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        super().load_state_dict(state_dict, strict=strict, assign=assign)
        if getattr(self.config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight
