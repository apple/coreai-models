# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause


import torch
import torch.nn as nn
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4ForCausalLM as HFGemma4ForCausalLM,
)

from coreai_models.models.base import BaseForCausalLMForiOS
from coreai_models.primitives.ios.cache import KVCacheHandler
from coreai_models.primitives.ios.embedding import GatherEmbeddings
from coreai_models.primitives.ios.quantization import dequantize_per_tensor, quantize_per_tensor
from coreai_models.primitives.ios.rms_norm import RMSNorm
from coreai_models.primitives.ios.rope import RoPECache, apply_rope
from coreai_models.primitives.ios.sdpa import SDPA


class ProportionalRoPECache(RoPECache):
    """RoPE cache with zero-padded inv_freq for NoPE dimensions.

    Computes cos/sin for the full head_dim but sets inv_freq = 0 for the
    non-rotated tail. apply_rope then acts as identity on those dimensions
    (cos=1, sin=0), so no slicing is needed anywhere in the model.
    """

    def __init__(
        self,
        head_dim: int,
        max_cache_size: int,
        base: float,
        partial_rotary_factor: float,
    ) -> None:
        self._partial_rotary_factor = partial_rotary_factor
        super().__init__(head_dim, max_cache_size, base)

    def _compute_sin_and_cos(self, dtype: torch.dtype = torch.float32) -> None:
        head_dim = self._head_dim
        max_cache_size = self._max_cache_size
        base = self._base

        with torch.device("cpu"):
            rope_angles = int(self._partial_rotary_factor * head_dim // 2)
            nope_angles = head_dim // 2 - rope_angles

            inv_freq = 1.0 / (
                base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32) / head_dim)
            )
            if nope_angles > 0:
                inv_freq = torch.cat(
                    [inv_freq, torch.zeros(nope_angles, dtype=torch.float32)], dim=0
                )

            seq_idx = torch.arange(end=max_cache_size, dtype=torch.int32)
            freqs = seq_idx[:, None] * inv_freq
            emb = torch.concatenate((freqs, freqs), dim=-1)
            self.cos_cached = torch.nn.Buffer(torch.cos(emb).to(dtype=dtype), persistent=False)
            self.sin_cached = torch.nn.Buffer(torch.sin(emb).to(dtype=dtype), persistent=False)


def _v_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Per-head RMSNorm without learned weights (Gemma4 v_norm has with_scale=False)."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class MLP(nn.Module):
    """GeLU-gated MLP using Conv2d for iOS layout (B, S, 1, D)."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.up_proj = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.down_proj = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _, D = x.shape
        x = x.reshape(B * L, D, 1, 1)
        gate = nn.functional.gelu(self.gate_proj(x))
        out = self.down_proj(gate * self.up_proj(x))
        return out.reshape(B, L, 1, D)


class Attention(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        layer_types: list[str] = getattr(config, "layer_types", None) or []
        layer_type = layer_types[layer_idx] if layer_idx < len(layer_types) else "sliding_attention"
        self.is_sliding = layer_type == "sliding_attention"

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self._rms_norm_eps = config.rms_norm_eps

        self.use_alt_attn = not self.is_sliding and getattr(config, "attention_k_eq_v", False)
        if self.use_alt_attn:
            n_kv_heads = (
                getattr(config, "num_global_key_value_heads", None) or config.num_key_value_heads
            )
        else:
            n_kv_heads = config.num_key_value_heads
        self.n_kv_heads = n_kv_heads

        self.head_dim = (
            config.head_dim
            if self.is_sliding
            else (getattr(config, "global_head_dim", None) or config.head_dim)
        )
        head_dim = self.head_dim

        # Type-local cache slot index.
        if layer_types:
            same_type = [i for i, t in enumerate(layer_types) if t == layer_type]
            self.cache_layer_idx = (
                same_type.index(layer_idx) if layer_idx in same_type else layer_idx
            )
        else:
            self.cache_layer_idx = layer_idx

        # KV sharing: last num_kv_shared_layers reuse K/V from a donor full-attention layer.
        num_kv_shared = getattr(config, "num_kv_shared_layers", 0)
        first_shared = config.num_hidden_layers - num_kv_shared
        self.is_kv_shared = num_kv_shared > 0 and layer_idx >= first_shared

        self.donor_cache_idx: int | None = None
        if self.is_kv_shared:
            same_type_before_shared = [
                i for i, t in enumerate(layer_types[:first_shared]) if t == layer_type
            ]
            if same_type_before_shared:
                donor_global = same_type_before_shared[-1]
                same_type_all = [i for i, t in enumerate(layer_types) if t == layer_type]
                self.donor_cache_idx = same_type_all.index(donor_global)

        self.sdpa = SDPA(head_dim=head_dim, scale=1.0)

        # Projections (Conv2d for iOS).
        if self.is_kv_shared:
            self.q_proj = nn.Conv2d(dim, n_heads * head_dim, kernel_size=1, bias=False)
        elif self.use_alt_attn:
            self.q_proj = nn.Conv2d(dim, n_heads * head_dim, kernel_size=1, bias=False)
            self.k_proj = nn.Conv2d(dim, n_kv_heads * head_dim, kernel_size=1, bias=False)
        else:
            self.q_proj = nn.Conv2d(dim, n_heads * head_dim, kernel_size=1, bias=False)
            self.k_proj = nn.Conv2d(dim, n_kv_heads * head_dim, kernel_size=1, bias=False)
            self.v_proj = nn.Conv2d(dim, n_kv_heads * head_dim, kernel_size=1, bias=False)
        self.o_proj = nn.Conv2d(n_heads * head_dim, dim, kernel_size=1, bias=False)

        # Per-head norms — single shared weight per head (same as HF).
        self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        if not self.is_kv_shared:
            self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)

    def _to_heads(self, t: torch.Tensor, B: int, L: int, n: int) -> torch.Tensor:
        """Conv2d output (B, n*hd, 1, L) → (B, n, L, hd) for norms/RoPE."""
        return t.transpose(-3, -1).reshape(B, L, n, self.head_dim).transpose(-2, -3)

    def _to_conv(self, t: torch.Tensor, B: int, L: int, n: int) -> torch.Tensor:
        """(B, n, L, hd) → (B, n*hd, 1, L) Conv2d format."""
        return t.transpose(-3, -2).reshape(B, L, 1, n * self.head_dim).transpose(-3, -1)

    def forward(
        self,
        x: torch.Tensor,
        sliding_rope_cos: torch.Tensor,
        sliding_rope_sin: torch.Tensor,
        full_rope_cos: torch.Tensor,
        full_rope_sin: torch.Tensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        sliding_cache: KVCacheHandler | None,
        full_cache: KVCacheHandler | None,
    ) -> torch.Tensor:
        B, L, _, _ = x.shape
        n_heads, n_kv = self.n_heads, self.n_kv_heads
        hd = self.head_dim

        xc = x.transpose(-3, -1)  # (B, dim, 1, L) for Conv2d

        active_cache = sliding_cache if self.is_sliding else full_cache
        rope_cos = sliding_rope_cos if self.is_sliding else full_rope_cos
        rope_sin = sliding_rope_sin if self.is_sliding else full_rope_sin

        if self.is_kv_shared:
            query = self._to_heads(self.q_proj(xc), B, L, n_heads)
            query = self.q_norm(query)
            query = apply_rope(query, rope_cos, rope_sin)
            query = self._to_conv(query, B, L, n_heads)

            donor = self.donor_cache_idx
            if active_cache is not None and donor is not None:
                key = active_cache._k_cache[donor]  # (1, n_kv*hd, 1, C)
                value = active_cache._v_cache[donor]  # (1, n_kv*hd, 1, C)
            else:
                seq_len = rope_cos.shape[1]
                key = torch.zeros(B, n_kv * hd, 1, seq_len, device=x.device, dtype=x.dtype)
                value = key
        else:
            query = self._to_heads(self.q_proj(xc), B, L, n_heads)
            key = self._to_heads(self.k_proj(xc), B, L, n_kv)

            query = self.q_norm(query)
            key = self.k_norm(key)

            query = apply_rope(query, rope_cos, rope_sin)
            key = apply_rope(key, rope_cos, rope_sin)

            if self.use_alt_attn:
                value_heads = _v_norm(key, self._rms_norm_eps)
            else:
                value_heads = self._to_heads(self.v_proj(xc), B, L, n_kv)
                value_heads = _v_norm(value_heads, self._rms_norm_eps)

            query = self._to_conv(query, B, L, n_heads)
            key = self._to_conv(key, B, L, n_kv)
            value = self._to_conv(value_heads, B, L, n_kv)

            if active_cache is not None:
                key, value = active_cache.update_and_fetch(
                    self.cache_layer_idx, in_step, key, value, L
                )

        output = self.sdpa(query, key, value, causal_mask)
        output = self.o_proj(output)
        return output.transpose(-3, -1)  # back to (B, L, 1, dim)


class TransformerBlock(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.self_attn = Attention(config, layer_idx=layer_idx)

        num_kv_shared = getattr(config, "num_kv_shared_layers", 0)
        first_shared = config.num_hidden_layers - num_kv_shared
        is_shared = num_kv_shared > 0 and layer_idx >= first_shared
        double_wide = getattr(config, "use_double_wide_mlp", False) and is_shared
        self.mlp = MLP(hidden_size, config.intermediate_size * (2 if double_wide else 1))

        eps = config.rms_norm_eps
        self.input_layernorm = RMSNorm(hidden_size, eps=eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=eps)
        self.pre_feedforward_layernorm = RMSNorm(hidden_size, eps=eps)
        self.post_feedforward_layernorm = RMSNorm(hidden_size, eps=eps)

        self.ple_dim = getattr(config, "hidden_size_per_layer_input", 0)
        if self.ple_dim:
            self.per_layer_input_gate = nn.Conv2d(
                hidden_size, self.ple_dim, kernel_size=1, bias=False
            )
            self.per_layer_projection = nn.Conv2d(
                self.ple_dim, hidden_size, kernel_size=1, bias=False
            )
            self.post_per_layer_input_norm = RMSNorm(hidden_size, eps=eps)

        with torch.device("cpu"):
            self.layer_scalar = nn.Buffer(torch.ones(1), persistent=True)

    def forward(
        self,
        x: torch.Tensor,
        sliding_rope_cos: torch.Tensor,
        sliding_rope_sin: torch.Tensor,
        full_rope_cos: torch.Tensor,
        full_rope_sin: torch.Tensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        sliding_cache: KVCacheHandler | None,
        full_cache: KVCacheHandler | None,
        per_layer_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r = self.self_attn(
            self.input_layernorm(x),
            sliding_rope_cos,
            sliding_rope_sin,
            full_rope_cos,
            full_rope_sin,
            in_step,
            causal_mask,
            sliding_cache,
            full_cache,
        )
        h = x + self.post_attention_layernorm(r)
        mlp_out = self.mlp(self.pre_feedforward_layernorm(h))
        h = h + self.post_feedforward_layernorm(mlp_out)

        if self.ple_dim and per_layer_input is not None:
            # per_layer_input: (B, L, 1, ple_dim)
            B, L, _, _ = h.shape
            hc = h.reshape(B * L, h.shape[-1], 1, 1)
            plc = per_layer_input.reshape(B * L, self.ple_dim, 1, 1)
            gate = nn.functional.gelu(self.per_layer_input_gate(hc))
            ple_out = self.per_layer_projection(gate * plc)
            ple_out = ple_out.reshape(B, L, 1, h.shape[-1])
            h = h + self.post_per_layer_input_norm(ple_out)

        return h * self.layer_scalar


class Gemma4TextModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.layers = nn.ModuleList(
            [TransformerBlock(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        sliding_rope_cos: torch.Tensor,
        sliding_rope_sin: torch.Tensor,
        full_rope_cos: torch.Tensor,
        full_rope_sin: torch.Tensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        sliding_cache: KVCacheHandler | None,
        full_cache: KVCacheHandler | None,
        per_layer_inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            ple = (
                per_layer_inputs[:, :, :, i * layer.ple_dim : (i + 1) * layer.ple_dim]
                if (per_layer_inputs is not None and layer.ple_dim > 0)
                else None
            )
            token_embeddings = layer(
                token_embeddings,
                sliding_rope_cos,
                sliding_rope_sin,
                full_rope_cos,
                full_rope_sin,
                in_step,
                causal_mask,
                sliding_cache,
                full_cache,
                ple,
            )
        return self.norm(token_embeddings)


class LoadPLEEmbeddings(nn.Module):
    """Holds the per-layer embedding table as a separate loadable module."""

    def __init__(self, config) -> None:
        super().__init__()
        ple_vocab = config.vocab_size_per_layer_input
        n_layers = config.num_hidden_layers
        ple_dim = config.hidden_size_per_layer_input
        self.ple_embedding_table = nn.Parameter(
            torch.zeros(ple_vocab, 1, n_layers * ple_dim, dtype=torch.int8),
            requires_grad=False,
        )

    def forward(self) -> torch.Tensor:
        return self.ple_embedding_table


class Gemma4Extend(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.model = Gemma4TextModel(config)
        self.prefill_mode = False

        hidden_size = config.hidden_size
        n_layers = config.num_hidden_layers
        layer_types: list[str] = getattr(config, "layer_types", None) or []

        n_sliding = sum(1 for t in layer_types if t == "sliding_attention") or n_layers
        n_full = sum(1 for t in layer_types if t == "full_attention")

        n_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim

        use_alt = getattr(config, "attention_k_eq_v", False)
        n_kv_heads_full = (
            getattr(config, "num_global_key_value_heads", None) or n_kv_heads
            if use_alt
            else n_kv_heads
        )
        global_head_dim = getattr(config, "global_head_dim", None) or head_dim

        self.sliding_kv_cache = KVCacheHandler(n_sliding, n_kv_heads * head_dim)
        self.full_kv_cache = (
            KVCacheHandler(n_full, n_kv_heads_full * global_head_dim) if n_full > 0 else None
        )

        # Separate RoPE caches for sliding and full attention.
        rope_params = getattr(config, "rope_parameters", None) or {}
        sliding_rope_params = rope_params.get("sliding_attention", {})
        full_rope_params = rope_params.get("full_attention", {})

        sliding_theta = float(sliding_rope_params.get("rope_theta", 10_000.0))
        full_theta = float(full_rope_params.get("rope_theta", 10_000.0))

        max_pos = config.max_position_embeddings
        self.sliding_rope = RoPECache(head_dim, max_pos, sliding_theta)

        full_partial = full_rope_params.get("partial_rotary_factor", 1.0)
        if n_full > 0:
            self.full_rope = (
                ProportionalRoPECache(global_head_dim, max_pos, full_theta, full_partial)
                if full_partial < 1.0
                else RoPECache(global_head_dim, max_pos, full_theta)
            )
        else:
            self.full_rope = None

        # Per-layer embeddings (PLE).
        self.ple_dim = getattr(config, "hidden_size_per_layer_input", 0)
        if self.ple_dim:
            self.per_layer_model_projection = nn.Conv2d(
                hidden_size, n_layers * self.ple_dim, kernel_size=1, bias=False
            )
            self.per_layer_projection_norm = RMSNorm(self.ple_dim, eps=config.rms_norm_eps)
            self._per_layer_input_scale = 2.0**-0.5
            self._per_layer_projection_scale = hidden_size**-0.5

        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.lm_head = None

        self.emb_zero_point = nn.Parameter(torch.zeros([], dtype=torch.int8), requires_grad=False)
        self.emb_scale = nn.Parameter(torch.ones([], dtype=torch.float16), requires_grad=False)

        self._final_logit_softcap = getattr(config, "final_logit_softcapping", None)

    def _compute_per_layer_inputs(
        self,
        ple_embeddings: torch.Tensor,
        transformer_input: torch.Tensor,
    ) -> torch.Tensor:
        """Combine pre-gathered PLE token embeddings with the ctx_ple projection."""
        B, L, _, H = transformer_input.shape
        n_layers = len(self.model.layers)

        # ctx_ple: project transformer_input via Conv2d, then norm.
        xc = transformer_input.reshape(B * L, H, 1, 1)
        ctx_ple_conv = self.per_layer_model_projection(xc)  # (B*L, n_layers*ple_dim, 1, 1)
        scale = self._per_layer_projection_scale
        ctx_ple = ctx_ple_conv.reshape(B, L, 1, n_layers * self.ple_dim) * scale

        flat = ctx_ple.reshape(B * L * n_layers, 1, 1, self.ple_dim)
        ctx_ple_normed = self.per_layer_projection_norm(flat)
        ctx_ple_normed = ctx_ple_normed.reshape(B, L, 1, n_layers * self.ple_dim)

        return (ple_embeddings + ctx_ple_normed) * self._per_layer_input_scale

    def forward(
        self,
        transformer_input: torch.Tensor,
        position_ids: torch.IntTensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        key_cache_full: torch.Tensor,
        value_cache_full: torch.Tensor,
        embedding_table: torch.Tensor,
        ple_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.sliding_kv_cache.register_kv_cache(key_cache, value_cache)
        if self.full_kv_cache is not None:
            self.full_kv_cache.register_kv_cache(key_cache_full, value_cache_full)

        sliding_rope_cos, sliding_rope_sin = self.sliding_rope.gather_cos_sin(position_ids)
        full_rope_cos, full_rope_sin = (
            self.full_rope.gather_cos_sin(position_ids)
            if self.full_rope is not None
            else (sliding_rope_cos, sliding_rope_sin)
        )

        per_layer_inputs: torch.Tensor | None = None
        if self.ple_dim and ple_embeddings is not None:
            per_layer_inputs = self._compute_per_layer_inputs(ple_embeddings, transformer_input)

        B, L, _, _ = transformer_input.shape
        out = self.model(
            transformer_input,
            sliding_rope_cos,
            sliding_rope_sin,
            full_rope_cos,
            full_rope_sin,
            in_step,
            causal_mask,
            self.sliding_kv_cache,
            self.full_kv_cache,
            per_layer_inputs,
        )

        if self.prefill_mode:
            sc = self.sliding_kv_cache
            result = sc.k_cache[0, 0, 0, 0, 0] + sc.v_cache[0, 0, 0, 0, 0]
            if self.full_kv_cache is not None:
                fc = self.full_kv_cache
                result = result + fc.k_cache[0, 0, 0, 0, 0] + fc.v_cache[0, 0, 0, 0, 0]
            return result

        logits: torch.Tensor
        if self.lm_head is not None:
            logits = self.lm_head(out.transpose(-2, -3))
        else:
            if embedding_table.dtype == torch.int8:
                embedding_table = dequantize_per_tensor(
                    embedding_table, self.emb_scale, self.emb_zero_point, out.dtype
                )
            embedding_table = embedding_table.reshape(
                embedding_table.shape[1], embedding_table.shape[0], embedding_table.shape[2]
            )
            out_2d = out.transpose(-3, -1).reshape(B, 1, out.shape[-1], L)
            logits = (embedding_table @ out_2d).transpose(-2, -1)

        if self._final_logit_softcap:
            logits = logits / self._final_logit_softcap
            logits = torch.tanh(logits)
            logits = logits * self._final_logit_softcap

        return logits


class Gemma4ForCausalLMForiOS(BaseForCausalLMForiOS):
    _HF_MODEL_CLASS = HFGemma4ForCausalLM
    # Gemma4 unified stores text weights under "model.language_model.*";
    # after stripping that prefix, layer keys look like "layers.N.*".
    _hf_layer_key_pattern: str = r"layers\.(\d+)\."

    def _init_model(self, config) -> None:
        self.extend = Gemma4Extend(config)
        if self.extend.ple_dim:
            self.load_ple_embeddings = LoadPLEEmbeddings(config)
            self.gather_ple_embeddings = GatherEmbeddings()

    def set_prefill_mode(self, prefill_mode: bool) -> None:
        self.extend.prefill_mode = prefill_mode

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        key_cache_full: torch.Tensor,
        value_cache_full: torch.Tensor,
    ) -> torch.Tensor:
        embedding_table = self.load_embeddings.embedding_table
        token_embeddings = self.gather_embeddings(input_ids, embedding_table)

        ple_embeds: torch.Tensor | None = None
        if self.extend.ple_dim:
            ple_embeds = self.gather_ple_embeddings(
                input_ids, self.load_ple_embeddings.ple_embedding_table
            )

        return self.extend(
            token_embeddings,
            position_ids,
            in_step,
            causal_mask,
            key_cache,
            value_cache,
            key_cache_full,
            value_cache_full,
            embedding_table,
            ple_embeds,
        )

    def _mutate_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:  # noqa: C901
        # Normalize: after stripping "model.language_model." keys look like "layers.N.*"
        for key in list(state_dict.keys()):
            if not key.startswith("model.") and not key.startswith("lm_head."):
                state_dict[f"model.{key}"] = state_dict.pop(key)

        max_layer = -1
        for k in state_dict:
            parts = k.split(".")
            if not k.startswith("model.layers.") or len(parts) < 4:
                continue
            if parts[2].isdigit():
                max_layer = max(max_layer, int(parts[2]))

        if max_layer < 0:
            return

        num_kv_shared = getattr(self.config, "num_kv_shared_layers", 0)
        first_shared = self.config.num_hidden_layers - num_kv_shared

        for i in range(max_layer + 1):
            is_kv_shared = num_kv_shared > 0 and i >= first_shared
            prefix = f"model.layers.{i}.self_attn"

            # Reshape Conv2d: add (1, 1) spatial dims.
            projs = ["q_proj", "o_proj"]
            if not is_kv_shared:
                layer_types = getattr(self.config, "layer_types", None) or []
                layer_type = layer_types[i] if i < len(layer_types) else "sliding_attention"
                use_alt = layer_type != "sliding_attention" and getattr(
                    self.config, "attention_k_eq_v", False
                )
                projs.append("k_proj")
                if not use_alt:
                    projs.append("v_proj")
            for proj in projs:
                wk = f"{prefix}.{proj}.weight"
                if wk in state_dict:
                    state_dict[wk] = state_dict[wk].unsqueeze(-1).unsqueeze(-1)

            # Drop unused kv weights for kv-shared layers.
            if is_kv_shared:
                for extra in [
                    f"{prefix}.k_proj.weight",
                    f"{prefix}.v_proj.weight",
                    f"{prefix}.k_norm.weight",
                    f"{prefix}.v_norm.weight",
                ]:
                    state_dict.pop(extra, None)

            # MLP Conv2d reshape.
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                wk = f"model.layers.{i}.mlp.{proj}.weight"
                if wk in state_dict:
                    state_dict[wk] = state_dict[wk].unsqueeze(-1).unsqueeze(-1)

            # PLE per-layer Conv2d reshape.
            for proj in ["per_layer_input_gate", "per_layer_projection"]:
                wk = f"model.layers.{i}.{proj}.weight"
                if wk in state_dict:
                    state_dict[wk] = state_dict[wk].unsqueeze(-1).unsqueeze(-1)

        # PLE model-level projection Conv2d reshape.
        ple_proj_key = "model.per_layer_model_projection.weight"
        if ple_proj_key in state_dict:
            state_dict[ple_proj_key] = state_dict[ple_proj_key].unsqueeze(-1).unsqueeze(-1)

        # Handle token embeddings.
        embedding_table = state_dict.pop("model.embed_tokens.weight").unsqueeze(1)
        if not self.disable_embedding_quantization:
            embedding_table, scale, zero_point = quantize_per_tensor(
                embedding_table, nbits=8, symmetric=True
            )
        else:
            scale = torch.tensor(1.0, dtype=embedding_table.dtype)
            zero_point = torch.tensor(0, dtype=torch.int8)

        state_dict["load_embeddings.embedding_table"] = embedding_table
        state_dict["gather_embeddings.scale"] = scale
        state_dict["gather_embeddings.zero_point"] = zero_point
        state_dict["extend.emb_scale"] = scale
        state_dict["extend.emb_zero_point"] = zero_point

        # Handle PLE embedding table separately.
        ple_dim = getattr(self.config, "hidden_size_per_layer_input", 0)
        ple_emb_key = "model.embed_tokens_per_layer.weight"
        if ple_dim and ple_emb_key in state_dict:
            ple_table = state_dict.pop(ple_emb_key).unsqueeze(1)
            if not self.disable_embedding_quantization:
                # Bake embed_scale (ple_dim**0.5) into the quantization so that
                # gather_ple_embeddings recovers correctly-scaled values on dequantize.
                ple_table, ple_scale, ple_zp = quantize_per_tensor(
                    (ple_table.float() * float(ple_dim**0.5)).to(ple_table.dtype),
                    nbits=8,
                    symmetric=True,
                )
            else:
                ple_scale = torch.tensor(1.0, dtype=ple_table.dtype)
                ple_zp = torch.tensor(0, dtype=torch.int8)
            state_dict["load_ple_embeddings.ple_embedding_table"] = ple_table
            state_dict["gather_ple_embeddings.scale"] = ple_scale
            state_dict["gather_ple_embeddings.zero_point"] = ple_zp

        # Move model weights under "extend." prefix.
        # PLE projection modules live on Gemma4Extend directly (not on Gemma4TextModel),
        # so strip the "model." part and map to "extend." only.
        _ple_direct = ("model.per_layer_model_projection.", "model.per_layer_projection_norm.")
        new_state_dict: dict[str, torch.Tensor] = {}
        keys_to_pop: set[str] = set()
        for k in state_dict:
            if k.startswith("model.") and "gather_embeddings" not in k:
                if any(k.startswith(p) for p in _ple_direct):
                    new_key = "extend." + k[len("model.") :]
                else:
                    new_key = f"extend.{k}"
                new_state_dict[new_key] = state_dict[k]
                keys_to_pop.add(k)
        for k in keys_to_pop:
            state_dict.pop(k)
        state_dict.update(new_state_dict)

        if not self.config.tie_word_embeddings:
            if "lm_head.weight" in state_dict:
                state_dict["extend.lm_head.weight"] = state_dict.pop("lm_head.weight")
        else:
            state_dict.pop("lm_head.weight", None)

    def _mutate_shared_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Add 'model.' prefix to keys that lost it after stripping 'model.language_model.'."""
        for key in list(state_dict.keys()):
            if not key.startswith("model.") and not key.startswith("lm_head."):
                state_dict[f"model.{key}"] = state_dict.pop(key)
