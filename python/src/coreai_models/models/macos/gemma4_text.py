# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import os

import torch
import torch.nn as nn
from safetensors.torch import save_file
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4ForCausalLM as HFGemma4ForCausalLM,
)
from typing_extensions import Self, override

from coreai_models.export._constants import (
    KEY_CACHE_NAME,
    KEY_CACHE_SLIDING_NAME,
    VALUE_CACHE_NAME,
    VALUE_CACHE_SLIDING_NAME,
)
from coreai_models.models.base import BaseForCausalLM
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.quantization import dequantize_per_tensor, quantize_per_tensor
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.rope import RoPE
from coreai_models.primitives.macos.sdpa import SDPA
from coreai_models.primitives.macos.switch import SwitchGLU

# Fuse q_norm + k_norm into a single per-head qk_norm (same as gemma3_text.py)
USE_FUSED_KV = True


def _v_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Per-head RMSNorm without learned weights (Gemma4 v_norm has with_scale=False)."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _bounded_window_causal_mask(
    query_len: int,
    capacity,
    valid_len,
    device: torch.device,
) -> torch.Tensor:
    """Lower-right causal mask for a bounded sliding-window cache buffer.

    `key`/`value` from `KVCache.update_and_fetch_windowed` are always the full
    `capacity`-sized buffer (see that method's docstring for why), so ramp-up
    (before the window fills) leaves stale/unwritten slots at the front that
    must be masked out. Equivalent to the standard lower-right causal mask with
    logical key length `valid_len`, but `valid_len` is used only as a *value*
    here (in a comparison), never to *size* a tensor.
    """
    num_past = valid_len - query_len
    row_idx = torch.arange(query_len, device=device)
    col_idx = torch.arange(capacity, device=device)
    return row_idx.unsqueeze(-1) >= (col_idx.unsqueeze(0) - num_past)


class _ProportionalRoPE(nn.Module):
    """Proportional RoPE: frequencies use the full head_dim as denominator.

    coreai_torch RoPE(dims=N) uses N as the frequency denominator, but HF's
    proportional RoPE uses the full global_head_dim. Pass explicit freqs to fix.
    """

    def __init__(self, rope_dims: int, head_dim: int, base: float) -> None:
        super().__init__()
        self._rope = RoPE()
        with torch.device("cpu"):
            rotated = 1.0 / (
                base ** (torch.arange(0, rope_dims, 2, dtype=torch.float32) / head_dim)
            )
            # Pad with zeros for NoPE dims so freqs.shape == (head_dim//2,).
            # coreai_torch splits the head into two halves and needs cos/sin of that size.
            # Zero-freq dims get cos=1, sin=0 → identity (no rotation), matching HF.
            nope = torch.zeros(head_dim // 2 - rope_dims // 2, dtype=torch.float32)
            self._freqs = torch.cat([rotated, nope])

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        return self._rope(x, position_ids=position_ids, freqs=self._freqs)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = nn.functional.gelu(self.gate_proj(x), approximate="tanh")
        return self.down_proj(gate * self.up_proj(x))


class GeGLU(nn.Module):
    """GELU-gated activation for Gemma4 MoE (hidden_activation=gelu_pytorch_tanh)."""

    def forward(self, up: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return nn.functional.gelu(gate, approximate="tanh") * up


class MoERouter(nn.Module):
    """Gemma4 MoE router: v_norm → scale → linear → softmax → top-k → per-expert scale.

    Uses _v_norm (no learnable weight) matching HF's Gemma4RMSNorm(with_scale=False),
    followed by a learnable scale parameter and hidden_size**-0.5 normalisation.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.scalar_root_size = config.hidden_size**-0.5
        self.top_k = getattr(config, "top_k_experts", 1) or 1
        self._norm_eps = config.rms_norm_eps
        self.proj = nn.Linear(config.hidden_size, config.num_experts or 0, bias=False)
        self.scale = nn.Parameter(torch.ones(config.hidden_size))
        self.per_expert_scale = nn.Parameter(torch.ones(config.num_experts or 0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = _v_norm(x, self._norm_eps) * self.scale * self.scalar_root_size
        scores = self.proj(h)
        probs = nn.functional.softmax(scores, dim=-1)
        top_k_weights, top_k_index = torch.topk(probs, k=self.top_k, dim=-1, largest=True)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]
        return probs, top_k_weights, top_k_index


class Attention(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        layer_types: list[str] = getattr(config, "layer_types", None) or []
        layer_type = layer_types[layer_idx] if layer_idx < len(layer_types) else "sliding_attention"
        self.is_sliding = layer_type == "sliding_attention"

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads

        # Full attention with attention_k_eq_v uses fewer KV heads and v=k.
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

        # KV sharing: the last num_kv_shared_layers layers reuse K/V from a donor.
        num_kv_shared = getattr(config, "num_kv_shared_layers", 0)
        first_shared = config.num_hidden_layers - num_kv_shared
        self.is_kv_shared = num_kv_shared > 0 and layer_idx >= first_shared

        # donor_cache_idx: type-local index into the same cache (sliding or full) as this layer.
        self.donor_cache_idx: int | None = None
        if self.is_kv_shared:
            same_type_before_shared = [
                i for i, t in enumerate(layer_types[:first_shared]) if t == layer_type
            ]
            if same_type_before_shared:
                donor_global = same_type_before_shared[-1]
                # Convert to type-local index within the appropriate cache
                same_type_all = [i for i, t in enumerate(layer_types) if t == layer_type]
                self.donor_cache_idx = same_type_all.index(donor_global)

        # Type-local cache slot — each layer type has its own (n_type_layers, ...) cache.
        if layer_types:
            same_type = [i for i, t in enumerate(layer_types) if t == layer_type]
            self.cache_layer_idx = (
                same_type.index(layer_idx) if layer_idx in same_type else layer_idx
            )
        else:
            self.cache_layer_idx = layer_idx

        window_size = config.sliding_window if self.is_sliding else 0
        # Gemma4 uses scaling=1.0; q/k norms absorb the 1/sqrt(head_dim) factor.
        self.sdpa = SDPA(scale=1.0, window_size=window_size, is_causal=True)

        # RoPE — proportional for full attention (partial_rotary_factor < 1).
        rope_params = (getattr(config, "rope_parameters", None) or {}).get(layer_type, {})
        rope_base = rope_params.get("rope_theta", 10_000.0)
        partial_factor = rope_params.get("partial_rotary_factor", 1.0)
        rope_dims = int(partial_factor * head_dim // 2) * 2 if partial_factor < 1.0 else None
        # If all dims are NoPE (tiny models with very small head_dim), skip RoPE.
        self.use_rope = rope_dims is None or rope_dims >= 2
        if self.use_rope:
            rope_type = rope_params.get("rope_type", "default")
            if rope_type == "proportional" and rope_dims is not None:
                # HF uses head_dim (global_head_dim) as the frequency denominator, not rope_dims.
                # coreai_torch RoPE(dims=N) uses N as denominator, so pass explicit freqs instead.
                self.rope = _ProportionalRoPE(rope_dims, head_dim, float(rope_base))
            else:
                self.rope = RoPE(base=float(rope_base), dims=rope_dims if rope_dims else None)

        # Projections:
        #   - regular: fused qkv_proj (checkpoint has q/k/v split, fused in _mutate_state_dict)
        #   - alt-attn (v=k): separate q_proj + k_proj, no v_proj
        #   - kv-shared: only q_proj
        if not self.is_kv_shared and not self.use_alt_attn:
            self.qkv_proj = nn.Linear(dim, (n_heads + 2 * n_kv_heads) * head_dim, bias=False)
        elif self.use_alt_attn:
            self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
            self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        else:
            self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        # Per-head norms — fused into qk_norm when USE_FUSED_KV.
        if USE_FUSED_KV and not self.is_kv_shared:
            self.qk_norm = RMSNorm(head_dim, eps=config.rms_norm_eps, n_heads=n_heads + n_kv_heads)
        else:
            self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps, n_heads=n_heads)
            if not self.is_kv_shared:
                self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps, n_heads=n_kv_heads)

        self._rms_norm_eps = config.rms_norm_eps

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None,
        cache_full: KVCache | None = None,
    ) -> torch.Tensor:
        batch_size, query_len, _ = x.shape
        n_heads, n_kv_heads = self.n_heads, self.n_kv_heads
        head_dim = self.head_dim

        seq_len = position_ids.shape[-1]
        torch._check_is_size(query_len)
        torch._check_is_size(seq_len)
        offset = seq_len - query_len
        torch._check_is_size(offset)
        rope_positions = position_ids.narrow(-1, offset, query_len)

        # Route to the correct type-local cache.
        active_cache = cache if self.is_sliding else cache_full
        attn_mask = None

        if self.is_kv_shared:
            query = (
                self.q_proj(x).reshape(batch_size, query_len, n_heads, head_dim).permute(0, 2, 1, 3)
            )
            query = self.q_norm(query)
            if self.use_rope:
                query = self.rope(query, position_ids=rope_positions)

            donor = self.donor_cache_idx
            if active_cache is not None and donor is not None:
                if self.is_sliding:
                    # The donor is a bounded sliding-window cache: read the full
                    # static-capacity buffer (see update_and_fetch_windowed's
                    # docstring) and mask out not-yet-written slots explicitly.
                    donor_valid = active_cache.clamped_len(seq_len)
                    key = active_cache._k_cache.narrow(0, donor, 1).squeeze(0)
                    value = active_cache._v_cache.narrow(0, donor, 1).squeeze(0)
                    attn_mask = _bounded_window_causal_mask(
                        query_len, active_cache.capacity(), donor_valid, x.device
                    )
                else:
                    donor_len = active_cache.clamped_len(seq_len)
                    key = (
                        active_cache._k_cache.narrow(0, donor, 1)
                        .narrow(-2, 0, donor_len)
                        .squeeze(0)
                    )
                    value = (
                        active_cache._v_cache.narrow(0, donor, 1)
                        .narrow(-2, 0, donor_len)
                        .squeeze(0)
                    )
            else:
                key = torch.zeros(
                    batch_size, n_kv_heads, seq_len, head_dim, device=x.device, dtype=x.dtype
                )
                value = key
        else:
            if self.use_alt_attn:
                q = (
                    self.q_proj(x)
                    .reshape(batch_size, query_len, n_heads, head_dim)
                    .permute(0, 2, 1, 3)
                )
                k = (
                    self.k_proj(x)
                    .reshape(batch_size, query_len, n_kv_heads, head_dim)
                    .permute(0, 2, 1, 3)
                )
            else:
                qkv = (
                    self.qkv_proj(x)
                    .reshape(batch_size, query_len, n_heads + 2 * n_kv_heads, head_dim)
                    .permute(0, 2, 1, 3)
                )
                q = qkv.narrow(1, 0, n_heads)
                k = qkv.narrow(1, n_heads, n_kv_heads)

            if USE_FUSED_KV:
                qk = torch.cat([q, k], dim=1)
                qk = self.qk_norm(qk)
                query = qk.narrow(1, 0, n_heads)
                key = qk.narrow(1, n_heads, n_kv_heads)
            else:
                query = self.q_norm(q)
                key = self.k_norm(k)

            query = self.rope(query, position_ids=rope_positions) if self.use_rope else query
            key = self.rope(key, position_ids=rope_positions) if self.use_rope else key

            if self.use_alt_attn:
                value = _v_norm(k, self._rms_norm_eps)
            else:
                value = _v_norm(qkv.narrow(1, n_heads + n_kv_heads, n_kv_heads), self._rms_norm_eps)

            if active_cache is not None:
                if self.is_sliding:
                    key, value = active_cache.update_and_fetch_windowed(
                        self.cache_layer_idx,
                        offset,
                        key,
                        value,
                        query_len=query_len,
                    )
                    valid_len = active_cache.clamped_len(offset + query_len)
                    attn_mask = _bounded_window_causal_mask(
                        query_len, active_cache.capacity(), valid_len, x.device
                    )
                else:
                    key, value = active_cache.update_and_fetch(
                        self.cache_layer_idx,
                        offset,
                        key,
                        value,
                        seq_len=seq_len,
                        query_len=query_len,
                    )

        output = (
            self.sdpa(query=query, key=key, value=value, attn_mask=attn_mask)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, query_len, n_heads * head_dim)
        )
        return self.o_proj(output)


class TransformerBlock(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.self_attn = Attention(config=config, layer_idx=layer_idx)

        first_shared = config.num_hidden_layers - getattr(config, "num_kv_shared_layers", 0)
        is_shared = getattr(config, "num_kv_shared_layers", 0) > 0 and layer_idx >= first_shared
        double_wide = getattr(config, "use_double_wide_mlp", False) and is_shared
        self.mlp = MLP(hidden_size, config.intermediate_size * (2 if double_wide else 1))

        eps = config.rms_norm_eps
        self.input_layernorm = RMSNorm(hidden_size, eps=eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=eps)
        self.pre_feedforward_layernorm = RMSNorm(hidden_size, eps=eps)

        self.enable_moe_block = getattr(config, "enable_moe_block", False)
        self.post_feedforward_layernorm = RMSNorm(hidden_size, eps=eps)

        if self.enable_moe_block:
            self.pre_feedforward_layernorm_2 = RMSNorm(hidden_size, eps=eps)
            self.post_feedforward_layernorm_1 = RMSNorm(hidden_size, eps=eps)
            self.post_feedforward_layernorm_2 = RMSNorm(hidden_size, eps=eps)
            self.router = MoERouter(config)
            self.switch_mlp = SwitchGLU(
                hidden_size,
                config.moe_intermediate_size or 0,
                config.num_experts or 0,
                activation=GeGLU(),
            )

        self.ple_dim = getattr(config, "hidden_size_per_layer_input", 0)
        if self.ple_dim:
            self.per_layer_input_gate = nn.Linear(hidden_size, self.ple_dim, bias=False)
            self.per_layer_projection = nn.Linear(self.ple_dim, hidden_size, bias=False)
            self.post_per_layer_input_norm = RMSNorm(hidden_size, eps=eps)

        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None,
        cache_full: KVCache | None = None,
        per_layer_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r = self.self_attn(self.input_layernorm(x), position_ids, cache, cache_full)
        h = x + self.post_attention_layernorm(r)

        mlp_out = self.mlp(self.pre_feedforward_layernorm(h))

        if self.enable_moe_block:
            mlp_out_1 = self.post_feedforward_layernorm_1(mlp_out)
            _, top_k_weights, top_k_index = self.router(h)
            normed = self.pre_feedforward_layernorm_2(h)
            moe_out = self.switch_mlp(normed, top_k_index.to(torch.uint16))
            moe_out = (moe_out * top_k_weights.unsqueeze(-1)).sum(dim=-2).to(h.dtype)
            moe_out = self.post_feedforward_layernorm_2(moe_out)
            mlp_out = mlp_out_1 + moe_out

        h = h + self.post_feedforward_layernorm(mlp_out)

        if self.ple_dim and per_layer_input is not None:
            gate = nn.functional.gelu(self.per_layer_input_gate(h), approximate="tanh")
            ple = self.per_layer_projection(gate * per_layer_input)
            h = h + self.post_per_layer_input_norm(ple)

        return h * self.layer_scalar


class Gemma4TextModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            hidden_size,
            padding_idx=config.pad_token_id or 0,
        )
        self.embed_scale = hidden_size**0.5
        self.layers = nn.ModuleList(
            [TransformerBlock(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

        self.ple_dim = getattr(config, "hidden_size_per_layer_input", 0)
        if self.ple_dim:
            n_layers = config.num_hidden_layers
            self.per_layer_model_projection = nn.Linear(
                hidden_size, n_layers * self.ple_dim, bias=False
            )
            self.per_layer_projection_norm = RMSNorm(self.ple_dim, eps=config.rms_norm_eps)
            self._per_layer_input_scale = 2.0**-0.5
            self._per_layer_projection_scale = hidden_size**-0.5
        self.ple_scale: torch.Tensor | None = None
        self.ple_zp: torch.Tensor | None = None

    def _compute_per_layer_inputs(
        self, ple_embeddings: torch.Tensor, inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        n_layers = len(self.layers)
        tok_ple = ple_embeddings.reshape(*inputs_embeds.shape[:-1], n_layers, self.ple_dim)
        ctx_ple = (
            self.per_layer_model_projection(inputs_embeds) * self._per_layer_projection_scale
        ).reshape(*inputs_embeds.shape[:-1], n_layers, self.ple_dim)
        ctx_ple = self.per_layer_projection_norm(ctx_ple)
        return (tok_ple + ctx_ple) * self._per_layer_input_scale

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
        cache_full: KVCache | None = None,
        ple_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.embed_tokens(input_ids) * self.embed_scale

        per_layer_inputs: torch.Tensor | None = None
        if self.ple_dim and ple_embeddings is not None:
            ple_dequant: torch.Tensor = ple_embeddings
            if ple_dequant.dtype == torch.int8:
                ple_dequant = dequantize_per_tensor(
                    ple_dequant, self.ple_scale, self.ple_zp, h.dtype
                )
            per_layer_inputs = self._compute_per_layer_inputs(ple_dequant, h)

        for i, layer in enumerate(self.layers):
            ple = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            h = layer(h, position_ids, cache, cache_full, ple)

        return self.norm(h)


class Gemma4ForCausalLM(BaseForCausalLM):
    """Gemma 4 text-only causal LM.

    Uses two separate KV caches to accommodate the different head dimensions across
    attention types (path A):
      - k_cache / v_cache: sliding-attention layers
        shape (n_sliding_layers, 1, num_key_value_heads, seq_len, head_dim)
      - k_cache_full / v_cache_full: full-attention layers
        shape (n_full_layers, 1, num_kv_heads_full, seq_len, global_head_dim)

    The Swift inference engine will need 4-state support to use these.

    The Gemma4 unified checkpoint stores text weights under ``model.language_model.*``.
    """

    _HF_MODEL_CLASS = HFGemma4ForCausalLM
    # After stripping "model.language_model." layer keys look like "layers.N.*"
    _hf_layer_key_pattern: str = r"layers\.(\d+)\."
    # Full state-name tuple, in forward()'s state-arg order (k_cache, v_cache,
    # k_cache_full, v_cache_full). The Swift runner expects keyCache/valueCache
    # to name the growable full-attention cache, not the bounded sliding one,
    # so the sliding pair gets the distinct slidingKeyCache/slidingValueCache
    # names instead of the generic (KEY_CACHE_NAME, VALUE_CACHE_NAME) default.
    _state_names = (
        KEY_CACHE_SLIDING_NAME,
        VALUE_CACHE_SLIDING_NAME,
        KEY_CACHE_NAME,
        VALUE_CACHE_NAME,
    )

    @classmethod
    def _build_sliding_kv_cache_tensors(
        cls, config, seq_len: int, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (k_cache, v_cache) for the sliding-attention layers only.

        The generic KVCache.create_cache_tensors sizes the layer dimension by
        config.num_hidden_layers (every layer); Gemma4 needs it sized by the
        sliding-attention layer count since full-attention layers live in the
        separate cache built by _build_full_kv_cache_tensors.
        """
        layer_types = getattr(config, "layer_types", None) or []
        n_sliding = sum(1 for t in layer_types if t == "sliding_attention")
        n_kv_heads = config.num_key_value_heads
        head_dim = (
            config.head_dim
            if getattr(config, "head_dim", None) is not None
            else config.hidden_size // config.num_attention_heads
        )
        k = torch.zeros(n_sliding, 1, n_kv_heads, seq_len, head_dim, dtype=dtype)
        return k, k.clone()

    @classmethod
    def _build_full_kv_cache_tensors(
        cls, config, seq_len: int, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (k_cache_full, v_cache_full) for the full-attention layers."""
        layer_types = getattr(config, "layer_types", None) or []
        n_full = sum(1 for t in layer_types if t == "full_attention")
        use_alt = getattr(config, "attention_k_eq_v", False)
        n_kv_full = (
            getattr(config, "num_global_key_value_heads", None) or config.num_key_value_heads
            if use_alt
            else config.num_key_value_heads
        )
        hd_full = getattr(config, "global_head_dim", None) or config.head_dim
        k = torch.zeros(n_full, 1, n_kv_full, seq_len, hd_full, dtype=dtype)
        return k, k.clone()

    @override
    def _init_model(self, config) -> None:
        self.model = Gemma4TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._final_logit_softcap = getattr(config, "final_logit_softcapping", None)

    @BaseForCausalLM.cast_logits_bfloat16_to_float16
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_cache_full: torch.Tensor | None = None,
        v_cache_full: torch.Tensor | None = None,
        ple_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sliding_cache = KVCache(k_cache, v_cache)
        full_cache: KVCache | None = None
        if k_cache_full is not None and v_cache_full is not None:
            full_cache = KVCache(k_cache_full, v_cache_full)
        out = self.model(input_ids, position_ids, sliding_cache, full_cache, ple_embeddings)
        logits = self.lm_head(out)
        if self._final_logit_softcap:
            logits = logits / self._final_logit_softcap
            logits = torch.tanh(logits)
            logits = logits * self._final_logit_softcap
        return logits

    def _pop_ple_weight(self: Self, state_dict: dict[str, torch.Tensor]) -> None:
        """Pop the externalized Per-Layer Embeddings (PLE) table out of the state dict."""
        ple_dim = getattr(self.config, "hidden_size_per_layer_input", 0)
        if not ple_dim:
            return
        ple_key = "model.embed_tokens_per_layer.weight"
        if ple_key in state_dict:
            ple_weight = state_dict.pop(ple_key)
            num_layers = self.config.num_hidden_layers
            expected_dim = num_layers * ple_dim
            if ple_weight.shape[1] > expected_dim:
                ple_weight = ple_weight[:, :expected_dim].contiguous()
            self._ple_weight = ple_weight

            ple_embed_scale = ple_dim**0.5
            _, ple_scale, ple_zp = quantize_per_tensor(
                ple_weight.float() * ple_embed_scale, nbits=8, symmetric=True
            )
            self._ple_scale_pending = ple_scale.to(torch.bfloat16)
            self._ple_zp_pending = ple_zp

        # Truncate per_layer_model_projection.weight similarly when num_layers is overridden
        # (smoke tests use a small num_hidden_layers, but the HF checkpoint's projection weight
        # is still sized for the full model).
        proj_key = "model.per_layer_model_projection.weight"
        if proj_key in state_dict:
            full_proj = state_dict[proj_key]
            expected_out = self.config.num_hidden_layers * ple_dim
            if full_proj.shape[0] > expected_out:
                state_dict[proj_key] = full_proj[:expected_out, :]

    @override
    def _mutate_shared_dict(self: Self, state_dict: dict[str, torch.Tensor]) -> None:
        """Add 'model.' prefix to keys left bare after stripping 'model.language_model.'."""
        for key in list(state_dict.keys()):
            if not key.startswith("model.") and not key.startswith("lm_head."):
                state_dict[f"model.{key}"] = state_dict.pop(key)
        emb_key = "model.embed_tokens.weight"
        if (
            getattr(self.config, "tie_word_embeddings", False)
            and emb_key in state_dict
            and "lm_head.weight" not in state_dict
        ):
            state_dict["lm_head.weight"] = state_dict[emb_key].clone()
        self._pop_ple_weight(state_dict)

    @override
    def _mutate_state_dict(self: Self, state_dict: dict[str, torch.Tensor]) -> None:
        # Normalize: after stripping "model.language_model." keys look like "layers.N.*"
        for key in list(state_dict.keys()):
            if not key.startswith("model.") and not key.startswith("lm_head."):
                state_dict[f"model.{key}"] = state_dict.pop(key)

        # For tied embeddings, ensure lm_head.weight exists before layer mutations.
        emb_key = "model.embed_tokens.weight"
        if (
            getattr(self.config, "tie_word_embeddings", False)
            and emb_key in state_dict
            and "lm_head.weight" not in state_dict
        ):
            state_dict["lm_head.weight"] = state_dict[emb_key].clone()

        self._pop_ple_weight(state_dict)

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
            q_key = f"{prefix}.q_proj.weight"
            k_key = f"{prefix}.k_proj.weight"
            v_key = f"{prefix}.v_proj.weight"

            if is_kv_shared:
                # Checkpoint may include k/v weights for kv-shared layers (HF ignores them).
                for extra in [k_key, v_key, f"{prefix}.k_norm.weight", f"{prefix}.v_norm.weight"]:
                    state_dict.pop(extra, None)
            elif q_key in state_dict and k_key in state_dict and v_key in state_dict:
                # Fuse q/k/v → qkv_proj; alt-attn layers have no v_proj so skip.
                state_dict[f"{prefix}.qkv_proj.weight"] = torch.cat(
                    [state_dict.pop(q_key), state_dict.pop(k_key), state_dict.pop(v_key)], dim=0
                )

            if not USE_FUSED_KV:
                continue

            q_norm_key = f"{prefix}.q_norm.weight"
            k_norm_key = f"{prefix}.k_norm.weight"

            # kv-shared: only q_norm remains — expand (head_dim,) → (n_heads, 1, head_dim)
            if q_norm_key in state_dict and k_norm_key not in state_dict:
                o_key = f"{prefix}.o_proj.weight"
                if o_key in state_dict:
                    q_norm_w = state_dict.pop(q_norm_key)
                    n_heads = state_dict[o_key].shape[1] // q_norm_w.shape[0]
                    hd = q_norm_w.shape[0]
                    state_dict[q_norm_key] = (
                        q_norm_w.unsqueeze(0).unsqueeze(0).expand(n_heads, 1, hd).contiguous()
                    )
                continue

            if q_norm_key not in state_dict or k_norm_key not in state_dict:
                continue

            q_norm_w = state_dict.pop(q_norm_key)  # (head_dim,)
            k_norm_w = state_dict.pop(k_norm_key)  # (head_dim,)
            head_dim = q_norm_w.shape[0]

            o_key = f"{prefix}.o_proj.weight"
            qkv_key = f"{prefix}.qkv_proj.weight"

            if o_key not in state_dict:
                state_dict[q_norm_key] = q_norm_w
                state_dict[k_norm_key] = k_norm_w
                continue

            n_heads = state_dict[o_key].shape[1] // head_dim

            if qkv_key in state_dict:
                n_kv_heads = (state_dict[qkv_key].shape[0] - n_heads * head_dim) // (2 * head_dim)
            elif k_key in state_dict:
                n_kv_heads = state_dict[k_key].shape[0] // head_dim
            else:
                state_dict[q_norm_key] = q_norm_w
                state_dict[k_norm_key] = k_norm_w
                continue

            q_rep = q_norm_w.unsqueeze(0).unsqueeze(0).expand(n_heads, 1, head_dim)
            k_rep = k_norm_w.unsqueeze(0).unsqueeze(0).expand(n_kv_heads, 1, head_dim)
            state_dict[f"{prefix}.qk_norm.weight"] = torch.cat([q_rep, k_rep], dim=0).contiguous()

        # HF stores MoE experts as:
        #   experts.gate_up_proj  (E, 2H, D) — gate and up concatenated
        #   experts.down_proj     (E, D, H)
        # SwitchGLU expects:
        #   switch_mlp.{gate,up,down}_proj.weight  (1, E, out_dim, in_dim)
        for i in range(max_layer + 1):
            gate_up_key = f"model.layers.{i}.experts.gate_up_proj"
            down_key = f"model.layers.{i}.experts.down_proj"
            if gate_up_key not in state_dict:
                continue
            gate_up = state_dict.pop(gate_up_key)  # (E, 2H, D)
            h_dim = gate_up.shape[1] // 2
            gate = gate_up[:, :h_dim, :]  # (E, H, D)
            up = gate_up[:, h_dim:, :]  # (E, H, D)
            state_dict[f"model.layers.{i}.switch_mlp.gate_proj.weight"] = gate.unsqueeze(0)
            state_dict[f"model.layers.{i}.switch_mlp.up_proj.weight"] = up.unsqueeze(0)
            if down_key in state_dict:
                down = state_dict.pop(down_key)  # (E, D, H)
                state_dict[f"model.layers.{i}.switch_mlp.down_proj.weight"] = down.unsqueeze(0)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        super().load_state_dict(state_dict, strict=strict, assign=assign)
        if hasattr(self, "_ple_scale_pending"):
            self.model.ple_scale = self._ple_scale_pending
            self.model.ple_zp = self._ple_zp_pending
            del self._ple_scale_pending
            del self._ple_zp_pending

    def dump_ple_embedding(self, output_path: str, model_name: str) -> str:
        """Dump the externalized PLE embedding table as a quantized INT8 safetensors file."""
        ple_weight = self._ple_weight
        ple_dim = self.config.hidden_size_per_layer_input
        embed_scale = str(ple_dim**0.5)
        ple_scaled = ple_weight.float() * float(embed_scale)
        ple_q, ple_scale, ple_zp = quantize_per_tensor(ple_scaled, nbits=8, symmetric=True)
        os.makedirs(output_path, exist_ok=True)
        ple_path = os.path.join(output_path, f"{model_name}_ple.safetensors")
        save_file(
            {"embed_tokens_per_layer": ple_q.contiguous()},
            ple_path,
            metadata={
                "embed_scale": embed_scale,
                "ple_scale": str(float(ple_scale)),
                "ple_zero_point": str(int(ple_zp)),
                "dtype": "SI8",
            },
        )
        return ple_path


class Gemma4TextModelEmbeddings(Gemma4TextModel):
    """Gemma4 text backbone whose forward starts from pre-computed ``inputs_embeds``.

    Identical to :class:`Gemma4TextModel` except that the in-graph
    ``embed_tokens`` lookup is skipped: the caller supplies already-embedded
    (and already ``embed_scale``-multiplied) hidden states. This is the split
    that makes VLM export work -- image soft-tokens are scatter-merged into the
    text embedding stream on the host before prefill, exactly as Qwen3-VL does
    (see ``coreai_models.vlm.gemma4``). The ``embed_tokens`` table itself is
    exported as the separate ``embed.aimodel`` component.

    ``ple_embeddings`` is still fed as an INT8 graph input (PLE is externalized
    the same way as the text-only path); only the primary token lookup moves out
    of the graph.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        # The token-embedding table is exported separately as embed.aimodel and
        # is not part of this inputs_embeds graph. Drop the inherited module so
        # its (unfed) weight isn't left dangling on the meta device after load.
        del self.embed_tokens

    def forward(  # type: ignore[override]
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
        cache_full: KVCache | None = None,
        ple_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # NOTE: inputs_embeds is expected to already include the embed_scale
        # factor (applied in the exported embed.aimodel), matching HF's
        # Gemma3nTextScaledWordEmbedding. Image soft-tokens are merged in at
        # their own (projected) scale, mirroring HF masked_scatter.
        # inputs_embeds arrives as f16 (host/embed.aimodel/vision.aimodel dtype);
        # Gemma 4 computes in bf16 (PLE scales + weights are bf16), so cast up
        # front. Keeping the graph *input* f16 matches the Swift Float16 path and
        # avoids a host-side bf16 tensor, exactly like the LLaVA/Qwen3-VL runners.
        h = inputs_embeds.to(self.norm.weight.dtype)

        per_layer_inputs: torch.Tensor | None = None
        if self.ple_dim and ple_embeddings is not None:
            ple_dequant: torch.Tensor = ple_embeddings
            if ple_dequant.dtype == torch.int8:
                ple_dequant = dequantize_per_tensor(
                    ple_dequant, self.ple_scale, self.ple_zp, h.dtype
                )
            per_layer_inputs = self._compute_per_layer_inputs(ple_dequant, h)

        for i, layer in enumerate(self.layers):
            ple = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            h = layer(h, position_ids, cache, cache_full, ple)

        return self.norm(h)


class Gemma4ForCausalLMEmbeddings(Gemma4ForCausalLM):
    """Gemma4 causal LM whose decoder graph takes ``inputs_embeds`` (VLM path).

    The exported ``main`` graph of a Gemma4 VLM bundle differs from the
    text-only Gemma4 decoder only in its first input: ``inputs_embeds`` (f16,
    ``[1, seq, hidden]``) instead of ``input_ids``. Everything else -- the dual
    sliding/full KV cache, the externalized INT8 ``ple_embeddings`` input, and
    the final-logit softcap -- is identical, so this reuses
    :class:`Gemma4ForCausalLM`'s cache builders, PLE plumbing and state-dict
    mutations wholesale.

    The token-embedding table is dropped from this graph (it lives in the
    separate ``embed.aimodel``); ``_mutate_state_dict`` therefore deletes
    ``model.embed_tokens.weight`` unless it is tied to ``lm_head``.
    """

    @override
    def _init_model(self, config) -> None:
        self.model = Gemma4TextModelEmbeddings(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._final_logit_softcap = getattr(config, "final_logit_softcapping", None)

    @BaseForCausalLM.cast_logits_bfloat16_to_float16
    def forward(  # type: ignore[override]
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.IntTensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_cache_full: torch.Tensor | None = None,
        v_cache_full: torch.Tensor | None = None,
        ple_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sliding_cache = KVCache(k_cache, v_cache)
        full_cache: KVCache | None = None
        if k_cache_full is not None and v_cache_full is not None:
            full_cache = KVCache(k_cache_full, v_cache_full)
        out = self.model(inputs_embeds, position_ids, sliding_cache, full_cache, ple_embeddings)
        logits = self.lm_head(out)
        if self._final_logit_softcap:
            logits = logits / self._final_logit_softcap
            logits = torch.tanh(logits)
            logits = logits * self._final_logit_softcap
        return logits

    @override
    def _mutate_state_dict(self: Self, state_dict: dict[str, torch.Tensor]) -> None:
        # Reuse the parent's fusion / PLE-pop logic, then drop embed_tokens: the
        # token lookup is exported as embed.aimodel, not baked into this graph.
        super()._mutate_state_dict(state_dict)
        emb_key = "model.embed_tokens.weight"
        if (
            getattr(self.config, "tie_word_embeddings", False)
            and emb_key in state_dict
            and "lm_head.weight" not in state_dict
        ):
            state_dict["lm_head.weight"] = state_dict[emb_key].clone()
        state_dict.pop(emb_key, None)
