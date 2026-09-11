# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Muse Glimmer DFlash drafter -- two-phase speculative decoding prototype.

Architecture (same backbone as the standard ring drafter):
- 5 transformer layers, all sliding window (2048)
- 32Q / 8KV heads, head_dim 128
- Separate q_norm / k_norm (not shared like the target)
- No gated attention, no sandwich norms
- Shares embed_tokens and lm_head with the target model
- Ring buffer KV cache with explicit attention mask
"""

from types import SimpleNamespace

import torch
import torch.nn as nn

from coreai_models.primitives.macos.cache import RingKVCache
from coreai_models.primitives.macos.mlp import MLP
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.rope import RoPE
from coreai_models.primitives.macos.sdpa import SDPA

# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------


def dflash_draft_mask(
    query_len: int,
    capacity: int,
    offset: int,
    device: torch.device,
) -> torch.Tensor:
    """Bidirectional attention mask for the draft phase."""
    slot = torch.arange(capacity, device=device)
    last_pos = offset + query_len - 1

    # Reconstruct absolute position stored in each ring slot.
    # Same derivation as ring_window_causal_mask but without the causal check.
    r = last_pos % capacity
    diff = r - slot  # range (-capacity, capacity)
    ring_back = torch.where(diff >= 0, diff, diff + capacity)
    k_pos = last_pos - ring_back  # (capacity,)

    total_valid = offset + query_len
    valid = (k_pos >= 0) & (k_pos < total_valid)
    return valid.unsqueeze(0).expand(query_len, capacity)


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    """Attention with separate inject_kv and draft paths."""

    def __init__(self, config: SimpleNamespace, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.window_size = config.sliding_window

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.head_dim = head_dim = config.head_dim

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)

        # Non-causal; explicit mask required
        self.sdpa = SDPA(is_causal=False)

        rope_theta = (
            config.rope_parameters.get("rope_theta", 500000.0)
            if isinstance(config.rope_parameters, dict)
            else 500000.0
        )
        self.rope = RoPE()
        with torch.device("cpu"):
            self._rope_freqs = 1.0 / (
                rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
            )

    # ---- Phase 1: KV injection (no attention) ----------------------------

    def inject_kv(
        self,
        features: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: RingKVCache,
    ) -> None:
        """Project features through Wk/Wv and write to ring cache."""
        batch_size, n_features, _ = features.shape
        n_kv_heads = self.n_kv_heads

        # K projection + norm + RoPE
        key = self.k_norm(
            self.k_proj(features)
            .reshape(batch_size, n_features, n_kv_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        freqs = self._rope_freqs.to(device=features.device)
        key = self.rope(key, position_ids=position_ids, freqs=freqs)

        # V projection (no norm, no RoPE)
        value = (
            self.v_proj(features)
            .reshape(batch_size, n_features, n_kv_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        # Ring offset = first absolute position being injected
        offset = int(position_ids[0, 0])
        cache.update_and_fetch(self.layer_idx, offset=offset, k=key, v=value, query_len=n_features)

    # ---- Phase 2: Draft forward (full attention) -------------------------

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: RingKVCache,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Attention forward for draft tokens."""
        batch_size, query_len, _ = x.shape
        n_heads, n_kv_heads = self.n_heads, self.n_kv_heads

        query = self.q_norm(
            self.q_proj(x)
            .reshape(batch_size, query_len, n_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        key = self.k_norm(
            self.k_proj(x)
            .reshape(batch_size, query_len, n_kv_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        value = (
            self.v_proj(x)
            .reshape(batch_size, query_len, n_kv_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        freqs = self._rope_freqs.to(device=query.device)
        seq_len = position_ids.shape[-1]
        offset = seq_len - query_len
        rope_positions = position_ids.narrow(-1, offset, query_len)
        query = self.rope(query, position_ids=rope_positions, freqs=freqs)
        key = self.rope(key, position_ids=rope_positions, freqs=freqs)

        # Write draft KV to cache, fetch full cache (injected + draft)
        key, value = cache.update_and_fetch(self.layer_idx, offset, key, value, query_len=query_len)

        attn_output = (
            self.sdpa(query=query, key=key, value=value, attn_mask=attn_mask)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, query_len, self.n_heads * self.head_dim)
        )
        return self.o_proj(attn_output)


class TransformerBlock(nn.Module):
    def __init__(self, config: SimpleNamespace, layer_idx: int) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.self_attn = Attention(config, layer_idx=layer_idx)
        self.mlp = MLP(hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

    def inject_kv(
        self,
        features: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: RingKVCache,
    ) -> None:
        """KV injection: project features and write to cache."""
        self.self_attn.inject_kv(features, position_ids, cache)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: RingKVCache,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Draft forward: attention + MLP."""
        r = self.self_attn(self.input_layernorm(x), position_ids, cache, attn_mask)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class DFlashDrafterModel(nn.Module):
    """DFlash drafter backbone with two-phase forward.

    Phase 1 (``inject_kv``): target encoder features -> per-layer Wk/Wv -> KV cache.
    Phase 2 (``draft``): ``[last_token, MASK * K]`` -> full transformer -> hidden states.
    """

    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

    def inject_kv(
        self,
        features: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: RingKVCache,
    ) -> None:
        """Write target encoder features to the KV cache."""
        for layer in self.layers:
            layer.inject_kv(features, position_ids, cache)  # type: ignore[operator]

    def draft(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: RingKVCache,
    ) -> torch.Tensor:
        """Run draft tokens through the transformer, attending to injected KV."""
        query_len = input_ids.shape[-1]
        seq_len = position_ids.shape[-1]
        offset = seq_len - query_len  # = n_injected

        h = self.embed_tokens(input_ids)
        # NO embed norm for DFlash — HF explicitly bypasses it:
        # "The assistant needs embedding without norm" (candidate_generator.py:1676)

        # Bidirectional mask over valid ring positions
        attn_mask = dflash_draft_mask(
            query_len=query_len,
            capacity=cache.capacity(),
            offset=offset,
            device=input_ids.device,
        )

        for layer in self.layers:
            h = layer(h, position_ids, cache, attn_mask)
        return self.norm(h)


class MuseGlimmerDFlashDrafterForCausalLM(nn.Module):
    """DFlash drafter with lm_head for speculative decoding.

    # Not exported — the ring drafter is the export target. This class provides
    # inject_kv + draft for Python-side acceptance testing and parity verification.

    Usage::

        model = MuseGlimmerDFlashDrafterForCausalLM(config)

        # Phase 1: inject target features into KV cache
        model.inject_kv(features, inject_positions, cache)

        # Phase 2: draft K tokens in parallel
        logits = model.draft(draft_input_ids, full_position_ids, cache)
    """

    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.config = config
        self.model = DFlashDrafterModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def inject_kv(
        self,
        features: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: RingKVCache,
    ) -> None:
        """Write encoder features to KV cache."""
        self.model.inject_kv(features, position_ids, cache)

    def draft(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: RingKVCache,
    ) -> torch.Tensor:
        """Draft K tokens, return logits ``(batch, K, vocab_size)``."""
        hidden = self.model.draft(input_ids, position_ids, cache)
        return self.lm_head(hidden)
