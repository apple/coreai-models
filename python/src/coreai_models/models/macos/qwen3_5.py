# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""macOS Qwen3.5 / Qwen3.5-MoE.

A hybrid decoder: ``config.layer_types`` marks each layer as ``full_attention``
(gated SDPA over a growing KV cache) or ``linear_attention`` (a gated delta net over
fixed-size conv + recurrent state), so the graph carries four states rather than two.
"""

import copy
import re
from typing import Any

import torch
import torch.nn as nn
from coreai_torch.composite_ops import GatedDeltaUpdate
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForCausalLM as HFQwen3_5ForCausalLM,
)
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
from typing_extensions import Self, override

from coreai_models._constants import (
    CONV_STATES_NAME,
    MAIN_GRAPH_NAME,
    RECURRENT_STATES_NAME,
)
from coreai_models._hf import resolve_rope_theta
from coreai_models.models.base import BaseForCausalLM, TraceSpec
from coreai_models.primitives.macos.cache import DeltaNetCache, KVCache
from coreai_models.primitives.macos.mlp import MLP
from coreai_models.primitives.macos.rms_norm import RMSNormGated
from coreai_models.primitives.macos.rms_norm import RMSNormPlusOne as RMSNorm
from coreai_models.primitives.macos.rope import initialize_rope
from coreai_models.primitives.macos.sdpa import SDPA
from coreai_models.primitives.macos.switch import SwitchGLU

_LAYER_KEY_RE = re.compile(r"model\.layers\.(\d+)\.")


class Attention(nn.Module):
    def __init__(
        self, config: Qwen3_5TextConfig, layer_idx: int, cache_layer_idx: int | None = None
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        # Row this layer owns in the KV cache. The cache is sized by the number of
        # full-attention layers, so in a hybrid stack this is not layer_idx.
        self.cache_layer_idx = layer_idx if cache_layer_idx is None else cache_layer_idx

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.head_dim = head_dim = getattr(config, "head_dim", dim // n_heads)
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        has_bias = config.attention_bias

        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)

        # Fused projection produces query | key | value | gate, in that order along
        # the output dim. Each "head" is head_dim wide.
        self.qkv_gate_proj = nn.Linear(
            dim,
            (2 * n_heads + 2 * n_kv_heads) * head_dim,
            bias=has_bias,
        )
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=has_bias)

        eps = config.rms_norm_eps
        self.qk_norm = RMSNorm(head_dim, eps=eps, n_heads=n_heads + n_kv_heads)

        # Partial rotary (0.25 on every checkpoint): the runtime's composite RoPE op
        # mis-lowers it on newer OS betas, so emit the rotation with raw torch ops, as
        # phi3.py does.
        self.rope = initialize_rope(
            dims=int(head_dim * partial_rotary_factor),
            base=resolve_rope_theta(config),
            decomposed=partial_rotary_factor < 1.0,
        )

        self.sdpa = SDPA(is_causal=True)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        batch_size, query_len, _ = x.shape
        n_heads, n_kv_heads, head_dim = self.n_heads, self.n_kv_heads, self.head_dim

        # One matmul, one reshape, one permute — out: (B, total_heads, S, head_dim)
        qkvg = (
            self.qkv_gate_proj(x)
            .reshape(batch_size, query_len, 2 * n_heads + 2 * n_kv_heads, head_dim)
            .permute(0, 2, 1, 3)
        )

        # query and key live adjacent so they share qk_norm and rope.
        query_key = qkvg.narrow(1, 0, n_heads + n_kv_heads)
        value = qkvg.narrow(1, n_heads + n_kv_heads, n_kv_heads)
        gate = qkvg.narrow(1, n_heads + 2 * n_kv_heads, n_heads)

        query_key = self.qk_norm(query_key)

        seq_len = position_ids.shape[-1]
        torch._check_is_size(query_len)
        torch._check_is_size(seq_len)
        offset = seq_len - query_len
        torch._check_is_size(offset)
        rope_positions = position_ids.narrow(-1, offset, query_len)

        query_key = self.rope(query_key, position_ids=rope_positions)
        query = query_key.narrow(1, 0, n_heads)
        key = query_key.narrow(1, n_heads, n_kv_heads)

        if cache is not None:
            key, value = cache.update_and_fetch(
                self.cache_layer_idx, offset, key, value, seq_len=seq_len, query_len=query_len
            )

        # Apply gate while still in (B, n_heads, S, head_dim) layout — no extra reshape.
        output = self.sdpa(query=query, key=key, value=value) * torch.nn.functional.sigmoid(gate)

        return self.o_proj(
            output.permute(0, 2, 1, 3).reshape(batch_size, query_len, n_heads * head_dim)
        )


class GatedDeltaNet(nn.Module):
    def __init__(
        self, config: Qwen3_5TextConfig, layer_idx: int, cache_layer_idx: int | None = None
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        # Row this layer owns in the conv/recurrent states, which are sized by the
        # number of linear-attention layers rather than num_hidden_layers.
        self.cache_layer_idx = layer_idx if cache_layer_idx is None else cache_layer_idx

        hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = conv_kernel_size = config.linear_conv_kernel_dim
        self.conv_dim = conv_dim = key_dim * 2 + value_dim
        if self.num_v_heads % self.num_k_heads != 0:
            err = f"Invalid head dims {self.num_v_heads}, {self.num_k_heads}."
            raise ValueError(err)
        self.head_ratio = self.num_v_heads // self.num_k_heads

        # Depthwise causal conv filter, tap-major as (kernel, channels) for the shift
        # chain in `forward`. `_mutate_state_dict` converts HF's (channels, 1, kernel).
        #
        # A bare Parameter rather than an `nn.Conv1d`: the quantizer fake-quants every
        # Conv1d it finds, and this filter's axis 1 has size 1, so the 4-bit recipe's
        # block_size-32 spec cannot apply -- it warns per linear-attention layer, and
        # neither `module_name_configs` nor `module_type_configs` suppresses that.
        self.conv_weight = nn.Parameter(torch.zeros(conv_kernel_size, conv_dim))

        # Fused input projection produces qkv | z | b | a along the output dim.
        # qkv goes through the conv; z/b/a bypass it and slice off the tail.
        self.in_proj_all = nn.Linear(
            hidden_size,
            conv_dim + value_dim + 2 * self.num_v_heads,
            bias=False,
        )

        self.out_proj = nn.Linear(value_dim, hidden_size, bias=False)

        self.norm = RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.ones(self.num_v_heads))
        self._gated_delta_update = GatedDeltaUpdate()

    def forward(
        self,
        x: torch.Tensor,
        cache: DeltaNetCache | None = None,
    ) -> torch.Tensor:
        B, S, _ = x.shape
        conv_kernel_size = self.conv_kernel_size

        # One matmul, then narrow out each of qkv / z / b / a (all views, no copies).
        fused = self.in_proj_all(x)
        mixed_qkv = fused.narrow(-1, 0, self.conv_dim)
        z = fused.narrow(-1, self.conv_dim, self.value_dim).reshape(
            B, S, self.num_v_heads, self.head_v_dim
        )
        b = fused.narrow(-1, self.conv_dim + self.value_dim, self.num_v_heads)
        a = fused.narrow(-1, self.conv_dim + self.value_dim + self.num_v_heads, self.num_v_heads)

        if cache is not None:
            # The cache holds the trailing conv_kernel_size tokens channels-last, the
            # layout `in_proj_all` emits, so this slice drops straight into the concat
            # below. Only the older conv_kernel_size - 1 rows are history for this step.
            conv_state = cache.conv_states[self.cache_layer_idx].narrow(1, 1, conv_kernel_size - 1)
            recurrent_state = cache.recurrent_states[self.cache_layer_idx]
        else:
            conv_state = torch.zeros(B, conv_kernel_size - 1, self.conv_dim, dtype=x.dtype)
            recurrent_state = torch.zeros(
                B, self.num_v_heads, self.head_k_dim, self.head_v_dim, dtype=x.dtype
            )

        # (batch, S + conv_kernel_size - 1, conv_dim)
        conv_input = torch.cat([conv_state, mixed_qkv], dim=1)

        if cache is not None:
            # Trailing conv_kernel_size tokens, already in the cache's layout.
            cache.update_conv_state(
                self.cache_layer_idx,
                conv_input.narrow(1, S - 1, conv_kernel_size),
            )

        # The depthwise causal conv as conv_kernel_size scaled shifts, channels-last
        # throughout. Prepending conv_kernel_size - 1 taps gives exactly S outputs.
        conv_out = conv_input.narrow(1, 0, S) * self.conv_weight[0]
        for tap in range(1, conv_kernel_size):
            conv_out = conv_out + conv_input.narrow(1, tap, S) * self.conv_weight[tap]

        qkv_activated = nn.functional.silu(conv_out)

        q, k, v = torch.split(qkv_activated, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(B, S, self.num_k_heads, self.head_k_dim)
        k = k.reshape(B, S, self.num_k_heads, self.head_k_dim)
        v = v.reshape(B, S, self.num_v_heads, self.head_v_dim)

        if self.head_ratio > 1:
            q = q.repeat_interleave(self.head_ratio, dim=2)
            k = k.repeat_interleave(self.head_ratio, dim=2)

        beta = b.sigmoid()
        g = (-self.A_log.float().exp() * nn.functional.softplus(a.float() + self.dt_bias)).to(
            q.dtype
        )

        core_attn_out, new_state = self._gated_delta_update(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            g.transpose(1, 2),
            beta.transpose(1, 2),
            recurrent_state,
        )

        if cache is not None:
            cache.update_recurrent_state(self.cache_layer_idx, new_state)

        out = self.norm(core_attn_out, z)
        return self.out_proj(out.reshape(B, S, self.value_dim))


class SparseMoeBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        dim = config.hidden_size
        num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.switch_mlp = SwitchGLU(dim, config.moe_intermediate_size, num_experts)
        self.shared_expert = MLP(dim, config.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # topk(softmax(logits)) renormalized over the kept top-k equals softmax over the
        # top-k of the raw logits (the full-expert denominator cancels), so select on the
        # logits and normalize over the kept k — a top_k-wide softmax, not num_experts-wide.
        scores, indices = torch.topk(self.gate(x), self.top_k, dim=-1)
        scores = torch.softmax(scores, dim=-1, dtype=torch.float32)
        indices = indices.to(torch.uint16)

        expert_out = self.switch_mlp(x, indices)
        expert_out = (expert_out * scores.unsqueeze(-1)).sum(dim=-2)

        shared_out = torch.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
        return (expert_out + shared_out).to(x.dtype)


class TransformerBlock(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]

        # Each state tensor carries one row per layer of its *own* type, so a layer's
        # row is its position among the layers sharing its type -- not layer_idx.
        cache_layer_idx = (config.layer_types or [])[:layer_idx].count(self.layer_type)

        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(config, layer_idx, cache_layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Attention(config, layer_idx, cache_layer_idx)

        rms_norm_eps = config.rms_norm_eps
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # Qwen3.5 is dense, Qwen3.5-MoE routes every layer; neither config carries
        # Qwen3-Next's `mlp_only_layers` / `decoder_sparse_step` interleaving.
        if getattr(config, "num_experts", 0):
            self.mlp = SparseMoeBlock(config)
        else:
            self.mlp = MLP(hidden_size, config.intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | DeltaNetCache | None = None,
    ) -> torch.Tensor:
        if self.layer_type == "linear_attention":
            r = self.linear_attn(self.input_layernorm(x), cache=cache)
        elif self.layer_type == "full_attention":
            r = self.self_attn(self.input_layernorm(x), position_ids, cache=cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class Qwen3_5Model(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        kv_cache: KVCache | None = None,
        delta_cache: DeltaNetCache | None = None,
    ) -> torch.Tensor:
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(
                h,
                position_ids,
                delta_cache if layer.layer_type == "linear_attention" else kv_cache,
            )
        return self.norm(h)


class Qwen3_5ForCausalLM(BaseForCausalLM):
    _HF_MODEL_CLASS = HFQwen3_5ForCausalLM
    # lm_head.weight sits at the checkpoint root (outside model.language_model.)
    # for models with untied word embeddings (e.g. 9B).
    _extra_hf_shared_keys = ["lm_head.weight"]
    # Hybrid: states are sized per layer type, so a plain layer-count cut isn't supported.
    supports_num_layers = False
    # coreai-opt's graph-mode prepare rejects the GatedDeltaNet source partition.
    supports_graph_quantization = False

    # Emit a second, prefill-only ``prefill`` entrypoint beside ``main``. Every state this
    # model carries is written by an in-place update, so all four survive the trim that
    # kills the LM head. The recurrent formulation of the linear-attention layers already
    # carries conv/recurrent state across calls, which is what makes the runner's chunked
    # prefill correct here as well as on `main`.
    exports_prefill_graph = True

    # Keys arrive either already prefixed ("model.layers.{i}.xxx", from `from_hf`) or
    # with "model.language_model." stripped off ("layers.{i}.xxx", from the
    # memory-efficient path). Both normalize to "model.xxx"; "lm_head." is left alone.
    _EXPECTED_KEY_PREFIXES = ("model.", "lm_head.", "layers.", "embed_tokens.", "norm.")

    @override
    def _init_model(self, config: Qwen3_5TextConfig) -> None:
        if isinstance(config, (Qwen3_5Config, Qwen3_5MoeConfig)):
            config = config.text_config
            self.config = config
        self.model = Qwen3_5Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @BaseForCausalLM.cast_logits_bfloat16_to_float16
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        conv_states: torch.Tensor,
        recurrent_states: torch.Tensor,
    ) -> torch.Tensor | tuple:
        kv_cache = KVCache(k_cache, v_cache)
        delta_cache = DeltaNetCache(conv_states, recurrent_states)
        out = self.model(input_ids, position_ids, kv_cache, delta_cache)
        if self.prefill_mode:
            # A bare `return` causes torch export to trace a leaf node with value
            # `None` rather than having no leaf nodes whatsoever. Remedied with
            # empty tuple.
            return ()
        return self.lm_head(out)

    # ------------------------------------------------------------------
    # Export contract
    #
    # The linear-attention layers add two fixed-shape states on top of the base KV pair.
    # All four are sized by the number of layers of the *relevant* type: in a 3:1 hybrid,
    # sizing the KV pair by num_hidden_layers would allocate 4x the rows anything writes,
    # tens of GB of dead cache at long context. Each layer reaches its row via
    # `cache_layer_idx`.
    # ------------------------------------------------------------------

    @classmethod
    @override
    def export_state_names(cls) -> dict[str, tuple[str, ...]]:
        base = super().export_state_names()[MAIN_GRAPH_NAME]
        return {MAIN_GRAPH_NAME: (*base, CONV_STATES_NAME, RECURRENT_STATES_NAME)}

    @classmethod
    def _count_layer_type(cls, config, layer_type: str) -> int:
        # Only the layers the model actually builds, in case num_hidden_layers was lowered.
        return (config.layer_types or [])[: config.num_hidden_layers].count(layer_type)

    @classmethod
    def kv_cache_layer_count(cls, config) -> int:
        """Only the full-attention layers own a KV-cache row."""
        return cls._count_layer_type(config, "full_attention")

    @classmethod
    def delta_cache_layer_count(cls, config) -> int:
        """Only the linear-attention layers own a conv/recurrent-state row."""
        return cls._count_layer_type(config, "linear_attention")

    @classmethod
    def _kv_sized_config(cls, config):
        """``config`` with its depth reported as the full-attention layer count.

        ``KVCache.create_cache_tensors`` sizes the layer dim from
        ``num_hidden_layers``. Handing it this copy is what keeps the cache from
        carrying rows no layer writes, without the primitive needing to know about
        hybrid stacks. A copy, so the real config is untouched.
        """
        kv_config = copy.deepcopy(config)
        kv_config.num_hidden_layers = cls.kv_cache_layer_count(config)
        return kv_config

    @classmethod
    def create_kv_cache_tensors(
        cls, config, dtype: torch.dtype = torch.float32, seq_len: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the KV pair with one row per full-attention layer."""
        return KVCache.create_cache_tensors(
            cls._kv_sized_config(config), dtype=dtype, seq_len=seq_len
        )

    @classmethod
    def create_delta_cache_tensors(
        cls, config, dtype: torch.dtype = torch.float32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the DeltaNet states, one row per linear-attention layer."""
        return DeltaNetCache.create_cache_tensors(
            config, n_layers=cls.delta_cache_layer_count(config), dtype=dtype
        )

    @override
    def build_reference_inputs(
        self,
        config,
        target_dtype: torch.dtype,
        spec: TraceSpec,
    ) -> dict[str, dict[str, Any]]:
        # The depth-reduced config only reaches the KV pair; everything downstream of
        # here reads the real one.
        graphs = super().build_reference_inputs(self._kv_sized_config(config), target_dtype, spec)
        conv_states, recurrent_states = self.create_delta_cache_tensors(config, dtype=target_dtype)
        graphs[MAIN_GRAPH_NAME]["conv_states"] = conv_states
        graphs[MAIN_GRAPH_NAME]["recurrent_states"] = recurrent_states
        return graphs

    @override
    def build_dynamic_shapes(self, config, spec: TraceSpec) -> dict[str, Any]:
        graphs = super().build_dynamic_shapes(config, spec)
        # `None` pins both to their traced shape: the conv window is conv_kernel_size
        # wide and the recurrent state is per-head, so neither depends on seq length.
        graphs[MAIN_GRAPH_NAME]["conv_states"] = None
        graphs[MAIN_GRAPH_NAME]["recurrent_states"] = None
        return graphs

    def _normalize_keys(
        self: Self, state_dict: dict[str, torch.Tensor], strict: bool = True
    ) -> None:
        """Bring HF state dict keys into this module's namespace, in-place.

        With ``strict``, a key outside ``_EXPECTED_KEY_PREFIXES`` raises; otherwise it is
        left as is, for ``load_state_dict`` to report as unexpected.
        """
        for k in list(state_dict.keys()):
            if not k.startswith(self._EXPECTED_KEY_PREFIXES):
                if strict:
                    err = (
                        f"Unexpected key in state dict: {k!r}. "
                        f"Expected one of {self._EXPECTED_KEY_PREFIXES}."
                    )
                    raise ValueError(err)
                continue
            if not k.startswith(("model.", "lm_head.")):
                state_dict[f"model.{k}"] = state_dict.pop(k)

    @override
    def _mutate_state_dict(self: Self, state_dict: dict[str, torch.Tensor]) -> None:
        self._normalize_keys(state_dict)

        # Fuse per-projection HF weights into the fused projections declared in
        # __init__. Idempotent: each fusion skips if its source keys are already popped,
        # so an already-fused state dict passes through unchanged.
        max_layer = max(
            (int(m.group(1)) for k in state_dict if (m := _LAYER_KEY_RE.match(k))), default=-1
        )

        # Self-attention: q_proj | k_proj | v_proj  →  qkv_gate_proj
        # (HF q_proj already includes the gate per head as
        #  [query (head_dim) | gate (head_dim)] along the output dim).
        for i in range(max_layer + 1):
            q_proj_key = f"model.layers.{i}.self_attn.q_proj.weight"
            k_proj_key = f"model.layers.{i}.self_attn.k_proj.weight"
            v_proj_key = f"model.layers.{i}.self_attn.v_proj.weight"

            if q_proj_key not in state_dict:
                continue

            layer = self.model.layers[i]
            n_heads = layer.self_attn.n_heads
            n_kv_heads = layer.self_attn.n_kv_heads
            head_dim = layer.self_attn.head_dim

            q_proj_w = state_dict.pop(q_proj_key)
            k_proj_w = state_dict.pop(k_proj_key)
            v_proj_w = state_dict.pop(v_proj_key)

            q_proj_w_split = q_proj_w.reshape(n_heads, 2, head_dim, -1)
            query_w = q_proj_w_split[:, 0].reshape(n_heads * head_dim, -1)
            gate_w = q_proj_w_split[:, 1].reshape(n_heads * head_dim, -1)

            state_dict[f"model.layers.{i}.self_attn.qkv_gate_proj.weight"] = torch.cat(
                [query_w, k_proj_w, v_proj_w, gate_w], dim=0
            )

            q_proj_b_key = f"model.layers.{i}.self_attn.q_proj.bias"
            if q_proj_b_key in state_dict:
                k_proj_b_key = f"model.layers.{i}.self_attn.k_proj.bias"
                v_proj_b_key = f"model.layers.{i}.self_attn.v_proj.bias"
                q_proj_b = state_dict.pop(q_proj_b_key)
                k_proj_b = state_dict.pop(k_proj_b_key)
                v_proj_b = state_dict.pop(v_proj_b_key)

                q_proj_b_split = q_proj_b.reshape(n_heads, 2, head_dim)
                query_b = q_proj_b_split[:, 0].reshape(n_heads * head_dim)
                gate_b = q_proj_b_split[:, 1].reshape(n_heads * head_dim)

                state_dict[f"model.layers.{i}.self_attn.qkv_gate_proj.bias"] = torch.cat(
                    [query_b, k_proj_b, v_proj_b, gate_b], dim=0
                )

            # q_norm | k_norm  →  qk_norm. The HF norms are scalar-per-head_dim;
            # broadcast across heads then concat along the head axis.
            q_norm_key = f"model.layers.{i}.self_attn.q_norm.weight"
            k_norm_key = f"model.layers.{i}.self_attn.k_norm.weight"
            if q_norm_key in state_dict and k_norm_key in state_dict:
                q_norm_w = state_dict.pop(q_norm_key).unsqueeze(0).unsqueeze(0)
                k_norm_w = state_dict.pop(k_norm_key).unsqueeze(0).unsqueeze(0)

                q_repeated = q_norm_w.expand(n_heads, 1, head_dim)
                k_repeated = k_norm_w.expand(n_kv_heads, 1, head_dim)

                state_dict[f"model.layers.{i}.self_attn.qk_norm.weight"] = torch.cat(
                    [q_repeated, k_repeated], dim=0
                )

        # GDN: in_proj_qkv | in_proj_z | in_proj_b | in_proj_a  →  in_proj_all
        for i in range(max_layer + 1):
            qkv_key = f"model.layers.{i}.linear_attn.in_proj_qkv.weight"
            z_key = f"model.layers.{i}.linear_attn.in_proj_z.weight"
            b_key = f"model.layers.{i}.linear_attn.in_proj_b.weight"
            a_key = f"model.layers.{i}.linear_attn.in_proj_a.weight"

            if qkv_key not in state_dict:
                continue

            qkv_w = state_dict.pop(qkv_key)
            z_w = state_dict.pop(z_key)
            b_w = state_dict.pop(b_key)
            a_w = state_dict.pop(a_key)

            state_dict[f"model.layers.{i}.linear_attn.in_proj_all.weight"] = torch.cat(
                [qkv_w, z_w, b_w, a_w], dim=0
            )

        # GDN: conv1d.weight [conv_dim, 1, kernel] → conv_weight [kernel, conv_dim].
        for i in range(max_layer + 1):
            conv_key = f"model.layers.{i}.linear_attn.conv1d.weight"
            if conv_key not in state_dict:
                continue
            conv_w = state_dict.pop(conv_key).squeeze(1).transpose(0, 1).contiguous()
            state_dict[f"model.layers.{i}.linear_attn.conv_weight"] = conv_w

        # MoE: experts.gate_up_proj [N, 2*I, H] → switch_mlp.{gate,up}_proj.weight [1, N, I, H]
        #       experts.down_proj    [N, H, I]   → switch_mlp.down_proj.weight       [1, N, H, I]
        for i in range(max_layer + 1):
            prefix = f"model.layers.{i}.mlp"
            gate_up_key = f"{prefix}.experts.gate_up_proj"
            down_key = f"{prefix}.experts.down_proj"

            if gate_up_key not in state_dict:
                continue

            gate_up = state_dict.pop(gate_up_key)
            gate_w, up_w = gate_up.chunk(2, dim=1)
            state_dict[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_w.unsqueeze(0)
            state_dict[f"{prefix}.switch_mlp.up_proj.weight"] = up_w.unsqueeze(0)

            down = state_dict.pop(down_key)
            state_dict[f"{prefix}.switch_mlp.down_proj.weight"] = down.unsqueeze(0)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        # Every load path lands here, including the shared (layerless) slice that
        # `from_hf_memory_efficient` assigns separately -- that slice never reaches
        # `_mutate_state_dict`, so this is where its keys get re-prefixed. Idempotent,
        # so keys already in this namespace pass through. Normalizes a copy, leaving
        # the caller's dict as it was.
        state_dict = dict(state_dict)
        self._normalize_keys(state_dict, strict=strict)
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        return result
