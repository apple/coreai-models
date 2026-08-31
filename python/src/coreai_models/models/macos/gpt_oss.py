# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import os

import torch
from coreai_torch._compression._floatx import Float4Tensor
from coreai_torch._compression.custom_layers import WeightDequantizeModule
from coreai_torch._compression.utils import wrap_for_parametrization
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssForCausalLM as HFGptOssForCausalLM,
)
from typing_extensions import Self, override

from coreai_models._hf import load_named_tensors_from_weight_files, resolve_rope_theta
from coreai_models.models.base import BaseForCausalLM, _is_layer_key_beyond, move_model_to_disk
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.rope import initialize_rope
from coreai_models.primitives.macos.sdpa import SDPA
from coreai_models.primitives.macos.switch import SwitchGLU

torch.serialization.add_safe_globals([Float4Tensor])

WeightDequantizedParametrization = wrap_for_parametrization(WeightDequantizeModule)


class GptOssSwiGLU(torch.nn.Module):
    def __init__(self: Self, alpha: float = 1.702, limit: float = 7.0) -> None:
        super().__init__()
        self._alpha = alpha
        self._limit = limit

    def forward(self: Self, up: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        gate = torch.clamp(gate, max=self._limit)
        up = torch.clamp(up, min=-self._limit, max=self._limit)

        scaled_gate = self._alpha * gate
        scaled_gate_f32 = scaled_gate.to(torch.float32)
        sigmoid_scaled_gate_f32 = torch.sigmoid(scaled_gate_f32)
        sigmoid_scaled_gate = sigmoid_scaled_gate_f32.to(scaled_gate.dtype)
        pseudo_silu_scaled_gate = gate * sigmoid_scaled_gate

        up_plus_1 = up + 1

        return pseudo_silu_scaled_gate * up_plus_1


class Attention(torch.nn.Module):
    def __init__(self: Self, config: GptOssConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        self.q_proj = torch.nn.Linear(
            config.hidden_size, self.num_attention_heads * self.head_dim, bias=True
        )
        self.k_proj = torch.nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = torch.nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj = torch.nn.Linear(
            self.head_dim * self.num_attention_heads, config.hidden_size, bias=True
        )

        sinks = torch.zeros((self.num_attention_heads,))
        self.sinks = torch.nn.Parameter(sinks)

        # Select attention type per HF convention: config.layer_types is a
        # list of "sliding_attention" / "full_attention" strings, one per
        # layer. Fall back to the alternating pattern used by HF's
        # GptOssModel.__init__ if the config omits it.
        # See transformers/models/gpt_oss/modeling_gpt_oss.py:297 (HF stores
        # self.sliding_window = config.sliding_window if layer_type ==
        # "sliding_attention" else None).
        layer_types = getattr(config, "layer_types", None) or [
            "sliding_attention",
            "full_attention",
        ] * (config.num_hidden_layers // 2)
        is_sliding = layer_types[layer_idx] == "sliding_attention"
        # Our SDPA's `window_size=W` (see coreai_torch._sdpa._maybe_construct_attn_mask)
        # implements the same predicate as HF's sliding_window_causal_mask_function:
        #     q_idx - window_size < k_idx <= q_idx
        # i.e. W keys attended (including self). Matches HF's semantics exactly.
        window_size = config.sliding_window if is_sliding else 0
        self.sdpa = SDPA(is_causal=True, window_size=window_size)

        self.rope = initialize_rope(
            dims=self.head_dim,
            base=resolve_rope_theta(config),
            scaling_config=config.rope_scaling,
        )

    def forward(
        self: Self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        batch_size, query_length, _ = x.shape
        sequence_length = position_ids.shape[-1]
        torch._check_is_size(sequence_length, message="int sequence length >= 0")
        offset = sequence_length - query_length
        torch._check_is_size(offset, message="int offset length >= 0")

        q: torch.Tensor = self.q_proj(x)
        k: torch.Tensor = self.k_proj(x)
        v: torch.Tensor = self.v_proj(x)
        q = q.reshape(batch_size, query_length, self.num_attention_heads, self.head_dim)
        k = k.reshape(batch_size, query_length, self.num_key_value_heads, self.head_dim)
        v = v.reshape(batch_size, query_length, self.num_key_value_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        rope_positions = position_ids.narrow(-1, offset, query_length)
        q = self.rope(q, position_ids=rope_positions)
        k = self.rope(k, position_ids=rope_positions)

        if cache is not None:
            k, v = cache.update_and_fetch(
                self.layer_idx,
                offset,
                k,
                v,
                seq_len=sequence_length,
                query_len=query_length,
            )

        o: torch.Tensor = self.sdpa(q, k, v, sinks=self.sinks)
        o = o.permute(0, 2, 1, 3)
        o = o.reshape(batch_size, query_length, self.num_attention_heads * self.head_dim)

        y = self.o_proj(o)
        return y


class MoeMlp(torch.nn.Module):
    def __init__(self: Self, config: GptOssConfig) -> None:
        super().__init__()
        self.num_active_experts = config.num_experts_per_tok
        self.router = torch.nn.Linear(config.hidden_size, config.num_local_experts, bias=True)
        self.experts = SwitchGLU(
            hidden_size=config.hidden_size,
            moe_intermediate_size=config.intermediate_size,
            num_experts=config.num_local_experts,
            bias=True,
            activation=GptOssSwiGLU(),
        )

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        routes = self.router(x)

        active_experts_scores, active_experts_indices = torch.topk(
            routes, self.num_active_experts, dim=-1, largest=True
        )
        active_experts_indices = active_experts_indices.to(torch.uint16)

        active_expert_weights = torch.softmax(active_experts_scores, dim=-1)
        active_expert_weights = active_expert_weights.unsqueeze(-1)

        y_active_experts = self.experts(x, active_experts_indices)
        y_active_experts_weighted = y_active_experts * active_expert_weights
        y_active_experts_summary = torch.sum(y_active_experts_weighted, dim=-2)
        return y_active_experts_summary


class TransformerBlock(torch.nn.Module):
    def __init__(self: Self, config: GptOssConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MoeMlp(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self: Self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, position_ids, cache)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x


class GptOssModel(torch.nn.Module):
    def __init__(self: Self, config: GptOssConfig) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_types = config.layer_types or [
            "sliding_attention",
            "full_attention",
        ] * (config.num_hidden_layers // 2)
        self.layers = torch.nn.ModuleList(
            [TransformerBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    def forward(
        self: Self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor | None = None,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, position_ids, cache)
        x = self.norm(x)
        return x


class GptOssForCausalLM(BaseForCausalLM):
    _HF_MODEL_CLASS = HFGptOssForCausalLM

    # Emit a second, prefill-only ``prefill`` entrypoint beside ``main``.
    exports_prefill_graph = False

    @override
    def _init_model(self: Self, config: GptOssConfig) -> None:
        self.config = config
        self.model = GptOssModel(config)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        super().load_state_dict(state_dict, strict=strict, assign=assign)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @override
    def _mutate_state_dict(self: Self, state_dict: dict[str, torch.Tensor]) -> None:
        keys = tuple(state_dict.keys())
        for k in keys:
            v = state_dict[k]
            need_to_pop_v = True
            if "_blocks" in k or "_scales" in k:
                assert "bias" not in k, "GPT-OSS only quantizes MoE weights"
                if "_blocks" in k:
                    v = v.flatten(-2)
                if "gate_up_proj" in k:
                    state_dict[
                        k.replace("gate_up_proj_blocks", "gate_proj_blocks").replace(
                            "gate_up_proj_scales", "gate_proj_scales"
                        )
                    ] = v[:, ::2, :].contiguous().unsqueeze(0)
                    state_dict[
                        k.replace("gate_up_proj_blocks", "up_proj_blocks").replace(
                            "gate_up_proj_scales", "up_proj_scales"
                        )
                    ] = v[:, 1::2, :].contiguous().unsqueeze(0)
                else:
                    assert "down_proj" in k
                    state_dict[k] = v.unsqueeze(0).contiguous()
                    need_to_pop_v = False
            else:
                if "gate_up_proj" in k and "bias" not in k:
                    state_dict[k.replace("gate_up_proj", "gate_proj.weight")] = (
                        v[..., ::2].transpose(-1, -2).contiguous().unsqueeze(0)
                    )
                    state_dict[k.replace("gate_up_proj", "up_proj.weight")] = (
                        v[..., 1::2].transpose(-1, -2).contiguous().unsqueeze(0)
                    )
                elif "down_proj" in k and "bias" not in k:
                    state_dict[k.replace("down_proj", "down_proj.weight")] = (
                        v.transpose(-1, -2).contiguous().unsqueeze(0)
                    )
                elif "gate_up_proj_bias" in k:
                    state_dict[k.replace("gate_up_proj_bias", "gate_proj.bias")] = (
                        v[..., ::2].contiguous().unsqueeze(0)
                    )
                    state_dict[k.replace("gate_up_proj_bias", "up_proj.bias")] = (
                        v[..., 1::2].contiguous().unsqueeze(0)
                    )
                elif "down_proj_bias" in k:
                    state_dict[k.replace("down_proj_bias", "down_proj.bias")] = v.unsqueeze(0)
                else:
                    need_to_pop_v = False
            if need_to_pop_v:
                del state_dict[k]

    @override
    @classmethod
    def from_hf(
        cls: type[Self],
        huggingface_model_id: str,
        max_context_length: int | None = None,
        target_dtype: torch.dtype = torch.float16,
        mmap_path: str | None = None,
        num_layers: int | None = None,
    ) -> Self:
        """Load model from HuggingFace model hub.

        Args:
            huggingface_model_id: The HuggingFace model identifier
            max_context_length: Optional maximum context length to override config
            target_dtype: Target dtype for the model weights
            mmap_path: Optional path to use for mmaping the model weights to disk.
                       If provided, the model weights will be saved to this path
                       and memory-mapped to reduce RAM usage during import.
            num_layers: Optional number of transformer layers. When set, only layers
                        0..num_layers-1 are loaded and the config is truncated.

        Returns:
            Instance of the model class loaded with HuggingFace weights
        """
        config = GptOssConfig.from_pretrained(huggingface_model_id)
        if cls._HF_MODEL_CLASS is None:
            raise ValueError(f"{cls.__name__} must define _HF_MODEL_CLASS class attribute")
        msg = "All HuggingFace model should have architectures field populated"
        assert config.architectures is not None, msg
        architecture = config.architectures[0]
        msg = (
            f"expecting {cls._HF_MODEL_CLASS.__name__} architecture, but "
            f"{huggingface_model_id} belongs to {architecture}"
        )
        assert architecture == cls._HF_MODEL_CLASS.__name__, msg

        named_tensors = load_named_tensors_from_weight_files(huggingface_model_id)
        state_dict: dict[str, torch.Tensor] = {}
        for name, tensor in named_tensors.items():
            if tensor.dtype == torch.uint8:
                if "_scales" in name:
                    tensor = tensor.view(torch.float8_e8m0fnu)
                else:
                    assert "_blocks" in name
                    tensor = Float4Tensor(tensor)
                state_dict[name] = tensor
            else:
                if tensor.dtype != target_dtype:
                    tensor = tensor.to(target_dtype)
                state_dict[name] = tensor

        config = cls._get_reauthored_config(config, max_context_length, num_layers=num_layers)

        if num_layers is not None:
            state_dict = {
                k: v for k, v in state_dict.items() if not _is_layer_key_beyond(k, num_layers)
            }

        model = cls(config, model_device="meta")
        model.to(dtype=target_dtype)
        model._mutate_state_dict(state_dict)
        if any(k.endswith("_blocks") for k in state_dict):
            for layer_idx, transformer in enumerate(model.model.layers):
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    prefix = f"model.layers.{layer_idx}.mlp.experts.{proj}"
                    blocks = state_dict.pop(f"{prefix}_blocks")
                    scales = state_dict.pop(f"{prefix}_scales")
                    parametrization = WeightDequantizedParametrization(
                        blocks, scales, output_dtype=target_dtype
                    )
                    proj_module = getattr(transformer.mlp.experts, proj)
                    torch.nn.utils.parametrize.register_parametrization(
                        proj_module,
                        "weight",
                        parametrization,
                    )
                    proj_module.parametrizations.weight.original = torch.nn.Parameter(
                        torch.zeros(1, dtype=target_dtype)
                    )
        model.load_state_dict(state_dict, assign=True, strict=False)

        if mmap_path is not None:
            move_model_to_disk(model, path=mmap_path)

        return model

    @override
    @classmethod
    def from_hf_memory_efficient(
        cls: type[Self],
        huggingface_model_id: str,
        max_context_length: int | None = None,
        target_dtype: torch.dtype = torch.float16,
        mmap_path: str | None = None,
        num_layers: int | None = None,
        hf_config_attr: str | None = None,
        hf_state_dict_prefix: str = "",
    ) -> Self:
        # GptOss falls back to the `from_hf` path
        assert hf_config_attr is None, (
            f"GptOss does not support hf_config_attr (got {hf_config_attr!r}); "
            "remove it from the registry entry or extend this override."
        )
        assert hf_state_dict_prefix == "", (
            f"GptOss does not support hf_state_dict_prefix (got {hf_state_dict_prefix!r}); "
            "remove it from the registry entry or extend this override."
        )
        file_mmap = os.path.join(mmap_path, "model.pt") if mmap_path is not None else None
        return cls.from_hf(
            huggingface_model_id,
            max_context_length=max_context_length,
            target_dtype=target_dtype,
            mmap_path=file_mmap,
            num_layers=num_layers,
        )

    @BaseForCausalLM.cast_logits_bfloat16_to_float16
    def forward(
        self: Self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor | tuple:
        cache = KVCache(k_cache, v_cache)
        out = self.model(input_ids, position_ids, cache)
        if self.prefill_mode:
            # A bare `return` causes torch export to trace a leaf node with value
            # `None` rather than having no leaf nodes whatsoever. Remedied with
            # empty tuple.
            return ()
        return self.lm_head(out)
