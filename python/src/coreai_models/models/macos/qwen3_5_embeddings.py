# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""macOS Qwen3.5 decoder entered at the merged embedding sequence, for the VLM runtime.

The runtime embeds the prompt itself and splices the vision tower's features into the
image-token positions, so this decoder starts one step later than ``Qwen3_5ForCausalLM``
in ``qwen3_5.py``. Everything after the embedding lookup -- the layers, the four states,
the ``prefill`` entrypoint and the logits -- is that decoder's own, and so is the weight
loading.
"""

from typing import Any

import torch
from typing_extensions import override

from coreai_models._constants import MAIN_GRAPH_NAME
from coreai_models.models.base import BaseForCausalLM, TraceSpec
from coreai_models.models.macos.qwen3_5 import Qwen3_5ForCausalLM
from coreai_models.primitives.macos.cache import DeltaNetCache, KVCache


class Qwen3_5ForCausalLMEmbeddings(Qwen3_5ForCausalLM):
    """Engine-compatible Qwen3.5 text decoder (inputs_embeds variant).

    forward(inputs_embeds, position_ids, k_cache, v_cache, conv_states, recurrent_states)
        -> logits

    ``model.embed_tokens`` is a parameter though ``forward`` never reads it: a tied
    checkpoint's LM head is that table, and loading uses the token decoder's parameter
    names.
    """

    @BaseForCausalLM.cast_logits_bfloat16_to_float16
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.IntTensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        conv_states: torch.Tensor,
        recurrent_states: torch.Tensor,
    ) -> torch.Tensor | tuple:
        kv_cache = KVCache(k_cache, v_cache)
        delta_cache = DeltaNetCache(conv_states, recurrent_states)
        h = inputs_embeds
        for layer in self.model.layers:
            h = layer(
                h,
                position_ids,
                delta_cache if layer.layer_type == "linear_attention" else kv_cache,
            )
        out = self.model.norm(h)
        if self.prefill_mode:
            # A bare `return` traces a leaf node with value `None`; the empty tuple
            # leaves no leaf at all.
            return ()
        return self.lm_head(out)

    @classmethod
    @override
    def export_input_names(cls) -> dict[str, tuple[str, ...]]:
        return {MAIN_GRAPH_NAME: ("inputs_embeds", "position_ids")}

    @override
    def build_reference_inputs(
        self,
        config,
        target_dtype: torch.dtype,
        spec: TraceSpec,
    ) -> dict[str, dict[str, Any]]:
        graph = super().build_reference_inputs(config, target_dtype, spec)[MAIN_GRAPH_NAME]
        graph.pop("input_ids")
        inputs_embeds = torch.randn(1, spec.query_len, config.hidden_size, dtype=target_dtype)
        # The dict binds positionally to forward, so the embeddings lead.
        return {MAIN_GRAPH_NAME: {"inputs_embeds": inputs_embeds, **graph}}

    @override
    def build_dynamic_shapes(self, config, spec: TraceSpec) -> dict[str, Any]:
        shapes = super().build_dynamic_shapes(config, spec)[MAIN_GRAPH_NAME]
        # The sequence axis is the one the token ids moved along; the hidden axis is the
        # model's own width and stays static.
        return {MAIN_GRAPH_NAME: {"inputs_embeds": shapes.pop("input_ids"), **shapes}}
