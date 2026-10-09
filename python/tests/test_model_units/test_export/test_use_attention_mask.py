# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the ``--attention-mask [default|batched|paged]`` export flag.

The flag threads CLI -> ``ExportConfig`` -> ``from_hf*`` -> model so the exported macOS
graph declares an additive ``attn_mask`` input instead of the implicit ``is_causal`` mask.
It is macOS-only and defaults to ``default``, leaving the standard causal export untouched.

Flag plumbing only -- nothing here downloads weights or runs an export.
"""

from __future__ import annotations

import pytest

from coreai_models.export.pipeline import AttentionMaskMode, ExportConfig
from coreai_models.llm.export import _resolve_export_config, build_parser


def test_defaults() -> None:
    config = ExportConfig(hf_model_id="org/model")
    assert config.attention_mask_mode is AttentionMaskMode.DEFAULT
    assert config.emits_attn_mask is False
    assert config.dynamic_max_batch_size is None


def test_batched_emits_mask_and_defaults_batch_to_8() -> None:
    """`--attention-mask batched` alone emits the mask and yields a batch-8 dynamic graph."""
    config = ExportConfig(
        hf_model_id="org/model", variant="macOS", attention_mask_mode=AttentionMaskMode.BATCHED
    )
    assert config.emits_attn_mask is True
    assert config.dynamic_max_batch_size == 8


def test_rejects_batched_mask_on_ios() -> None:
    """iOS has its own causal-mask contract; the additive mask is macOS-only."""
    with pytest.raises(ValueError, match="macOS"):
        ExportConfig(
            hf_model_id="org/model", variant="iOS", attention_mask_mode=AttentionMaskMode.BATCHED
        )


def test_rejects_paged() -> None:
    """``paged`` is reserved for the hardware paged backend; it fails with an explicit error."""
    with pytest.raises(NotImplementedError, match="not available"):
        ExportConfig(hf_model_id="org/model", attention_mask_mode=AttentionMaskMode.PAGED)


def test_rejects_dynamic_batch_on_ios() -> None:
    with pytest.raises(ValueError, match="macOS"):
        ExportConfig(hf_model_id="org/model", variant="iOS", dynamic_max_batch_size=8)


def test_rejects_dynamic_batch_below_two() -> None:
    with pytest.raises(ValueError, match="dynamic_max_batch_size"):
        ExportConfig(hf_model_id="org/model", dynamic_max_batch_size=1)


def test_rejects_dynamic_batch_without_attn_mask() -> None:
    """A dynamic batch without a batched attn_mask would trace a graph the runtime can't serve."""
    with pytest.raises(ValueError, match="requires attention_mask_mode=BATCHED"):
        ExportConfig(hf_model_id="org/model", variant="macOS", dynamic_max_batch_size=8)


def test_cli_resolves_attention_mask_onto_config() -> None:
    assert (
        _resolve_export_config(build_parser().parse_args(["qwen3-0.6b"])).attention_mask_mode
        is AttentionMaskMode.DEFAULT
    )
    assert (
        _resolve_export_config(
            build_parser().parse_args(["qwen3-0.6b", "--attention-mask", "batched"])
        ).attention_mask_mode
        is AttentionMaskMode.BATCHED
    )


def test_cli_resolves_dynamic_batch_size_onto_config() -> None:
    args = build_parser().parse_args(
        ["qwen3-0.6b", "--dynamic-batch-size", "8", "--attention-mask", "batched"]
    )
    assert _resolve_export_config(args).dynamic_max_batch_size == 8
