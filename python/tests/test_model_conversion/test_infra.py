# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Export-pipeline infrastructure tests"""

import asyncio
from pathlib import Path

import pytest
from coreai.authoring import AIModelAsset
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM as HFQwen3ForCausalLM,
)

from coreai_models.export import pipeline as export_pipeline
from coreai_models.export._constants import IOS_DEFAULT_MAX_CONTEXT_LENGTH
from coreai_models.export.ios import KEY_CACHE_INPUT_NAME, VALUE_CACHE_INPUT_NAME
from coreai_models.export.pipeline import ExportConfig, _async_export_model


def _tiny_qwen3_config(max_position_embeddings: int) -> Qwen3Config:
    """A small, randomly-initialized Qwen3 config for fast pipeline exports."""
    return Qwen3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=max_position_embeddings,
        tie_word_embeddings=True,
    )


def _save_tiny_qwen3(config: Qwen3Config, dest: Path) -> str:
    """Instantiate a Qwen3 model from ``config`` and save it so the pipeline can
    load it via ``from_pretrained``. Returns the model directory path."""
    model = HFQwen3ForCausalLM(config)
    model.save_pretrained(str(dest))
    return str(dest)


def _cache_context_length(descriptor_type: str) -> int:
    """Extract the context-length (last) dimension from a state descriptor's
    type string, e.g. ``"NDArray (Float16, 2 × 1 × 32 × 1 × 2048)"`` -> 2048."""
    inside = descriptor_type[descriptor_type.index("(") + 1 : descriptor_type.rindex(")")]
    shape_part = inside.split(",", 1)[1]
    dims = [int(dim.strip()) for dim in shape_part.split("×")]
    return dims[-1]


class TestIOSPipelineMaxContextLength:
    @staticmethod
    def test_rejects_context_length_above_hf_config(tmp_path: Path) -> None:
        """The pipeline must reject a --max-context-length larger than the
        model's ``max_position_embeddings`` from its HuggingFace config."""
        native_max = 4096
        config = _tiny_qwen3_config(max_position_embeddings=native_max)
        model_dir = _save_tiny_qwen3(config, tmp_path / "model")

        export_config = ExportConfig(
            hf_model_id=model_dir,
            variant="iOS",
            max_context_length=native_max + 1,
            compute_precision="float16",
            compression="none",
            output_dir=str(tmp_path / "out"),
            overwrite=True,
        )

        with pytest.raises(ValueError, match="max_position_embeddings"):
            asyncio.run(_async_export_model(export_config))

    @staticmethod
    def test_defaults_ios_context_length_to_4096(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no explicit --max-context-length, iOS exports default to 4096.

        Loads the dumped program and checks every function's KV-cache states:
        the context-length dim must match between key/value caches, and the
        maximum context length across all cache-bearing functions must be 4096.
        """
        # Native context window well above the iOS default so we know the cap
        # (4096) — not the model's own limit — is what bounds the export.
        config = _tiny_qwen3_config(max_position_embeddings=8192)
        model_dir = _save_tiny_qwen3(config, tmp_path / "model")

        # The tokenizer/metadata bundling step is unrelated to what we assert
        # here and would require tokenizer files for this synthetic model.
        monkeypatch.setattr(export_pipeline, "bundle_llm_asset", lambda **kwargs: None)

        export_config = ExportConfig(
            hf_model_id=model_dir,
            variant="iOS",
            max_context_length=None,
            compute_precision="float16",
            compression="none",
            output_dir=str(tmp_path / "out"),
            overwrite=True,
        )

        bundle_path = asyncio.run(_async_export_model(export_config))

        aimodel_path = next(Path(bundle_path).glob("*.aimodel"))
        summary = AIModelAsset.load(aimodel_path).summary(include_statistics=False)

        max_context_length = 0
        functions_with_cache = 0
        for function_name in summary.function_names:
            states = dict(summary.function_states(function_name))
            if KEY_CACHE_INPUT_NAME not in states or VALUE_CACHE_INPUT_NAME not in states:
                continue
            functions_with_cache += 1

            key_ctx = _cache_context_length(states[KEY_CACHE_INPUT_NAME])
            value_ctx = _cache_context_length(states[VALUE_CACHE_INPUT_NAME])
            assert key_ctx == value_ctx, (
                f"{function_name}: key/value cache context-length dims differ "
                f"({key_ctx} vs {value_ctx})"
            )
            max_context_length = max(max_context_length, key_ctx)

        assert functions_with_cache > 0, "expected at least one function with KV-cache states"
        assert max_context_length == IOS_DEFAULT_MAX_CONTEXT_LENGTH == 4096
