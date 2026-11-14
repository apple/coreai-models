# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for loading a checkpoint through BaseForCausalLM.from_hf_memory_efficient."""

from pathlib import Path

import pytest
import torch
from safetensors import SafetensorError
from safetensors.torch import load_file, save_file
from transformers import Qwen3Config
from transformers import Qwen3ForCausalLM as HFQwen3ForCausalLM

from coreai_models.models.macos.qwen3 import Qwen3ForCausalLM


def _save_tiny_checkpoint(folder: Path) -> HFQwen3ForCausalLM:
    """Save a Qwen3 checkpoint of two small layers to *folder*."""
    config = Qwen3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
    )
    hf_model = HFQwen3ForCausalLM(config).to(torch.float32)
    hf_model.save_pretrained(folder, safe_serialization=True)
    return hf_model


class TestFromHfMemoryEfficient:
    """Test loading a checkpoint that is already on disk."""

    def test_local_checkpoint_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """A local checkpoint directory is read in place, with no Hub download."""
        hf_model = _save_tiny_checkpoint(tmp_path)
        monkeypatch.setattr(
            "coreai_models.models.base.snapshot_download",
            lambda *args, **kwargs: pytest.fail("snapshot_download was called"),
        )

        model = Qwen3ForCausalLM.from_hf_memory_efficient(
            str(tmp_path), max_context_length=64, target_dtype=torch.float32
        )

        torch.testing.assert_close(
            model.model.embed_tokens.weight, hf_model.model.embed_tokens.weight
        )

    def test_truncated_weights_raise(self, tmp_path: Path):
        """A truncated weight file raises instead of loading part of the model."""
        _save_tiny_checkpoint(tmp_path)
        weights = tmp_path / "model.safetensors"
        data = weights.read_bytes()
        weights.write_bytes(data[: len(data) // 2])  # keep the first half of the file

        with pytest.raises(SafetensorError):
            Qwen3ForCausalLM.from_hf_memory_efficient(str(tmp_path), max_context_length=64)

    def test_missing_layer_weights_raise(self, tmp_path: Path):
        """A checkpoint without one layer's weights raises instead of leaving them empty."""
        _save_tiny_checkpoint(tmp_path)
        weights = tmp_path / "model.safetensors"
        tensors = load_file(str(weights))
        # Drop every tensor of the second layer (`model.layers.1.*`) and save the rest back.
        kept = {name: tensor for name, tensor in tensors.items() if ".layers.1." not in name}
        save_file(kept, str(weights))

        with pytest.raises(RuntimeError, match="Parameters not loaded"):
            Qwen3ForCausalLM.from_hf_memory_efficient(str(tmp_path), max_context_length=64)
