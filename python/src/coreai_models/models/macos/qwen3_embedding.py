# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Qwen3-Embedding text encoder for Core AI export.

The Qwen3 decoder of ``models.macos.qwen3`` without the LM head, followed by the checkpoint's
sentence-embedding head: the final hidden state of the last real token, L2-normalized.
Attention is causal and the host pads on the right, so padding never reaches a real token; the
graph reads ``attention_mask`` only to find the last real token.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from coreai_models.models.macos.qwen3 import Qwen3ForCausalLM, Qwen3Model


class Qwen3EmbeddingEncoder(nn.Module):
    """``(input_ids, attention_mask) -> embedding`` ``[B, hidden_size]``: unit L2 norm, float32.

    Inputs are right-padded ``[B, L]`` int32. A row with no real token returns the first
    position's vector.
    """

    def __init__(self, model: Qwen3Model) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch, length = input_ids.shape
        positions = torch.arange(length, dtype=torch.int32, device=input_ids.device)
        hidden = self.model(input_ids, positions.unsqueeze(0).expand(batch, length))
        last = (attention_mask.sum(dim=-1) - 1).clamp(min=0)
        index = last.reshape(batch, 1, 1).expand(batch, 1, hidden.shape[-1])
        pooled = hidden.gather(1, index).squeeze(1).to(torch.float32)
        return F.normalize(pooled, dim=-1)


def load_qwen3_embedding_encoder(
    model_dir: str | Path, dtype: torch.dtype = torch.float32
) -> Qwen3EmbeddingEncoder:
    """Build the encoder from a local snapshot dir with ``Qwen3ForCausalLM.from_hf``'s weights."""
    causal_lm = Qwen3ForCausalLM.from_hf(str(model_dir), target_dtype=dtype)
    return Qwen3EmbeddingEncoder(causal_lm.model).eval()


__all__ = ["Qwen3EmbeddingEncoder", "load_qwen3_embedding_encoder"]
