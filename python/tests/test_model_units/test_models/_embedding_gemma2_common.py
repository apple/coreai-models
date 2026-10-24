# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tiny EmbeddingGemma 2 configs and helpers shared by the macOS and iOS test modules."""

import torch
import torch.nn.functional as F

from coreai_models.models.macos.embedding_gemma2_vision import EmbeddingGemma2VisionConfig

TINY_TEXT_CONFIG = {
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 96,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "hidden_activation": "gelu_pytorch_tanh",
    "rms_norm_eps": 1e-6,
    "pad_token_id": 0,
    "sliding_window": 3,
    "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
    "per_layer_config": {
        "01": {"head_dim": 32, "num_key_value_heads": 1},
        "03": {"head_dim": 32, "num_key_value_heads": 1},
    },
    "rope_parameters": {
        "full_attention": {"rope_theta": 1_000_000.0, "rope_type": "default"},
        "sliding_attention": {"rope_theta": 10_000.0, "rope_type": "default"},
    },
    "hidden_size_per_layer_input": 24,
    "embedding_dim": 48,
    "max_position_embeddings": 512,
}

TINY_VISION_CONFIG = EmbeddingGemma2VisionConfig(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    head_dim=16,
    patch_size=4,
    pooling_kernel_size=3,
    position_embedding_size=64,
    rms_norm_eps=1e-6,
    rope_theta=100.0,
    text_hidden_size=32,
    max_soft_tokens=12,
)


def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.double(), b.double(), dim=-1)


def _st_mean_pool_normalize(
    token_embeddings: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """sentence-transformers Pooling(mean, include_prompt=True) + Normalize, re-implemented.

    Source: sentence_transformers 6.1.0 ``sentence_transformer/modules/pooling.py`` (mean branch of
    ``_forward_padded``) and ``base/modules/normalize.py``.
    """
    mask = attention_mask.unsqueeze(-1).expand_as(token_embeddings).to(token_embeddings.dtype)
    mean = (token_embeddings * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
    return F.normalize(mean, p=2, dim=-1)


def _ragged_batch(lengths: list[int], seq_len: int, vocab: int, seed: int = 1):
    gen = torch.Generator().manual_seed(seed)
    ids = torch.zeros(len(lengths), seq_len, dtype=torch.int32)
    mask = torch.zeros(len(lengths), seq_len, dtype=torch.int32)
    for row, n in enumerate(lengths):
        ids[row, :n] = torch.randint(1, vocab, (n,), generator=gen, dtype=torch.int32)
        mask[row, :n] = 1
    return ids, mask
