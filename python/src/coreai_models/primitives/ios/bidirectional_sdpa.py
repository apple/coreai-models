# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Re-authored bidirectional scaled dot-product attention for SAM3.

Pure PyTorch ops (split / cat / matmul / softmax) in BC1S layout.

Mask convention: ``attention_mask`` (when provided) is in GPU layout
(B, num_heads, query_seq, key_seq), with ``-40000.0`` for masked
positions. ``None`` is unmasked attention.
"""

import torch
import torch.nn as nn


class BidirectionalSDPA(nn.Module):
    """Per-head split/cat SDPA for iOS.

    Chunks the query when ``query_seq > query_chunk_size``
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        query_chunk_size: int = 576,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.query_chunk_size = query_chunk_size
        # Scale as f16 buffer to avoid f32 constants in the graph.
        self.register_buffer("_scale", torch.tensor(head_dim**-0.5, dtype=torch.float16))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query_seq = query.shape[3]
        if query_seq > self.query_chunk_size:
            return self._chunked_forward(query, key, value, attention_mask)
        return self._standard_forward(query, key, value, attention_mask)

    def _chunked_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        chunk_size = self.query_chunk_size
        query_seq = query.shape[3]
        chunks = []
        for start in range(0, query_seq, chunk_size):
            end = min(start + chunk_size, query_seq)
            q_chunk = query[:, :, :, start:end]
            mask_chunk = attention_mask[:, :, start:end, :] if attention_mask is not None else None
            chunks.append(self._standard_forward(q_chunk, key, value, mask_chunk))
        return torch.cat(chunks, dim=3)

    def _standard_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        D = self.head_dim
        scale = self._scale.to(dtype=query.dtype)

        key_t = key.transpose(-3, -1)  # (B, key_seq, 1, num_heads*head_dim)
        queries = query.split(D, dim=1)
        keys = list(key_t.split(D, dim=-1))
        values = value.split(D, dim=1)
        n_heads = len(queries)

        for i in range(len(keys)):
            keys[i] = keys[i].permute(0, 2, 3, 1)  # (B, 1, D, key_seq)

        scores = []
        for head_idx in range(n_heads):
            q = queries[head_idx].permute(0, 2, 3, 1)  # (B, 1, query_seq, D)
            k = keys[head_idx]  # (B, 1, D, key_seq)
            attn_score = q @ k  # (B, 1, query_seq, key_seq)
            attn_score = attn_score.permute(0, 3, 1, 2)  # (B, key_seq, 1, query_seq)
            scores.append(attn_score)

        full_scores = torch.cat(scores, dim=2) * scale

        if attention_mask is not None:
            mask = attention_mask.permute(0, 3, 1, 2)
            full_scores = full_scores + mask

        full_scores = full_scores.softmax(dim=1)
        attn_per_head = full_scores.split(1, dim=2)

        weights = []
        for head_idx in range(n_heads):
            s = attn_per_head[head_idx].permute(0, 2, 3, 1)  # (B, 1, query_seq, key_seq)
            v = values[head_idx].permute(0, 2, 3, 1)  # (B, 1, key_seq, D)
            weight = s @ v  # (B, 1, query_seq, D)
            weight = weight.permute(0, 3, 1, 2)  # (B, D, 1, query_seq)
            weights.append(weight)

        return torch.cat(weights, dim=1)
