# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import os
from collections.abc import Sequence

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    GPT NeoX style: rotates [repeat] half the hidden dims of the input.
    for sin [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
    """

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 : x.shape[-1]]
    return torch.cat((-x2, x1), dim=-1)


@torch.library.custom_op("coreai::rope_gather_cached_cos_sin", mutates_args=[])
def rope_gather_cached_cos_sin(
    position_ids: torch.Tensor, cos_cached: torch.Tensor, sin_cached: torch.Tensor
) -> list[torch.Tensor]:
    """Gather cached cos/sin by position for in-graph RoPE.

    The exported composite survives exactly one lookup per graph, and only when
    position_ids is the graph's own position input; a second lookup, or a derived
    operand, SIGSEGVs delegate specialization. Multi-position (M-RoPE) models
    (Qwen2/3-VL) need a lookup per row, so precompute cos/sin on the host
    (gather_cos_sin_host / mrope_cos_sin) and feed them as rope_cos/rope_sin inputs.
    Runtime composite limitation; see #280.
    """
    position_ids = position_ids.to(torch.int32)
    rope_cos = cos_cached[position_ids]
    rope_sin = sin_cached[position_ids]
    return rope_cos, rope_sin


@rope_gather_cached_cos_sin.register_fake
def _fake(
    position_ids: torch.Tensor, cos_cached: torch.Tensor, sin_cached: torch.Tensor
) -> list[torch.Tensor]:
    position_ids = position_ids.to(torch.int32)
    rope_cos = cos_cached[position_ids]
    rope_sin = sin_cached[position_ids]
    return rope_cos, rope_sin


def apply_rope(
    x: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    head_axis: int = 1,
) -> torch.Tensor:
    # head_axis is where the broadcast head dim lives: 1 for (B, heads, S, dim),
    # 2 for (B, S, heads, dim) -- lets cos/sin broadcast without a transpose.
    rope_cos = rope_cos.unsqueeze(head_axis)
    rope_sin = rope_sin.unsqueeze(head_axis)

    torch._check(len(rope_cos.shape) == 4)
    torch._check(len(rope_sin.shape) == 4)

    # Apply rotary position embedding
    return (x * rope_cos) + (rotate_half(x) * rope_sin)


# On iOS, it is more efficient to compute RoPE using precomputed and cached cos/sin values
class RoPECache(torch.nn.Module):
    """
    RoPE module.

    Paper reference: https://arxiv.org/abs/2104.09864
    """

    def __init__(
        self,
        head_dim: int,
        max_cache_size: int,
        base: float = 500_000,
    ) -> None:
        super().__init__()
        self._head_dim = head_dim
        self._max_cache_size = max_cache_size
        self._base = base
        self._use_hf_impl = os.environ.get("USE_HF_IMPL", "False").lower() == "true"
        self._compute_sin_and_cos()

    def _apply(self, fn):
        # The `.to()` function implicitly calls into this function,
        # and we should recompute the cos / sin rather than just do
        # a simple cast.
        super()._apply(fn)
        # Read dtype/device from the buffer post-apply so device-only
        # and dtype-only .to(...) calls are both honored.
        new_dtype = self.cos_cached.dtype
        new_device = self.cos_cached.device
        self._compute_sin_and_cos(new_dtype)
        self.cos_cached = self.cos_cached.to(new_device)
        self.sin_cached = self.sin_cached.to(new_device)
        return self

    def _compute_sin_and_cos(self, dtype: torch.dtype = torch.float32) -> None:
        head_dim = self._head_dim
        max_cache_size = self._max_cache_size
        base = self._base

        with torch.device("cpu"):
            theta = 1.0 / (
                base
                ** (torch.arange(start=0, end=head_dim, step=2, dtype=torch.float32) / head_dim)
            )

            if self._use_hf_impl:
                theta = theta.to(dtype)
                theta = theta.float()

            # Create position index [0, 1, ..., seq_len -1]
            seq_idx = torch.arange(end=max_cache_size, dtype=torch.int32)

            # Calculate product of position index and theta
            freqs = seq_idx[:, None] * theta

            # Cache cos sin values
            emb = torch.concatenate((freqs, freqs), dim=-1)
            self.cos_cached = torch.nn.Buffer(torch.cos(emb).to(dtype=dtype), persistent=False)
            self.sin_cached = torch.nn.Buffer(torch.sin(emb).to(dtype=dtype), persistent=False)

    def gather_cos_sin(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather the cached cos/sin values using the position_ids.

        This goes through the ``coreai::rope_gather_cached_cos_sin`` custom op, which
        lowers to the in-graph gather composite. That composite tolerates only one
        lookup per graph, fed straight from the graph's position input -- see the
        warning on :func:`rope_gather_cached_cos_sin`. For multi-position (M-RoPE)
        models, or any case needing more than one lookup, do the lookup on the host via
        :meth:`gather_cos_sin_host` / :meth:`mrope_cos_sin` and pass cos/sin in as graph
        inputs.
        """
        return rope_gather_cached_cos_sin(position_ids, self.cos_cached, self.sin_cached)

    def gather_cos_sin_host(
        self, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Host-side cos/sin gather (plain indexing), bypassing the in-graph composite.

        Precompute rope_cos/rope_sin outside the graph and feed them as inputs -- the
        supported pattern for M-RoPE models (a lookup per time/height/width row), where
        the in-graph composite survives only one lookup (see rope_gather_cached_cos_sin).
        Shaped like gather_cos_sin.
        """
        position_ids = position_ids.to(torch.int32)
        return self.cos_cached[position_ids], self.sin_cached[position_ids]

    def mrope_cos_sin(
        self,
        rows: Sequence[torch.Tensor],
        section_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Combine M-RoPE position rows into one cos/sin on the host.

        ``rows`` is one position_ids per section (time/height/width); ``section_masks``
        is ``(len(rows), head_dim)``, row ``s`` being 1 on the head-dim entries that take
        their position from ``rows[s]``. The masks let one path serve both contiguous
        (Qwen2-VL) and interleaved (Qwen3.5) layouts, and collapse to ordinary 1-D RoPE
        for text-only prompts. Host-only (via gather_cos_sin_host), shaped like gather_cos_sin.
        """
        if len(rows) != section_masks.shape[0]:
            msg = f"expected {section_masks.shape[0]} position rows, got {len(rows)}"
            raise ValueError(msg)
        cos_total: torch.Tensor | None = None
        sin_total: torch.Tensor | None = None
        for row, mask in zip(rows, section_masks, strict=True):
            row_cos, row_sin = self.gather_cos_sin_host(row)
            mask = mask.to(row_cos.dtype)
            cos_total = row_cos * mask if cos_total is None else cos_total + row_cos * mask
            sin_total = row_sin * mask if sin_total is None else sin_total + row_sin * mask
        assert cos_total is not None and sin_total is not None
        return cos_total, sin_total
