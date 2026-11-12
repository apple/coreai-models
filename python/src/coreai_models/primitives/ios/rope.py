# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import os

import torch


def rotate_half(x: torch.Tensor, dim: int = -1, ndim: int = 1) -> torch.Tensor:
    """GPT-NeoX rotate-half along ``dim``: ``[x1, x2] -> [-x2, x1]``.

    ``ndim > 1`` splits ``dim`` into ``ndim`` equal parts (one per spatial axis) and rotates each
    in one concat: ``[-x1, x0, -x3, x2, ...]``.
    """
    lead = (slice(None),) * (dim % x.dim())
    half = x.shape[dim] // (2 * ndim)
    parts = []
    for i in range(ndim):
        a = x[(*lead, slice(2 * i * half, (2 * i + 1) * half))]
        b = x[(*lead, slice((2 * i + 1) * half, (2 * i + 2) * half))]
        parts += [-b, a]
    return torch.cat(parts, dim=dim)


@torch.library.custom_op("coreai::rope_gather_cached_cos_sin", mutates_args=[])
def rope_gather_cached_cos_sin(
    position_ids: torch.Tensor, cos_cached: torch.Tensor, sin_cached: torch.Tensor
) -> list[torch.Tensor]:
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
    head_axis: int | None = 1,
    ndim: int = 1,
    dim: int = -1,
    fused: bool = False,
) -> torch.Tensor:
    # `head_axis` is where the (broadcast) head dimension lives in `x`. The
    # default of 1 matches the `(B, n_heads, S, head_dim)` head-format layout.
    # Callers that keep Q/K in `(B, S, n_heads, head_dim)` pass `head_axis=2` so
    # cos/sin broadcast over heads without an extra transpose into head format.
    # `head_axis=None` takes cos/sin that already carry the head axis.
    # `dim` is the rotation axis; channels-first (BC1S) layouts such as
    # `(B, n_heads, head_dim, S)` pass `dim=2`.
    if head_axis is not None:
        rope_cos = rope_cos.unsqueeze(head_axis)
        rope_sin = rope_sin.unsqueeze(head_axis)

    torch._check(len(rope_cos.shape) == 4)
    torch._check(len(rope_sin.shape) == 4)

    if ndim > 1:
        torch._check(x.shape[dim] % ndim == 0)

    if ndim == 1 or fused:
        # Apply rotary position embedding. `fused` emits one rotate-half concat
        # and one full-width multiply-add instead of `ndim` per-part ones; each
        # element sees the same mul and add, so values are identical and only the
        # graph shrinks.
        return (x * rope_cos) + (rotate_half(x, dim, ndim) * rope_sin)

    # N-dimensional RoPE (Gemma4 vision's 2D grid RoPE, HF's
    # `apply_multidimensional_rope`): split `dim` into `ndim` equal parts
    # (one spatial axis each) and apply standard rotate-half RoPE within each part
    # independently, using that part's own cos/sin slice, then concatenate back.
    # `rope_cos`/`rope_sin` carry every axis's frequencies pre-concatenated along
    # the same dim (baked once per axis at load time), so splitting them the
    # same way as `x` pairs each axis with its own slice.
    per = x.shape[dim] // ndim
    x_parts = x.split(per, dim=dim)
    cos_parts = rope_cos.split(per, dim=dim)
    sin_parts = rope_sin.split(per, dim=dim)
    out = [
        (xp * cp) + (rotate_half(xp, dim) * sp)
        for xp, cp, sp in zip(x_parts, cos_parts, sin_parts, strict=True)
    ]
    return torch.cat(out, dim=dim)


def rope_inv_freq(dims: int, base: float, device: torch.device | None = None) -> torch.Tensor:
    """RoPE inverse frequencies ``1 / base ** (arange(0, dims, 2) / dims)``, ``[dims // 2]``
    float32. Bit-identical to the default frequencies of ``RoPE`` and ``RoPECache``."""
    return 1.0 / (base ** (torch.arange(0, dims, 2, dtype=torch.float32, device=device) / dims))


def rope_cos_sin(
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    dtype: torch.dtype,
    ndim: int = 1,
    full: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin of ``positions * inv_freq``, computed in float32 and cast to ``dtype``.

    ``positions`` is ``[...]`` for ``ndim == 1`` or ``[..., ndim]`` (one coordinate per axis);
    positions pass through as given, so padding coordinates such as -1 stay -1. ``inv_freq``
    ``[F]`` is upcast to float32 as given, so a buffer cast with the model keeps its rounding.
    Returns ``[..., F]`` for ``ndim == 1``, the half-dim layout ``RoPE``'s ``cos``/``sin`` take,
    or ``[..., 2F]`` with ``full``, the layout ``apply_rope`` takes. Axial tables (``ndim > 1``)
    need ``full``: ``[a0, a0, a1, a1, ...]``, one doubled ``[F]`` block per axis.
    """
    if ndim > 1 and not full:
        raise ValueError("axial RoPE tables (ndim > 1) need full=True")
    angles = positions.float().unsqueeze(-1) * inv_freq.to(positions.device, torch.float32)
    if ndim > 1:
        angles = torch.cat([angles[..., i, :] for i in range(ndim) for _ in range(2)], dim=-1)
    elif full:
        angles = torch.cat((angles, angles), dim=-1)
    return angles.cos().to(dtype), angles.sin().to(dtype)


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
        """Gather the cached cos/sin values using the position_ids"""
        return rope_gather_cached_cos_sin(position_ids, self.cos_cached, self.sin_cached)
