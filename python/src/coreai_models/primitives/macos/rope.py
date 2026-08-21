# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import math
import os

import coreai_torch
import coreai_torch.composite_ops
import torch
from typing_extensions import Self


class RoPE(coreai_torch.composite_ops.RoPE):
    """Apply rotary positional embedding to input tensors."""

    def __init__(
        self: Self,
        scale: float = 1.0,
        base: float = 1e4,
        dims: int | None = None,
        interleaved: bool = False,
    ) -> None:
        _use_hf_impl = os.environ.get("USE_HF_IMPL", "False").lower() == "true"
        super().__init__(
            scale=scale,
            base=base,
            dims=dims,
            interleaved=interleaved,
            _use_hf_impl=_use_hf_impl,
        )


class DecomposedRoPE(torch.nn.Module):
    """Apply rotary positional embedding using raw torch ops (no composite op).

    This bypasses coreai_torch.composite_ops.RoPE entirely, implementing
    the rotation math with standard torch operations. Useful when the
    composite op's MLIR lowering is buggy for partial rotary embeddings.

    The math matches _rope_with_cos_and_sin_impl from coreai_torch exactly:
      inv_freq = 1 / (base ^ (arange(0, half_dim) / half_dim))
      angle = position_ids * inv_freq
      cos, sin = angle.cos(), angle.sin()
      y1 = cos * x1 - sin * x2
      y2 = sin * x1 + cos * x2
      output = cat(y1, y2, passthrough)
    """

    def __init__(
        self: Self,
        dims: int | None = None,
        base: float = 1e4,
    ) -> None:
        super().__init__()
        self.dims = dims
        self.base = base

    def forward(
        self: Self,
        input: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        offset: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Apply rotary positional embedding.

        Args:
            input: Tensor of shape (..., num_heads, seq_len, head_dim).
            position_ids: Tensor of shape (batch, seq_len) with position indices.
            offset: Scalar or tensor offset when position_ids is None.

        Returns:
            Tensor with RoPE applied to the first `dims` elements of head_dim.
        """
        embedding_dim = input.shape[-1]
        if self.dims is not None and self.dims < embedding_dim:
            rotation_dims = self.dims
        else:
            rotation_dims = embedding_dim
        half_dim = rotation_dims // 2

        # Compute position_ids if not provided
        if position_ids is not None:
            # position_ids: (batch, seq_len) -> (batch, 1, seq_len) for head broadcasting
            pos = position_ids.unsqueeze(1)
        else:
            q_len = input.shape[-2]
            if offset is not None and isinstance(offset, torch.Tensor):
                pos = offset.unsqueeze(-1).unsqueeze(-1) + torch.arange(q_len, device=input.device)
            else:
                int_offset = offset if offset is not None else 0
                pos = int_offset + torch.arange(q_len, device=input.device)

        pos = pos.float()

        # Compute inverse frequencies in f32: 1 / (base ^ (i / half_dim))
        exponent = torch.arange(half_dim, dtype=torch.float32, device=input.device) / half_dim
        inv_freq = 1.0 / torch.pow(self.base, exponent)

        # Compute angles: (batch, 1, seq_len, 1) * (half_dim,) -> (batch, 1, seq_len, half_dim)
        angle = pos.unsqueeze(-1) * inv_freq

        # Compute cos/sin in input dtype
        cos = angle.cos().to(input.dtype)
        sin = angle.sin().to(input.dtype)

        # Split input into two halves (non-interleaved)
        x1 = input[..., :half_dim]
        x2 = input[..., half_dim:rotation_dims]

        # Apply rotation
        y1 = cos * x1 - sin * x2
        y2 = sin * x1 + cos * x2

        # Concatenate rotated part and passthrough
        if rotation_dims < embedding_dim:
            return torch.cat((y1, y2, input[..., rotation_dims:]), dim=-1)
        return torch.cat((y1, y2), dim=-1)


class YarnRoPE(torch.nn.Module):
    def __init__(
        self: Self,
        dims: int,
        interleaved: bool = False,
        max_position_embeddings=2048,
        base=10000,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
        truncate: bool = True,
    ) -> None:
        super().__init__()

        def yarn_find_correction_dim(num_rotations):
            return (
                dims * math.log(original_max_position_embeddings / (num_rotations * 2 * math.pi))
            ) / (2 * math.log(base))

        def yarn_find_correction_range():
            low = yarn_find_correction_dim(beta_fast)
            high = yarn_find_correction_dim(beta_slow)
            if truncate:
                low = math.floor(low)
                high = math.ceil(high)
            return max(low, 0), min(high, dims - 1)

        def yarn_get_mscale(scale=1, mscale=1):
            if scale <= 1:
                return 1.0
            return 0.1 * mscale * math.log(scale) + 1.0

        def yarn_linear_ramp_mask(min_val, max_val, dim):
            if min_val == max_val:
                max_val += 0.001  # Prevent singularity

            linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
            return torch.clip(linear_func, 0, 1)

        # Initialize constants that aren't a part of state-dict on with cpu
        # device so that they don't get "faked" on meta device when initializing
        # model structure.
        with torch.device("cpu"):
            self.dims = dims
            self.mscale = yarn_get_mscale(scaling_factor, mscale) / yarn_get_mscale(
                scaling_factor, mscale_all_dim
            )
            freq_extra = base ** (torch.arange(0, dims, 2, dtype=torch.float32) / dims)
            freq_inter = scaling_factor * freq_extra
            low, high = yarn_find_correction_range()
            freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dims // 2)
            self._freqs = (freq_inter * freq_mask + freq_extra * (1 - freq_mask)) / (
                freq_inter * freq_extra
            )
            self._rope = RoPE(scale=1.0, interleaved=interleaved)

    def forward(
        self: Self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        offset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.mscale != 1.0:
            head_dim = x.shape[-1]
            message = "torch.export fails partial Yarn RoPE"
            torch._check(self.dims >= head_dim, message=message)
            # In principle the general formula that supports partial Yarn RoPE is
            #     x[..., : self.dims] = self.mscale * x[..., : self.dims]
            # In practice torch.export does not support partial sliced assignment,
            # so we apply mscale to the full tensor (full Yarn RoPE only).
            x = self.mscale * x
        return self._rope(
            x,
            position_ids=position_ids,
            freqs=self._freqs.to(x.device),
            offset=offset,
        )


class LongRoPE(torch.nn.Module):
    """LongRoPE: per-dimension frequency rescaling with attention scaling.

    Uses precomputed per-dimension factors (long_factor or short_factor) to
    rescale inv_freq, plus an attention_factor that scales the Q/K vectors
    before the dot product. Mirrors HF's _compute_longrope_parameters.
    """

    def __init__(
        self: Self,
        dims: int,
        base: float = 1e4,
        interleaved: bool = False,
        long_factor: list[float] | None = None,
        short_factor: list[float] | None = None,
        original_max_position_embeddings: int = 4096,
        max_position_embeddings: int = 131072,
        attention_factor: float | None = None,
        config_max_position_embeddings: int | None = None,
    ) -> None:
        super().__init__()
        # attention_factor is a model property derived from the config's full
        # context ratio, NOT the runtime context length.
        config_max = config_max_position_embeddings or max_position_embeddings
        factor = config_max / original_max_position_embeddings

        if attention_factor is None:
            if factor <= 1.0:
                attention_factor = 1.0
            else:
                attention_factor = math.sqrt(
                    1 + math.log(factor) / math.log(original_max_position_embeddings)
                )

        with torch.device("cpu"):
            self.dims = dims
            self.attention_factor = attention_factor

            if max_position_embeddings <= original_max_position_embeddings:
                factors = short_factor if short_factor is not None else long_factor
            else:
                factors = long_factor if long_factor is not None else short_factor
            ext_factors = torch.tensor(factors, dtype=torch.float32)
            inv_freq_shape = torch.arange(0, dims, 2, dtype=torch.float32) / dims
            inv_freq = 1.0 / (ext_factors * base**inv_freq_shape)
            self._freqs = inv_freq
            self._rope = RoPE(scale=1.0, dims=dims, interleaved=interleaved)

    def forward(
        self: Self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        offset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.attention_factor != 1.0:
            if self.dims < x.shape[-1]:
                x = torch.cat(
                    [self.attention_factor * x[..., : self.dims], x[..., self.dims :]],
                    dim=-1,
                )
            else:
                x = self.attention_factor * x
        return self._rope(
            x,
            position_ids=position_ids,
            freqs=self._freqs.to(x.device),
            offset=offset,
        )


def initialize_rope(
    dims: int | None = None,
    base: float = 1e4,
    interleaved: bool = False,
    scaling_config: dict | None = None,
    max_position_embeddings: int | None = None,
    original_max_position_embeddings: int | None = None,
    config_max_position_embeddings: int | None = None,
) -> torch.nn.Module:
    # When FORCE_DECOMPOSED_ROPE=1, bypass the composite op entirely and use
    # raw torch ops. This works around MLIR lowering bugs for partial rotary
    # (similar to the FLUX.2 inline RoPE corruption: rdar://178555985).
    if os.environ.get("FORCE_DECOMPOSED_ROPE") == "1":
        return DecomposedRoPE(dims=dims, base=float(base))

    if scaling_config is not None:
        rope_type = scaling_config.get("type") or scaling_config.get("rope_type", "default")
    else:
        rope_type = "default"

    rope: torch.nn.Module
    match rope_type:
        case "default" | "linear":
            scale = 1 / scaling_config["factor"] if rope_type == "linear" else 1.0
            rope = RoPE(scale=float(scale), base=float(base), dims=dims, interleaved=interleaved)

        case "yarn":
            if dims is None:
                msg = "dims is required for yarn rope"
                raise ValueError(msg)
            scaling_factor = scaling_config["factor"]
            rope_kwargs = {
                key: scaling_config[key]
                for key in [
                    "original_max_position_embeddings",
                    "beta_fast",
                    "beta_slow",
                    "mscale",
                    "mscale_all_dim",
                ]
                if key in scaling_config
            }
            # Default truncate=True preserves prior behavior for gemma3 / qwen3_next / etc.
            # gpt-oss sets truncate=False; match HF `_compute_yarn_parameters` (line 359 of
            # transformers/modeling_rope_utils.py).
            truncate = scaling_config.get("truncate", True)
            rope = YarnRoPE(
                dims,
                interleaved=interleaved,
                max_position_embeddings=max_position_embeddings,
                base=float(base),
                scaling_factor=float(scaling_factor),
                truncate=bool(truncate),
                **rope_kwargs,
            )

        case "longrope":
            if dims is None:
                msg = "dims is required for longrope"
                raise ValueError(msg)
            original_max_pos = (
                original_max_position_embeddings
                or scaling_config.get("original_max_position_embeddings")
                or 4096
            )
            rope = LongRoPE(
                dims,
                base=float(base),
                interleaved=interleaved,
                long_factor=scaling_config.get("long_factor"),
                short_factor=scaling_config.get("short_factor"),
                original_max_position_embeddings=original_max_pos,
                max_position_embeddings=max_position_embeddings or 131072,
                attention_factor=scaling_config.get("attention_factor"),
                config_max_position_embeddings=config_max_position_embeddings,
            )

        case _:
            msg = f"Unsupported RoPE type {rope_type}"
            raise ValueError(msg)

    return rope
