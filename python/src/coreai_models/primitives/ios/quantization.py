# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Quantization utilities for iOS export

Provides per-tensor quantization for embeddings and weights to achieve
better compression than global quantization.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def quantize_per_tensor(
    tensor: torch.Tensor,
    nbits=8,
    symmetric: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantizes a tensor to int8 format.

    Params:
        tensor: tensor to quantize
        nbits: Number of bits to use for the quantized element type, currently only support nbits=8
        symmetric: When `True`, the zero-point is fixed at 0 (Default: `True`)
        returns: The quantized tensor, scale, and zero point
    """
    assert nbits == 8, f"Currently only supports quantizing to 8 bits, received nbits={nbits}"
    dtype_min = -(2 ** (nbits - 1))
    dtype_max = 2 ** (nbits - 1) - 1

    # Determine which axes to reduce (all except the axis we're quantizing along)

    if symmetric:
        # Symmetric quantization: scale based on max absolute value
        abs_tensor = torch.abs(tensor)

        # Compute max along all axes except the quantization axis
        scale_tensor = abs_tensor.max()

        # Avoid division by zero
        scale_tensor = torch.clamp(scale_tensor, min=1e-6)

        # Scale = max_abs / max_quant_value
        scale = scale_tensor / dtype_max

        # Zero point is always 0 for symmetric quantization
        zero_point = torch.zeros_like(scale, dtype=torch.int8)
    else:
        # Asymmetric quantization
        max_tensor = tensor.max()
        min_tensor = tensor.min()

        # Scale = (max - min) / (dtype_max - dtype_min)
        scale = (max_tensor - min_tensor) / (dtype_max - dtype_min)
        scale = torch.clamp(scale, min=1e-6)

        # Zero point = round(dtype_min - min / scale)
        zero_point = torch.clamp(
            torch.round(dtype_min - min_tensor / scale), dtype_min, dtype_max
        ).to(torch.int8)

    # Quantize: Q = clamp(round(x / scale + zero_point), dtype_min, dtype_max)
    quantized = torch.clamp(torch.round(tensor / scale + zero_point), dtype_min, dtype_max).to(
        torch.int8
    )

    return quantized, scale.squeeze(), zero_point.squeeze()


@torch.library.custom_op("coreai::dequantize_per_tensor", mutates_args=[])
def dequantize_per_tensor(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Dequantize a per-tensor quantized tensor.

    Args:
        quantized: INT8 quantized tensor
        scale: Per-tensor scale factor
        zero_point: Per-tensor zero point (optional, for asymmetric)
        target_dtype: Target dtype for dequantized tensor

    Returns:
        Dequantized tensor in target_dtype
    """
    # Convert to float
    dequantized = quantized.to(torch.float32)

    # Subtract zero point if asymmetric
    if zero_point is not None:
        dequantized = dequantized - zero_point.to(torch.float32)

    # Scale
    dequantized = dequantized * scale

    return dequantized.to(target_dtype)


@dequantize_per_tensor.register_fake
def dequantize_per_tensor_meta(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor = None,
    target_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    return torch.zeros_like(quantized).to(target_dtype)


# ---------------------------------------------------------------------------
# CoreAI-native quantized weights (bit-exact passthrough of a pre-quantized
# checkpoint, e.g. Gemma4's "-qat-mobile-transformers" variants)
# ---------------------------------------------------------------------------

# Registers `torch.ops.coreai.constexpr_blockwise_shift_scale`, which
# `coreai_torch`'s converter lowers directly to the native `coreai.blockwise_shift_scale`
# op -- CoreAI's own per-channel/per-block affine quantized weight representation. Unlike
# this module's `quantize_per_tensor`/`dequantize_per_tensor` (per-tensor INT8 only), this
# op supports per-block scale of any rank and INT2/INT4/INT8 data, and the exported graph
# carries sub-byte (2/4-bit) data as a genuinely packed constant rather than a widened
# container -- see `coreai_models.models.ios.gemma4_text` for the export-time promotion
# that makes that packing happen.
import coreai_torch._compression.custom_layers as _native_quant_ops  # noqa: E402,F401

_NBITS_TO_TORCH_DTYPE: dict[int, torch.dtype] = {2: torch.int2, 4: torch.int4, 8: torch.int8}


def dequantize_native(
    quantized_data: torch.Tensor,
    scale: torch.Tensor,
    nbits: int,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize a symmetric, per-block-scaled CoreAI-native quantized tensor.

    ``quantized_data`` holds one signed logical value per element (unpacked -- see
    ``unpack_signed_subbyte``); ``scale`` must have the same rank as ``quantized_data``
    and each of its dimensions must divide the corresponding one of ``quantized_data``
    (CoreAI's blockwise convention: a whole dimension is one block when ``scale``'s size
    there is 1). No zero-point: Gemma4's QAT weights are all symmetric.
    """
    return torch.ops.coreai.constexpr_blockwise_shift_scale(
        quantized_data, scale, None, None, _NBITS_TO_TORCH_DTYPE[nbits], target_dtype
    )


def unpack_signed_subbyte(packed: torch.Tensor, nbits: int, unpacked_width: int) -> torch.Tensor:
    """Unpack a Gemma-QAT packed-uint8 tensor to one signed int8-container value per element.

    Mirrors ``transformers.integrations.gemma_quant``'s packing convention: ``8 // nbits``
    values per byte, low-order bits first, stored unsigned and shifted by ``2**(nbits-1)``
    to a signed range (e.g. nbits=4: unsigned [0, 15] -> signed [-8, 7]). ``nbits=8`` is a
    no-op (already one signed byte per element). Unpacking (rather than reusing the packed
    bytes as-is) is required because CoreAI's own native sub-byte packing convention
    (applied downstream by ``IntxTensor.from_unpacked``) is plain two's complement, not
    Gemma's offset encoding -- the two conventions store the same logical value in
    different bit patterns.
    """
    if nbits == 8:
        return packed.to(torch.int8)[..., :unpacked_width]
    values_per_byte = 8 // nbits
    mask = (1 << nbits) - 1
    bias = 1 << (nbits - 1)
    packed_u8 = packed.to(torch.uint8)
    parts = [
        ((packed_u8 >> (nbits * i)) & mask).to(torch.int8) - bias for i in range(values_per_byte)
    ]
    unpacked = torch.stack(parts, dim=-1).reshape(*packed_u8.shape[:-1], -1)
    return unpacked[..., :unpacked_width]


def unpack_unsigned_subbyte(packed: torch.Tensor, nbits: int, unpacked_width: int) -> torch.Tensor:
    """Unpack a Gemma-QAT packed-uint8 tensor to one unsigned LUT-index value per element
    (range ``[0, 2**nbits)``), for the palettization (``lut_to_dense``) path.

    Same packing convention as ``unpack_signed_subbyte`` (``8 // nbits`` values per byte,
    low-order bits first), but without the sign-shift: CoreAI's native palettization op
    consumes raw unsigned indices into a LUT, not signed values directly.
    """
    if nbits == 8:
        return packed.to(torch.uint8)[..., :unpacked_width]
    values_per_byte = 8 // nbits
    mask = (1 << nbits) - 1
    packed_u8 = packed.to(torch.uint8)
    parts = [(packed_u8 >> (nbits * i)) & mask for i in range(values_per_byte)]
    unpacked = torch.stack(parts, dim=-1).reshape(*packed_u8.shape[:-1], -1)
    return unpacked[..., :unpacked_width]


def dequantize_native_dense(
    dense_data: torch.Tensor, scale: torch.Tensor, target_dtype: torch.dtype
) -> torch.Tensor:
    """Apply a per-block/per-channel scale to an already-dense (not packed) tensor via
    CoreAI's native ``blockwise_shift_scale``.

    Unlike ``dequantize_native``, ``dense_data`` is a computed intermediate (e.g. the
    output of a LUT decode via ``coreai.lut_to_dense``), not a stored sub-byte constant --
    the density savings there come from the LUT *indices* being packed, not from this
    tensor -- so no ``input_dtype`` sub-byte declaration is needed or wanted here.
    """
    return torch.ops.coreai.constexpr_blockwise_shift_scale(
        dense_data, scale, None, None, None, target_dtype
    )
