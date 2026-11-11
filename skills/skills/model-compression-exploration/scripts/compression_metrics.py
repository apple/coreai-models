"""Compression metrics: theoretical size, average bitwidth, divisibility.

The pure-math helpers (``compute_compressed_size_mb``,
``compute_average_bitwidth``, ``check_divisibility``, ``fp16_baseline_mb``)
operate on plain ``LayerSpec`` objects and have no coreai-opt dependency.

``extract_layer_specs`` is the bridge to a coreai-opt-prepared model. See
its docstring for the dual extraction paths (quantizer-driven and
parametrize-walking).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class LayerSpec:
    """Describes how a single weight tensor is stored after compression.

    Attributes:
        name: Module name (used for reporting).
        weight_shape: Shape of the weight tensor.
        n_bits: Bits per stored value. 16 means the layer is left in fp16.
        granularity: One of "per_tensor", "per_channel", "per_block",
            "per_grouped_channel".
        axis: Channel axis the granularity is applied along.
        block_size: Block size for "per_block" granularity.
        group_size: Channels-per-group for "per_grouped_channel"
            granularity.
        has_zero_point: True for asymmetric quantization (adds an int
            zero-point per scale group).
        has_lut: True for palettization (adds a fp16 LUT per group).
        has_per_channel_scales: Palettization-only toggle that adds one
            fp16 scale per output channel.
    """

    name: str
    weight_shape: tuple
    n_bits: int
    granularity: str = "per_channel"
    axis: int = 0
    block_size: Optional[int] = None
    group_size: Optional[int] = None
    has_zero_point: bool = False
    has_lut: bool = False
    has_per_channel_scales: bool = False


def fp16_baseline_mb(model) -> float:
    """Total parameters × 2 bytes, in MB. Assumes fp16 baseline."""
    total = sum(p.numel() for p in model.parameters())
    return (total * 2) / (1024 * 1024)


def check_divisibility(model, axis: int, block_size: int) -> dict[str, tuple[int, int]]:
    """Return layers whose weight.shape[axis] is not divisible by block_size.

    Per-block quantization and per-grouped-channel palettization silently
    skip these layers. Run this before applying such configs and override
    the offending layers via ``set_module_name``.

    Returns:
        Dict mapping module_name -> (offending_dim, block_size).
    """
    incompatible: dict[str, tuple[int, int]] = {}
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            continue
        if axis >= weight.ndim or axis < -weight.ndim:
            # The requested axis does not exist on this weight (e.g. axis=1
            # on a 1-D LayerNorm/RMSNorm/BatchNorm weight), so there is no
            # dimension to block along and divisibility does not apply. Note
            # axis=0 exists on 1-D weights and is checked normally: grouped
            # palettization groups along axis 0 (see coreai_opt's
            # kmeans_palettizer_config presets) and can target them.
            continue
        dim = weight.shape[axis]
        if dim % block_size != 0:
            incompatible[name] = (dim, block_size)
    return incompatible


def _scale_groups(weight_shape: tuple, spec: LayerSpec) -> int:
    if spec.granularity == "per_tensor":
        return 1
    if spec.granularity == "per_channel":
        return weight_shape[spec.axis]
    if spec.granularity == "per_block":
        if spec.block_size is None:
            raise ValueError(f"per_block granularity requires block_size: {spec.name}")
        groups_along = math.ceil(weight_shape[spec.axis] / spec.block_size)
        other = math.prod(d for i, d in enumerate(weight_shape) if i != spec.axis)
        return groups_along * other
    if spec.granularity == "per_grouped_channel":
        if spec.group_size is None:
            raise ValueError(
                f"per_grouped_channel granularity requires group_size: {spec.name}"
            )
        return math.ceil(weight_shape[spec.axis] / spec.group_size)
    raise ValueError(f"Unknown granularity {spec.granularity!r}")


def compute_compressed_size_mb(
    specs: list[LayerSpec],
    extra_fp16_params: int = 0,
) -> float:
    """Theoretical compressed size in MB.

    Args:
        specs: One LayerSpec per compressed-or-uncompressed weight tensor.
        extra_fp16_params: Additional element count for biases, BN,
            LayerNorm parameters etc. that are not represented in ``specs``.

    Returns:
        Total size in MB, where one fp16 element = 2 bytes.
    """
    total_bits = 0
    for spec in specs:
        numel = 1
        for d in spec.weight_shape:
            numel *= d
        # Either weight bits (quant) or index bits (palett).
        total_bits += numel * spec.n_bits

        if spec.n_bits == 16:
            # Pure fp16, no overhead.
            continue

        n_groups = _scale_groups(spec.weight_shape, spec)
        # Scales are fp16.
        total_bits += n_groups * 16
        # Zero-points share the weight dtype (asymmetric quant only).
        if spec.has_zero_point:
            total_bits += n_groups * spec.n_bits
        # LUT for palettization: 2^n_bits fp16 centroids per group.
        if spec.has_lut:
            total_bits += n_groups * (2**spec.n_bits) * 16
        # Optional per-channel scale on palettization.
        if spec.has_per_channel_scales:
            total_bits += spec.weight_shape[spec.axis] * 16

    total_bits += extra_fp16_params * 16
    return total_bits / 8 / (1024 * 1024)


def compute_average_bitwidth(specs: list[LayerSpec]) -> float:
    """Weighted average bitwidth: sum(numel_i * bits_i) / sum(numel_i)."""
    total_weighted = 0
    total_numel = 0
    for spec in specs:
        numel = 1
        for d in spec.weight_shape:
            numel *= d
        total_weighted += numel * spec.n_bits
        total_numel += numel
    if total_numel == 0:
        return 16.0
    return total_weighted / total_numel


def compression_ratio(fp16_mb: float, compressed_mb: float) -> float:
    """fp16_mb / compressed_mb, guarded against divide-by-zero."""
    if compressed_mb <= 0:
        return 0.0
    return fp16_mb / compressed_mb


# ---------------------------------------------------------------------------
# Extractor — touches coreai-opt internals
# ---------------------------------------------------------------------------


def extract_layer_specs(prepared, *, quantizer=None) -> list[LayerSpec]:
    """Walk a coreai-opt-prepared model and return one LayerSpec per weight.

    Two extraction paths, both touching coreai-opt internals:

    1. **Compressor-driven** — pass ``quantizer=<Quantizer or KMeansPalettizer>``.
       If the compressor exposes ``_get_fake_quantize_modules()`` we use it
       (covers graph + eager quantization). Otherwise we fall through to (2).
    2. **Parametrize walk** — used when no compressor is passed, or when the
       compressor doesn't expose the FQ accessor (palettization). Reads
       dtype / granularity / n_bits off ``module.parametrizations.weight``.

    Disabled FQs (e.g., divisibility-skipped layers) are reported as fp16
    — that exposes silent skips that would otherwise inflate compression
    ratios.

    Raises:
        ValueError: ``prepared`` is a ``torch.fx.GraphModule`` but no
            compressor with ``_get_fake_quantize_modules`` was supplied —
            the parametrize walk would silently report fp16.
    """
    import torch.fx

    if quantizer is not None and hasattr(quantizer, "_get_fake_quantize_modules"):
        return _extract_via_quantizer(prepared, quantizer)

    if isinstance(prepared, torch.fx.GraphModule):
        raise ValueError(
            "extract_layer_specs got a torch.fx.GraphModule but no quantizer "
            "with _get_fake_quantize_modules() was provided. Pass "
            "quantizer=<Quantizer instance>, or use ExecutionMode.EAGER."
        )

    return _extract_via_parametrize(prepared)


def _extract_via_parametrize(prepared) -> list[LayerSpec]:
    """Walk ``module.parametrizations.weight`` to find FQ / palettize hooks."""
    import torch.nn.utils.parametrize as P

    specs: list[LayerSpec] = []
    for name, module in prepared.named_modules():
        weight = getattr(module, "weight", None)
        # Skip <module>.parametrizations submodules: their .weight is a
        # ParametrizationList, not a Tensor.
        if not isinstance(weight, torch.Tensor):
            continue
        if not P.is_parametrized(module, "weight"):
            specs.append(
                LayerSpec(name=name, weight_shape=tuple(weight.shape), n_bits=16)
            )
            continue
        param = next(iter(module.parametrizations.weight))
        specs.append(_spec_from_parametrize(name, weight.shape, param))
    return specs


def _extract_via_quantizer(prepared, quantizer) -> list[LayerSpec]:
    """Use ``quantizer._get_fake_quantize_modules()``; works for graph + eager."""
    fq_dict = quantizer._get_fake_quantize_modules()

    weight_fq: dict[str, object] = {}
    for module_name, fqs in fq_dict.items():
        for fq in fqs:
            target = getattr(fq, "quantization_target", None)
            if target is None or str(getattr(target, "value", target)) != "weight":
                continue
            if hasattr(fq, "is_disabled") and fq.is_disabled():
                continue  # silent skip → leave as fp16
            weight_fq[module_name] = fq

    name_to_shape = _resolve_weight_shapes(prepared, weight_fq.keys())

    specs: list[LayerSpec] = []
    for name, shape in name_to_shape.items():
        if name in weight_fq:
            specs.append(_spec_from_fake_quantize(name, shape, weight_fq[name]))
        else:
            specs.append(LayerSpec(name=name, weight_shape=shape, n_bits=16))
    return specs


def _resolve_weight_shapes(prepared, fq_names) -> dict[str, tuple]:
    """Map weight-bearing module names to shapes, falling back to
    ``named_parameters()`` for FQ-named layers not found via
    ``named_modules()`` (the graph-mode case)."""
    shapes: dict[str, tuple] = {}
    for name, module in prepared.named_modules():
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor):
            shapes[name] = tuple(weight.shape)

    missing = [n for n in fq_names if n not in shapes]
    if missing:
        param_lookup = dict(prepared.named_parameters())
        for name in missing:
            for suffix in (".weight", ""):
                if (key := f"{name}{suffix}") in param_lookup:
                    shapes[name] = tuple(param_lookup[key].shape)
                    break
    return shapes


def _spec_from_fake_quantize(name: str, weight_shape, fq) -> LayerSpec:
    """Translate a FakeQuantizeImplBase into a LayerSpec."""
    n_bits = getattr(fq, "n_bits", None) or _bits_from_dtype(fq.dtype)
    return LayerSpec(
        name=name,
        weight_shape=tuple(weight_shape),
        n_bits=n_bits,
        has_zero_point=str(getattr(fq, "qscheme", "")).lower().endswith("asymmetric"),
        **_granularity_fields(fq.granularity),
    )


def _spec_from_parametrize(name: str, weight_shape, param) -> LayerSpec:
    """Translate a parametrize entry into a LayerSpec.

    Quantization parametrizations expose ``qparams_calculator``;
    palettization ones expose ``n_bits`` directly.
    """
    weight_shape = tuple(weight_shape)
    if hasattr(param, "qparams_calculator"):
        calc = param.qparams_calculator
        return LayerSpec(
            name=name,
            weight_shape=weight_shape,
            n_bits=_bits_from_dtype(calc.dtype),
            has_zero_point=str(getattr(calc, "qscheme", ""))
            .lower()
            .endswith("asymmetric"),
            **_granularity_fields(calc.granularity),
        )
    if hasattr(param, "n_bits"):  # palettization
        return LayerSpec(
            name=name,
            weight_shape=weight_shape,
            n_bits=param.n_bits,
            has_lut=True,
            has_per_channel_scales=bool(
                getattr(param, "enable_per_channel_scale", False)
            ),
            **_granularity_fields(param.granularity),
        )
    raise ValueError(f"Unrecognized parametrization on {name}: {type(param).__name__}")


def _bits_from_dtype(dtype) -> int:
    import re

    match = re.search(r"(\d+)", str(dtype))
    if not match:
        raise ValueError(f"Cannot extract bit width from dtype {dtype!r}")
    return int(match.group(1))


def _granularity_fields(granularity) -> dict:
    """Return LayerSpec kwargs (granularity, axis, block_size, group_size)
    for a coreai-opt granularity object."""
    name = type(granularity).__name__
    axis = getattr(granularity, "axis", 0) or 0
    if name == "PerTensorGranularity":
        return {"granularity": "per_tensor"}
    if name == "PerChannelGranularity":
        return {"granularity": "per_channel", "axis": axis}
    if name == "PerBlockGranularity":
        return {
            "granularity": "per_block",
            "axis": axis,
            "block_size": granularity.block_size,
        }
    if name == "PerGroupedChannelGranularity":
        return {
            "granularity": "per_grouped_channel",
            "axis": axis,
            "group_size": granularity.group_size,
        }
    raise ValueError(f"Unknown granularity type {name!r}")
