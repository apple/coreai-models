# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for diffusion pre-export torch quantization.

Every diffusion pipeline quantizes its weights before ``torch.export`` through coreai-opt,
rather than rewriting the exported Core AI program afterwards. These tests cover the
presets, the module exclusions the text encoders and SD3's denoiser rely on, and the
guards around the quantizer's in-place behaviour.

CPU-only, toy modules. Nothing here downloads weights or runs an export.
"""

from __future__ import annotations

import copy
import json

import pytest
import torch
import torch.nn as nn
from coreai_opt.quantization import QuantizerConfig

from coreai_models.diffusion.components import (
    FLUX2_COMPONENTS,
    FLUX2_MULTIFUNCTION_TRANSFORMER,
    SD3_COMPONENTS,
    SD_COMPONENTS,
    WAN_COMPONENTS,
    TextEncoderWrapper,
    quant_weight_owner,
)
from coreai_models.diffusion.pipeline import _quantize_component_weights, _resolve_compression
from coreai_models.diffusion.presets import _MODULE_TYPE_EXCLUSIONS, PRESETS
from coreai_models.export.compression import quantize_pytorch_model


class _TwoLinears(nn.Module):
    """Divisible-by-32 shapes so per-block quantization applies to both layers."""

    def __init__(self, in_features: int = 256) -> None:
        super().__init__()
        self.a = nn.Linear(in_features, 512, bias=False)
        self.b = nn.Linear(512, 128, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(self.a(x))


class _MulNorm(nn.Module):
    """Stand-in for ``Qwen3RMSNorm``, a rank-1 ``weight`` consumed by ``torch.mul``."""

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x


class _NormThenLinear(nn.Module):
    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.norm = _MulNorm(dim)
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


def _int4_config() -> dict:
    return PRESETS["4bit"]["config"]


def _quantize(model: nn.Module, config: dict) -> nn.Module:
    """Call ``quantize_pytorch_model`` the way the diffusion pipeline does.

    The signature is shaped for the LLM path; the dynamic-shape and calibration arguments
    are only read when the config sets ``calibrate_activations``.
    """
    return quantize_pytorch_model(model, (torch.randn(1, 4, 256),), None, config, 0, ())


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [n for n, p in PRESETS.items() if p["config"] is not None])
def test_presets_are_weight_only_and_eager(name: str) -> None:
    """Every preset quantizes weights only, in eager mode.

    Eager mode keeps the component wrapper as a live module, which multi-function export
    re-traces at each of its shapes.
    """
    config = PRESETS[name]["config"]
    assert config["execution_mode"] == "eager"
    assert config["global_config"]["op_input_spec"] is None
    assert config["global_config"]["op_output_spec"] is None
    assert set(config["global_config"]["op_state_spec"]) == {"weight"}


def test_symmetric_presets_use_clipping_not_plain_symmetric() -> None:
    """The symmetric presets use ``symmetric_with_clipping``.

    The post-export MLIR pass used a signed range of [-7, 7] for int4. coreai-opt's
    ``symmetric`` covers the full [-8, 7], so ``symmetric_with_clipping`` is the match.
    Swapping the two changes every quantized weight.
    """
    symmetric = PRESETS["4bit"]["config"]
    asymmetric = PRESETS["4bit-asym"]["config"]

    def qscheme(cfg: dict) -> str:
        return str(cfg["global_config"]["op_state_spec"]["weight"]["qscheme"])

    assert qscheme(symmetric) == "symmetric_with_clipping"
    assert qscheme(asymmetric) == "asymmetric"


def _weight_scale_shape(preset: str) -> tuple[int, ...]:
    model = _TwoLinears().eval()
    _quantize(model, copy.deepcopy(PRESETS[preset]["config"]))
    return tuple(model.a.parametrizations["weight"][0].scale.shape)


def test_per_block_blocks_on_the_input_channel_axis() -> None:
    """The presets leave ``axis`` to coreai-opt, which blocks a Linear along axis 1."""
    # [out=512, in=256] weight, one scale per 32 input channels.
    assert _weight_scale_shape("4bit") == (512, 256 // 32)


def test_per_channel_scales_along_the_output_channel_axis() -> None:
    """The presets leave ``axis`` to coreai-opt, which scales a Linear along axis 0."""
    assert _weight_scale_shape("8bit") == (512, 1)


def test_int8_preset_is_no_longer_a_silent_no_op() -> None:
    """``8bit`` used to be inert, since the MLIR pass implemented int4 alone."""
    spec = PRESETS["8bit"]["config"]["global_config"]["op_state_spec"]["weight"]
    assert spec["dtype"] == "int8"


@pytest.mark.parametrize("name", [n for n, p in PRESETS.items() if p["config"] is not None])
def test_presets_satisfy_the_coreai_opt_schema(name: str) -> None:
    """Catches a schema typo without downloading any weights."""
    config = copy.deepcopy(PRESETS[name]["config"])
    QuantizerConfig.from_dict({"quantization_config": config})


@pytest.mark.parametrize(
    ("fq_name", "why"),
    [
        ("diffusers.models.normalization.RMSNorm", "SD3 qk-norm"),
        ("transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm", "FLUX.2 text encoder"),
        ("transformers.models.gemma2.modeling_gemma2.Gemma2RMSNorm", "Sana Sprint text encoder"),
        ("transformers.models.umt5.modeling_umt5.UMT5LayerNorm", "WAN text encoder"),
    ],
)
def test_required_module_exclusions_are_present(fq_name: str, why: str) -> None:
    """Each entry was added because an export failed without it.

    Dropping any one raises an ``unresolved axis=None`` error minutes into
    loading the real model, so pin them by name rather than trusting the dict's length.
    """
    assert _int4_config()["module_type_configs"][fq_name] is None, why


def test_resolve_compression_round_trips_preset_names() -> None:
    assert _resolve_compression("none") is None
    assert _resolve_compression("4bit") == PRESETS["4bit"]["config"]


def test_resolve_compression_accepts_a_coreai_opt_json_config() -> None:
    """``--compression '<json>'`` takes the same shape the presets hold."""
    config = copy.deepcopy(PRESETS["4bit"]["config"])
    assert _resolve_compression(json.dumps(config)) == config


# ---------------------------------------------------------------------------
# Component wiring
# ---------------------------------------------------------------------------


def test_quant_weight_owner_returns_the_wrapped_model() -> None:
    """The owner is the wrapped model, not the export wrapper built around it."""
    inner = _TwoLinears()
    assert quant_weight_owner(TextEncoderWrapper(inner)) is inner


def test_quant_weight_owner_is_shared_across_wrappers_of_one_module() -> None:
    """Two specs over one module resolve to the same owner, which is what dedup keys on."""
    inner = _TwoLinears()
    assert quant_weight_owner(TextEncoderWrapper(inner)) is quant_weight_owner(
        TextEncoderWrapper(inner)
    )


def test_quant_weight_owner_names_the_offender_when_the_convention_is_broken() -> None:
    """A quantizable wrapper storing its child elsewhere should say so plainly.

    The owner is found by attribute name, so a wrapper that deviates would otherwise fail
    with a bare AttributeError, or silently break dedup.
    """

    class _WrapperWithoutModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.denoiser = _TwoLinears()

    with pytest.raises(AttributeError, match="_WrapperWithoutModel.*`.model`"):
        quant_weight_owner(_WrapperWithoutModel())


@pytest.mark.parametrize(
    ("registry", "names"),
    [
        (SD_COMPONENTS, ("text_encoder", "unet")),
        (SD3_COMPONENTS, ("text_encoder", "text_encoder_2")),
        (WAN_COMPONENTS, ("text_encoder",)),
    ],
)
def test_cheap_components_reuse_their_export_dummies(registry, names) -> None:
    """Only the denoisers need a separate trace factory.

    A text encoder's export dummy is already a short sequence, so overriding it would add
    a function without saving anything.
    """
    for name in names:
        spec = registry[name]
        assert spec.quant_dummy_fn is None
        assert spec.quant_trace_fn() is spec.dummy_fn


@pytest.mark.parametrize("registry", [SD3_COMPONENTS, WAN_COMPONENTS])
def test_denoisers_override_the_quant_trace(registry) -> None:
    """The SD3 and WAN denoisers trace thousands of tokens at their export shape."""
    spec = registry["transformer"]
    assert spec.quant_dummy_fn is not None
    assert spec.quant_trace_fn() is not spec.dummy_fn


def test_vae_components_stay_unquantized() -> None:
    for registry in (FLUX2_COMPONENTS, SD_COMPONENTS, SD3_COMPONENTS, WAN_COMPONENTS):
        vaes = [name for name in registry if "vae" in name]
        assert vaes
        assert all(not registry[name].quantizable for name in vaes)


def test_multifunction_trace_defaults_to_a_function_variant() -> None:
    """``MultiFunctionComponentSpec`` carries ``functions``, so the resolver uses the first."""
    spec = FLUX2_MULTIFUNCTION_TRANSFORMER
    assert callable(spec.quant_trace_fn())


# ---------------------------------------------------------------------------
# The quantization helper
# ---------------------------------------------------------------------------


def test_graph_mode_is_rejected() -> None:
    """Graph mode returns a GraphModule, which would replace the component wrapper."""
    config = {**_int4_config(), "execution_mode": "graph"}
    with pytest.raises(ValueError, match="eager"):
        _quantize_component_weights(
            _TwoLinears(), _TwoLinears(), (torch.randn(1, 4, 256),), config, set()
        )


def test_quantization_is_skipped_for_an_already_quantized_weight_owner() -> None:
    """The eight FLUX.2 transformer specs share one module, so only the first quantizes.

    Eager finalize zeroes the dense weight, so a second pass would quantize an empty
    placeholder.
    """
    owner = _TwoLinears()
    wrapper = _TwoLinears()
    already_done = {id(owner)}
    _quantize_component_weights(
        wrapper, owner, (torch.randn(1, 4, 256),), _int4_config(), already_done
    )
    assert not hasattr(wrapper.a, "parametrizations")


def test_quantization_marks_the_weight_owner_as_done() -> None:
    model = _TwoLinears().eval()
    quantized: set[int] = set()
    _quantize_component_weights(model, model, (torch.randn(1, 4, 256),), _int4_config(), quantized)
    assert quantized == {id(model)}
    assert hasattr(model.a, "parametrizations")


def test_quantize_pytorch_model_rewrites_weights_in_place() -> None:
    """coreai-opt eager mode mutates the module, which is what lets the specs share weights."""
    model = _TwoLinears().eval()
    returned = _quantize(model, _int4_config())
    assert returned is model
    assert hasattr(model.a, "parametrizations")


def test_rank_one_weight_raises_without_the_module_exclusion() -> None:
    """``_MODULE_TYPE_EXCLUSIONS`` covers this case.

    ``torch.mul`` is a registered op and the parameter is named ``weight``, so the global
    per-block spec matches a rank-1 tensor. coreai-opt has no default axis for ``torch.mul``,
    so it raises instead of skipping the weight.
    """
    config = {**_int4_config(), "module_type_configs": {}}
    with pytest.raises(ValueError, match="unresolved axis"):
        _quantize(_NormThenLinear().eval(), config)


def test_rank_one_weight_is_tolerated_with_the_module_exclusion() -> None:
    fq_name = f"{_MulNorm.__module__}.{_MulNorm.__qualname__}"
    config = {**_int4_config(), "module_type_configs": {fq_name: None}}
    model = _NormThenLinear().eval()
    _quantize(model, config)
    assert hasattr(model.linear, "parametrizations")
    assert not hasattr(model.norm, "parametrizations")


def test_exclusions_are_keyed_by_importable_class_paths() -> None:
    """A typo in a fully-qualified name would be a silent no-op, so check that each resolves."""
    import importlib

    for fq_name in _MODULE_TYPE_EXCLUSIONS:
        module_name, _, class_name = fq_name.rpartition(".")
        module = importlib.import_module(module_name)
        assert hasattr(module, class_name), fq_name
