# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for diffusion pre-export torch quantization.

Every diffusion pipeline quantizes its weights before ``torch.export`` through coreai-opt,
rather than rewriting the exported Core AI program afterwards. These tests cover the
presets, the module exclusions the text encoders rely on, and the
guards around the quantizer's in-place behaviour.

CPU-only, toy modules. Nothing here downloads weights or runs an export.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from coreai_opt.quantization import QuantizerConfig

from coreai_models.diffusion.components import (
    FLUX2_COMPONENTS,
    FLUX2_MULTIFUNCTION_TRANSFORMER,
    SANA_SPRINT_COMPONENTS,
    WAN_COMPONENTS,
    quant_weight_owner,
)
from coreai_models.diffusion.pipeline import (
    _apply_component_precision,
    _check_quantization_precision,
    _pipeline_component_name,
    _quantize_component_weights,
    _resolve_component_precision,
    _resolve_compression,
    _resolve_palettization,
    _uses_int4,
)
from coreai_models.diffusion.presets import _EMBEDDING_SUBCLASSES, _MODULE_TYPE_EXCLUSIONS, PRESETS
from coreai_models.export.compression import quantize_pytorch_model


class _TwoLinears(nn.Module):
    """Divisible-by-32 shapes so per-block quantization applies to both layers."""

    def __init__(self, in_features: int = 256) -> None:
        super().__init__()
        self.a = nn.Linear(in_features, 512, bias=False)
        self.b = nn.Linear(512, 128, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(self.a(x))


class _ModelWrapper(nn.Module):
    """Minimal export-wrapper stand-in: stores the wrapped module as ``.model``."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.model = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


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


# Component names every supported diffusers pipeline registers.
_COMPONENTS = ["transformer", "text_encoder", "vae", "tokenizer", "scheduler"]


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


def _preset_configs() -> list[tuple[str, dict]]:
    """``(label, config)`` for every non-None config a preset holds, uniform or per component."""
    out = []
    for name, preset in PRESETS.items():
        if preset.get("config") is not None:
            out.append((name, preset["config"]))
        for component, config in preset.get("component_configs", {}).items():
            if config is not None:
                out.append((f"{name}[{component}]", config))
    return out


@pytest.mark.parametrize(
    ("label", "config"), _preset_configs(), ids=[c[0] for c in _preset_configs()]
)
def test_presets_are_weight_only_and_eager(label: str, config: dict) -> None:
    """Every preset quantizes weights only, in eager mode.

    Eager mode keeps the component wrapper as a live module, which multi-function export
    re-traces at each of its shapes.
    """
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


@pytest.mark.parametrize(
    ("label", "config"), _preset_configs(), ids=[c[0] for c in _preset_configs()]
)
def test_presets_satisfy_the_coreai_opt_schema(label: str, config: dict) -> None:
    """Catches a schema typo without downloading any weights."""
    QuantizerConfig.from_dict({"quantization_config": copy.deepcopy(config)})


def test_every_preset_has_exactly_one_config_shape() -> None:
    """``config``, ``component_configs`` and ``component_palettization`` are alternatives."""
    shapes = ("config", "component_configs", "component_palettization")
    for name, preset in PRESETS.items():
        assert sum(shape in preset for shape in shapes) == 1, name


def test_neural_engine_quantization_only_accompanies_palettization() -> None:
    """The GPU export rejects palettization presets, so it never sees this key."""
    for name, preset in PRESETS.items():
        if "component_quantization" in preset:
            assert "component_palettization" in preset, name


_PALETTIZATION_PRESETS = [n for n, p in PRESETS.items() if "component_palettization" in p]


@pytest.mark.parametrize("name", _PALETTIZATION_PRESETS)
def test_palettization_presets_satisfy_the_coreai_opt_schema(name: str) -> None:
    from coreai_opt.palettization import KMeansPalettizerConfig

    for config in PRESETS[name]["component_palettization"].values():
        KMeansPalettizerConfig.from_dict({"kmeans_palettization_config": copy.deepcopy(config)})


@pytest.mark.parametrize("name", [n for n, p in PRESETS.items() if "component_quantization" in p])
def test_neural_engine_quantization_satisfies_the_coreai_opt_schema(name: str) -> None:
    for config in PRESETS[name]["component_quantization"].values():
        QuantizerConfig.from_dict({"quantization_config": copy.deepcopy(config)})


def test_palettization_presets_are_neural_engine_only() -> None:
    """The GPU export would otherwise drop the palettization and export dense."""
    with pytest.raises(ValueError, match="neural-engine"):
        _resolve_palettization("te6bit-pal", neural_engine=False)
    assert set(_resolve_palettization("te6bit-pal", neural_engine=True)) == {"text_encoder"}


@pytest.mark.parametrize("compression", ["8bit", "t8bit-te4bit-asym"])
def test_neural_engine_export_rejects_quantization_presets(compression: str) -> None:
    with pytest.raises(ValueError, match="palettization preset"):
        _resolve_palettization(compression, neural_engine=True)


def test_neural_engine_export_accepts_none() -> None:
    assert _resolve_palettization("none", neural_engine=True) == {}


def test_neural_engine_preset_selects_the_neural_engine_registry(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from coreai_models.diffusion import export

    monkeypatch.setattr("sys.argv", ["export", "sana-sprint-0.6b-neural-engine", "--dry-run"])
    export.main()
    assert "neural_engine:     True" in capsys.readouterr().out


def test_neural_engine_export_rejects_other_pipelines(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from coreai_models.diffusion import export

    argv = ["export", "flux2-klein-4b", "--compute-unit", "neural-engine", "--dry-run"]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit):
        export.main()
    assert "only supported for Sana Sprint, not flux2" in capsys.readouterr().err


def test_int_scales_take_the_dtype_the_weights_are_quantized_in() -> None:
    """Int scales follow the weight dtype, which is why the mixed preset casts its text encoder.

    The Core AI GPU int4 dequant kernel rejects bf16 scales (``scale should be half or float``).
    """
    for dtype in (torch.bfloat16, torch.float16):
        model = _TwoLinears().to(dtype).eval()
        quantize_pytorch_model(
            model,
            (torch.randn(1, 4, 256, dtype=dtype),),
            None,
            copy.deepcopy(_int4_config()),
            0,
            (),
        )
        assert model.a.parametrizations["weight"][0].scale.dtype == dtype


def test_uniform_presets_keep_the_pipeline_precision() -> None:
    assert _resolve_component_precision("4bit", _COMPONENTS) == {}
    assert _resolve_component_precision("none", _COMPONENTS) == {}


def test_component_precision_in_json_is_kept_out_of_the_coreai_opt_config() -> None:
    """coreai-opt forbids unknown keys, so the quantization side must not see it."""
    value = json.dumps({**_int4_config(), "component_precision": {"transformer": "float32"}})
    assert _resolve_component_precision(value, _COMPONENTS) == {"transformer": "float32"}
    config = _resolve_compression(value, _COMPONENTS)["transformer"]
    assert config == _int4_config()
    QuantizerConfig.from_dict({"quantization_config": config})


def test_component_precision_rejects_unknown_components_and_precisions() -> None:
    with pytest.raises(ValueError, match="text_encodr"):
        _resolve_component_precision(
            json.dumps({"component_precision": {"text_encodr": "float16"}}), _COMPONENTS
        )
    with pytest.raises(ValueError, match="fp16"):
        _resolve_component_precision(
            json.dumps({"component_precision": {"text_encoder": "fp16"}}), _COMPONENTS
        )


def test_apply_component_precision_casts_only_the_named_components() -> None:
    pipe = SimpleNamespace(
        text_encoder=_TwoLinears().to(torch.bfloat16), transformer=_TwoLinears().to(torch.bfloat16)
    )
    _apply_component_precision(pipe, {"text_encoder": "float16"})
    assert pipe.text_encoder.a.weight.dtype == torch.float16
    assert pipe.transformer.a.weight.dtype == torch.bfloat16


def test_uses_int4_finds_4bit_specs_anywhere_in_a_config() -> None:
    assert _uses_int4(PRESETS["4bit"]["config"])
    assert _uses_int4(PRESETS["4bit-asym"]["config"])
    assert not _uses_int4(PRESETS["8bit"]["config"])
    mixed = PRESETS["t8bit-te4bit-asym"]["component_configs"]
    assert _uses_int4(mixed["text_encoder"]) and not _uses_int4(mixed["transformer"])
    nested = {
        "module_type_configs": {"x.Linear": {"op_state_spec": {"weight": {"dtype": "uint4"}}}}
    }
    assert _uses_int4(nested)


def test_int4_from_bf16_is_rejected_before_export() -> None:
    """It would export fine and abort at runtime: ``scale should be half or float``."""
    with pytest.raises(ValueError, match="text_encoder: INT4 weights quantized from bfloat16"):
        _check_quantization_precision(
            _int4_config(), _TwoLinears().to(torch.bfloat16), "text_encoder"
        )


@pytest.mark.parametrize(
    ("preset", "dtype"),
    [
        ("4bit", torch.float16),
        ("4bit", torch.float32),
        ("8bit", torch.bfloat16),
        ("none", torch.bfloat16),
    ],
)
def test_supported_precisions_pass_the_check(preset: str, dtype: torch.dtype) -> None:
    _check_quantization_precision(PRESETS[preset]["config"], _TwoLinears().to(dtype), "transformer")


def test_mixed_preset_passes_the_check_on_a_bf16_pipeline() -> None:
    """Sana exports in bf16: the preset's fp16 text encoder is what makes its int4 legal."""
    pipe = SimpleNamespace(
        text_encoder=_TwoLinears().to(torch.bfloat16), transformer=_TwoLinears().to(torch.bfloat16)
    )
    configs = _resolve_compression("t8bit-te4bit-asym", _COMPONENTS)
    with pytest.raises(ValueError, match="text_encoder"):
        _check_quantization_precision(configs["text_encoder"], pipe.text_encoder, "text_encoder")

    _apply_component_precision(pipe, _resolve_component_precision("t8bit-te4bit-asym", _COMPONENTS))
    for name in ("text_encoder", "transformer"):
        _check_quantization_precision(configs[name], getattr(pipe, name), name)


def test_mixed_preset_keeps_the_transformer_at_int8() -> None:
    """Int4 in the transformer visibly degrades detail and text; the text encoder tolerates it."""
    resolved = _resolve_compression("t8bit-te4bit-asym", _COMPONENTS)
    assert resolved["transformer"] == PRESETS["8bit"]["config"]
    assert resolved["text_encoder"] == PRESETS["4bit-asym"]["config"]
    assert resolved["vae"] is None


@pytest.mark.parametrize(
    ("fq_name", "why"),
    [
        ("diffusers.models.normalization.RMSNorm", "diffusers RMSNorm (qk-norm)"),
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
    assert _resolve_compression("none", _COMPONENTS) == dict.fromkeys(_COMPONENTS)
    assert _resolve_compression("4bit", _COMPONENTS) == dict.fromkeys(
        _COMPONENTS, PRESETS["4bit"]["config"]
    )


def test_resolve_compression_accepts_a_coreai_opt_json_config() -> None:
    """``--compression '<json>'`` takes the same shape the presets hold."""
    config = copy.deepcopy(PRESETS["4bit"]["config"])
    assert _resolve_compression(json.dumps(config), _COMPONENTS) == dict.fromkeys(
        _COMPONENTS, config
    )


def test_resolve_compression_keys_configs_by_component() -> None:
    """Listed components get their own config; unlisted ones stay dense."""
    int4, int8 = PRESETS["4bit"]["config"], PRESETS["8bit"]["config"]
    value = json.dumps({"components": {"transformer": int4, "text_encoder": int8}})
    resolved = _resolve_compression(value, _COMPONENTS)
    assert resolved["transformer"] == int4
    assert resolved["text_encoder"] == int8
    assert resolved["vae"] is None


def test_resolve_compression_rejects_an_unknown_component() -> None:
    """A typo would otherwise leave that component dense with no warning."""
    value = json.dumps({"components": {"transfomer": PRESETS["4bit"]["config"]}})
    with pytest.raises(ValueError, match="transfomer"):
        _resolve_compression(value, _COMPONENTS)


@pytest.mark.parametrize("name", [n for n, p in PRESETS.items() if "component_configs" in p])
def test_per_component_presets_name_real_components(name: str) -> None:
    """Every mixed preset only names components the pipelines actually have."""
    _resolve_compression(name, _COMPONENTS)


def test_pipeline_component_name_is_found_by_identity() -> None:
    """All FLUX.2 transformer variants wrap the one ``transformer``, so they share its config."""
    transformer, text_encoder = _TwoLinears(), _TwoLinears()
    pipe = SimpleNamespace(components={"transformer": transformer, "text_encoder": text_encoder})
    assert _pipeline_component_name(pipe, _ModelWrapper(transformer).model) == "transformer"
    assert _pipeline_component_name(pipe, text_encoder) == "text_encoder"
    with pytest.raises(ValueError, match="not a component"):
        _pipeline_component_name(pipe, _TwoLinears())


# ---------------------------------------------------------------------------
# Component wiring
# ---------------------------------------------------------------------------


def test_quant_weight_owner_returns_the_wrapped_model() -> None:
    """The owner is the wrapped model, not the export wrapper built around it."""
    inner = _TwoLinears()
    assert quant_weight_owner(_ModelWrapper(inner)) is inner


def test_quant_weight_owner_is_shared_across_wrappers_of_one_module() -> None:
    """Two specs over one module resolve to the same owner, which is what dedup keys on."""
    inner = _TwoLinears()
    assert quant_weight_owner(_ModelWrapper(inner)) is quant_weight_owner(_ModelWrapper(inner))


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


@pytest.mark.parametrize("registry", [WAN_COMPONENTS])
def test_denoisers_override_the_quant_trace(registry) -> None:
    """The WAN denoiser traces thousands of tokens at its export shape."""
    spec = registry["transformer"]
    assert spec.quant_dummy_fn is not None
    assert spec.quant_trace_fn() is not spec.dummy_fn


def test_vae_components_stay_unquantized() -> None:
    for registry in (FLUX2_COMPONENTS, SANA_SPRINT_COMPONENTS, WAN_COMPONENTS):
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

    for fq_name in (*_MODULE_TYPE_EXCLUSIONS, *_EMBEDDING_SUBCLASSES):
        module_name, _, class_name = fq_name.rpartition(".")
        module = importlib.import_module(module_name)
        assert hasattr(module, class_name), fq_name


class _Embed(nn.Module):
    def __init__(self, embedding: nn.Embedding) -> None:
        super().__init__()
        self.embed_tokens = embedding

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


def _gemma2_embedding() -> nn.Embedding:
    from transformers.models.gemma2.modeling_gemma2 import Gemma2TextScaledWordEmbedding

    return Gemma2TextScaledWordEmbedding(1000, 256, 0, embed_scale=16.0)


def _quantize_embedding(model: nn.Module, config: dict) -> None:
    ids = torch.randint(0, 1000, (1, 16))
    quantize_pytorch_model(model, (ids,), None, copy.deepcopy(config), 0, ())


def test_embedding_subclass_raises_without_the_override() -> None:
    """coreai-opt defaults the axis by exact type, so Sana's Gemma2 embedding gets none."""
    config = {**_int4_config(), "module_type_configs": _MODULE_TYPE_EXCLUSIONS}
    with pytest.raises(ValueError, match="unresolved axis"):
        _quantize_embedding(_Embed(_gemma2_embedding()).eval(), config)


@pytest.mark.parametrize(
    ("name", "scale_shape"), [("4bit", (1000, 8)), ("4bit-asym", (1000, 8)), ("8bit", (1000, 1))]
)
def test_embedding_subclass_matches_the_base_class_default(name: str, scale_shape: tuple) -> None:
    """The override reproduces what coreai-opt picks for a plain ``nn.Embedding``."""
    subclass = _Embed(_gemma2_embedding()).eval()
    base = _Embed(nn.Embedding(1000, 256)).eval()
    base.embed_tokens.weight.data.copy_(subclass.embed_tokens.weight.data)

    _quantize_embedding(subclass, PRESETS[name]["config"])
    _quantize_embedding(base, PRESETS[name]["config"])

    scale = subclass.embed_tokens.parametrizations["weight"][0].scale
    assert tuple(scale.shape) == scale_shape
    assert torch.equal(scale, base.embed_tokens.parametrizations["weight"][0].scale)
