# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Diffusion export pipeline orchestration.

Exports a HuggingFace diffusion model to a set of Core AI .aimodel files — one
per component — plus tokenizer files and a pipeline.json descriptor.

Supports:
- FLUX.2 Klein (DiT-based)
- Wan 2.1 (text-to-video)
- Sana Sprint (few-step linear-attention DiT)
"""

import copy
import inspect
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download

from coreai_models._constants import DEFAULT_INCLUDE_DEBUG_INFO
from coreai_models.converter import export_stateless
from coreai_models.diffusion.components import (
    MultiFunctionComponentSpec,
    NeuralEngineComponentSpec,
    get_component_registry,
    quant_weight_owner,
)
from coreai_models.diffusion.gpu import export_multifunction
from coreai_models.diffusion.models import get_pipeline_type, unknown_pipeline_type_error
from coreai_models.diffusion.presets import PRESETS, list_presets
from coreai_models.export.compression import is_compression_mode_graph, quantize_pytorch_model
from coreai_models.export.metadata import build_aimodel_metadata

logger = logging.getLogger(__name__)


_PRECISIONS = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@dataclass
class DiffusionExportConfig:
    """Configuration for a diffusion model export."""

    hf_model_id: str
    output_dir: str = "outputs"
    components: list[str] | None = None
    compute_precision: str = "float16"
    compression: str = "none"
    overwrite: bool = False
    vae_tile_size: int | None = None
    include_debug_info: bool = DEFAULT_INCLUDE_DEBUG_INFO
    multifunction: bool = True
    # FLUX.2 only: one symbolic "main" per asset. Takes precedence over multifunction if True.
    dynamic: bool = False
    # "gpu", or "neural-engine" for the re-authored float16 Neural Engine components.
    compute_unit: str = "gpu"


def export_diffusion(config: DiffusionExportConfig) -> dict[str, str]:
    """Export diffusion model components to Core AI format.

    Args:
        config: Export configuration.

    Returns:
        Dict mapping component name to its .aimodel path.
    """
    model_dtype = _PRECISIONS.get(config.compute_precision, torch.float32)
    neural_engine = config.compute_unit == "neural-engine"
    if neural_engine and config.compression != "none":
        raise ValueError("--compute-unit neural-engine supports --compression none only")

    # 1. Determine pipeline type and load HF pipeline
    pipeline_type = get_pipeline_type(config.hf_model_id)
    if config.dynamic and pipeline_type != "flux2":
        raise ValueError(f"dynamic export is only supported for FLUX.2, not {pipeline_type}.")
    # Neural Engine components fold constants into float32 weights, then set their own dtypes.
    load_dtype = torch.float32 if neural_engine else model_dtype
    hf_pipe = _load_hf_pipeline(config.hf_model_id, pipeline_type, load_dtype)

    registry = get_component_registry(
        hf_pipe,
        pipeline_type=pipeline_type,
        multifunction=config.multifunction,
        dynamic=config.dynamic,
        compute_unit=config.compute_unit,
    )
    component_names = config.components or list(registry.keys())
    logger.info(f"Pipeline type: {pipeline_type}, components: {component_names}")

    quant_configs = _resolve_compression(config.compression, list(hf_pipe.components))
    _apply_component_precision(
        hf_pipe, _resolve_component_precision(config.compression, list(hf_pipe.components))
    )
    # Each spec's config, checked before any export so a bad combination fails without
    # leaving a half-written bundle.
    spec_configs: dict[str, dict] = {}
    if any(c is not None for c in quant_configs.values()):
        for name in component_names:
            spec = registry.get(name)
            if spec is None or isinstance(spec, NeuralEngineComponentSpec) or not spec.quantizable:
                continue
            owner = quant_weight_owner(spec.wrapper_fn(hf_pipe))
            component = _pipeline_component_name(hf_pipe, owner)
            component_config = quant_configs[component]
            if component_config is not None:
                _check_quantization_precision(component_config, owner, component)
                spec_configs[name] = component_config

    # Output goes to <output_dir>/<model-name>/. A dynamic export reuses the static asset
    # names, so it gets its own directory rather than overwriting the static bundle; so does
    # a Neural Engine export (<model-name>_neural_engine/).
    model_subdir = config.hf_model_id.split("/")[-1]
    if config.dynamic:
        model_subdir += "-dynamic"
    if neural_engine:
        model_subdir += "_neural_engine"
    output_path = Path(config.output_dir) / model_subdir
    output_path.mkdir(parents=True, exist_ok=True)

    # 2. Export each component
    results: dict[str, str] = {}

    # Modules already quantized. Several specs can share one, and only the first pass
    # sees dense weights, since eager finalize frees them in place.
    quantized_modules: set[int] = set()

    for name in component_names:
        if name not in registry:
            logger.warning(f"Unknown component '{name}', skipping. Valid: {list(registry.keys())}")
            continue

        spec = registry[name]
        asset_path = output_path / f"{spec.asset_name}.aimodel"

        if asset_path.exists() and not config.overwrite:
            logger.info(f"Skipping {name}: {asset_path} exists (use --overwrite)")
            results[name] = str(asset_path)
            continue

        if isinstance(spec, NeuralEngineComponentSpec):
            from coreai_models.embedding.embedding_gemma2.ios import convert_ane_functions

            logger.info(f"Exporting {name} -> {spec.asset_name}.aimodel (Neural Engine)")
            program = convert_ane_functions(
                [
                    (
                        "main",
                        spec.build_fn(hf_pipe),
                        spec.dummy_fn(hf_pipe),
                        spec.input_names,
                        spec.output_names,
                    )
                ],
                include_debug_info=config.include_debug_info,
            )
            _save_program(program, asset_path, output_path, config.hf_model_id, spec.asset_name)
            results[name] = str(asset_path)
            continue

        wrapper = spec.wrapper_fn(hf_pipe)

        # Quantize weights here, before export.
        quant_config = spec_configs.get(name)
        if quant_config is not None:
            logger.info(f"Quantizing {name}...")
            _quantize_component_weights(
                wrapper,
                quant_weight_owner(wrapper),
                spec.quant_trace_fn()(hf_pipe),
                quant_config,
                quantized_modules,
            )

        if isinstance(spec, MultiFunctionComponentSpec):
            logger.info(
                f"Exporting {name} -> {spec.asset_name}.aimodel "
                f"(multi-function: {[f.name for f in spec.functions]})"
            )
            functions = [(fv.name, wrapper, fv.dummy_fn(hf_pipe)) for fv in spec.functions]
            program = export_multifunction(
                functions,
                spec.input_names,
                spec.output_names,
                include_debug_info=config.include_debug_info,
            )
        else:
            logger.info(f"Exporting {name} -> {spec.asset_name}.aimodel")

            dummy_kwargs: dict[str, Any] = {}
            if "vae" in name and config.vae_tile_size is not None:
                dummy_kwargs["tile_size"] = config.vae_tile_size
            dummy_inputs = spec.dummy_fn(hf_pipe, **dummy_kwargs)
            dynamic_shapes = spec.dynamic_shapes_fn() if spec.dynamic_shapes_fn else None

            program = export_stateless(
                wrapper,
                dummy_inputs,
                spec.input_names,
                spec.output_names,
                dynamic_shapes=dynamic_shapes,
                include_debug_info=config.include_debug_info,
            )

        _save_program(program, asset_path, output_path, config.hf_model_id, spec.asset_name)
        del program
        results[name] = str(asset_path)

    # 3. Save sidecar assets (tokenizer, BN stats, etc.)
    if pipeline_type == "flux2":
        _save_flux2_sidecar_assets(hf_pipe, output_path, overwrite=config.overwrite)
    elif pipeline_type == "sana_sprint":
        _save_pipeline_tokenizer(hf_pipe, output_path, overwrite=config.overwrite)
    else:
        _save_tokenizer(config.hf_model_id, output_path, hf_pipe, overwrite=config.overwrite)

    # 4. Write pipeline.json
    _write_metadata_json(
        hf_pipe,
        config.hf_model_id,
        pipeline_type,
        output_path,
        config.compression,
        results,
        vae_tile_size=config.vae_tile_size,
        dynamic=config.dynamic,
        neural_engine_components=[
            name for name in results if isinstance(registry[name], NeuralEngineComponentSpec)
        ],
    )

    # Summary
    logger.info("=== Export Summary ===")
    for name, path in results.items():
        logger.info(f"  {name}: {path}")

    return results


def _save_program(
    program: Any, asset_path: Path, output_path: Path, model_id: str, asset_name: str
) -> None:
    if asset_path.exists():
        shutil.rmtree(asset_path)
    logger.info(f"Saving {asset_name} to {asset_path}...")
    program.save_asset(asset_path, build_aimodel_metadata(model_id, component=asset_name))

    # Clean up leftover .mlirb from older export runs
    mlirb_path = output_path / f"{asset_name}.mlirb"
    if mlirb_path.exists():
        mlirb_path.unlink()
    logger.info(f"Exported {asset_name} -> {asset_path}")


# ---------------------------------------------------------------------------
# HF pipeline loading
# ---------------------------------------------------------------------------


def _load_hf_pipeline(model_id: str, pipeline_type: str, model_dtype: torch.dtype) -> Any:
    """Load the appropriate HuggingFace pipeline based on type."""
    logger.info(f"Loading {model_id} (type={pipeline_type}, dtype={model_dtype})...")

    if pipeline_type == "flux2":
        from diffusers import Flux2KleinPipeline

        hf_pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=model_dtype)
        return hf_pipe

    if pipeline_type == "sana_sprint":
        from diffusers import SanaSprintPipeline

        return SanaSprintPipeline.from_pretrained(model_id, torch_dtype=model_dtype)

    if pipeline_type == "wan":
        from diffusers import WanPipeline

        hf_pipe = WanPipeline.from_pretrained(model_id, torch_dtype=model_dtype)
        return hf_pipe

    raise unknown_pipeline_type_error(pipeline_type)


# ---------------------------------------------------------------------------
# Sidecar assets
# ---------------------------------------------------------------------------


def _save_flux2_sidecar_assets(hf_pipe: Any, output_path: Path, overwrite: bool) -> None:
    """Save FLUX.2-specific sidecar files: tokenizer + VAE batch norm stats."""
    # Tokenizer
    tok_dir = output_path / "tokenizer"
    if tok_dir.exists() and not overwrite:
        logger.info(f"Skipping tokenizer: {tok_dir} exists (use --overwrite)")
    else:
        logger.info("Saving tokenizer...")
        try:
            if tok_dir.exists():
                shutil.rmtree(tok_dir)
            hf_pipe.tokenizer.save_pretrained(str(tok_dir))

            # Patch tokenizer class (Qwen2 -> GPT2) for swift-transformers compatibility
            for cfg_name in ("tokenizer_config.json", "config.json"):
                cfg_file = tok_dir / cfg_name
                if cfg_file.exists():
                    cfg = json.loads(cfg_file.read_text())
                    if cfg.get("tokenizer_class") in ("Qwen2Tokenizer", "Qwen2TokenizerFast"):
                        cfg["tokenizer_class"] = "GPT2Tokenizer"
                        cfg_file.write_text(json.dumps(cfg, indent=2))

            # Ensure config.json exists (some tokenizers only write tokenizer_config.json)
            tok_config = tok_dir / "tokenizer_config.json"
            config_json = tok_dir / "config.json"
            if tok_config.exists() and not config_json.exists():
                shutil.copy2(tok_config, config_json)

            logger.info(f"Saved tokenizer to {tok_dir}")
        except Exception as e:
            logger.warning(f"Could not save tokenizer: {e}")

    # VAE batch norm statistics
    try:
        bn = hf_pipe.vae.bn
        np.save(output_path / "vae_bn_mean.npy", bn.running_mean.float().cpu().numpy())
        np.save(output_path / "vae_bn_var.npy", bn.running_var.float().cpu().numpy())
        logger.info("Saved VAE batch norm stats")
    except Exception as e:
        logger.warning(f"Could not save VAE BN stats: {e}")


def _save_pipeline_tokenizer(hf_pipe: Any, output_path: Path, overwrite: bool) -> None:
    """Save the loaded tokenizer as `tokenizer/`, loaded by swift-transformers `AutoTokenizer`."""
    tok_dir = output_path / "tokenizer"
    if tok_dir.exists() and not overwrite:
        logger.info(f"Skipping tokenizer: {tok_dir} exists (use --overwrite)")
        return
    if tok_dir.exists():
        shutil.rmtree(tok_dir)
    hf_pipe.tokenizer.save_pretrained(str(tok_dir))
    logger.info(f"Saved tokenizer to {tok_dir}")


def _save_tokenizer(model_id: str, output_path: Path, hf_pipe: Any, overwrite: bool) -> None:
    """Save the tokenizer subdirs the model needs.

    Downloads `tokenizer/` from the
    HF snapshot and copies it alongside the exported assets.
    """
    for subdir in ["tokenizer"]:
        dst_dir = output_path / subdir
        if dst_dir.exists() and not overwrite:
            logger.info(f"Skipping {subdir}: {dst_dir} exists (use --overwrite)")
            continue

        logger.info(f"Saving {subdir}...")
        try:
            try:
                model_dir = Path(
                    snapshot_download(
                        model_id,
                        allow_patterns=[f"{subdir}/*"],
                    )
                )
            except Exception:
                model_dir = Path(snapshot_download(model_id, allow_patterns=[f"{subdir}/*"]))

            src_dir = model_dir / subdir
            if not src_dir.exists():
                logger.warning(f"No {subdir}/ subfolder found in downloaded model")
                continue

            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)
            logger.info(f"Saved {subdir} to {dst_dir}")
        except Exception as e:
            logger.warning(f"Could not save {subdir}: {e}")


# ---------------------------------------------------------------------------
# metadata.json (v0.2 schema — aligned with LLM and segmenter bundles)
# ---------------------------------------------------------------------------

METADATA_VERSION = "0.2"


def _prepare_assets(json_path: Path, exported_assets: dict[str, str]) -> dict[str, str]:
    """Asset map for the manifest: this run's exports merged over the previous export's.

    A partial run (--components) only knows what it just built, so without the merge it
    would drop the rest of the bundle. Prior entries whose files are gone are dropped.
    """
    output_path = json_path.parent
    assets: dict[str, str] = {}
    if json_path.exists():
        try:
            with open(json_path) as f:
                prior_assets = json.load(f).get("assets")
        except (OSError, json.JSONDecodeError):
            logger.warning(f"Ignoring unreadable {json_path}; rebuilding the asset list")
            prior_assets = None
        if isinstance(prior_assets, dict):
            assets = {
                name: filename
                for name, filename in prior_assets.items()
                if (output_path / str(filename)).exists()
            }
    preserved = [name for name in assets if name not in exported_assets]
    if preserved:
        logger.info(f"Preserving previously exported assets: {sorted(preserved)}")
    for name, path_str in exported_assets.items():
        assets[name] = Path(path_str).name
    return assets


def _prepare_compute_units(
    json_path: Path, exported_assets: dict[str, str], neural_engine_components: list[str]
) -> dict[str, str]:
    """``compute_units`` for the manifest: this run's components over the previous export's."""
    units: dict[str, str] = {}
    if json_path.exists():
        try:
            with open(json_path) as f:
                prior = json.load(f).get("diffusion", {}).get("compute_units")
        except (OSError, json.JSONDecodeError, AttributeError):
            prior = None
        if isinstance(prior, dict):
            units = {k: v for k, v in prior.items() if k not in exported_assets}
    for name in exported_assets:
        if name in neural_engine_components:
            units[name] = "neural_engine"
    return units


def _write_metadata_json(
    hf_pipe: Any,
    model_id: str,
    pipeline_type: str,
    output_path: Path,
    compression: str,
    exported_assets: dict[str, str],
    *,
    vae_tile_size: int | None = None,
    dynamic: bool = False,
    neural_engine_components: list[str] | None = None,
) -> None:
    """Write metadata.json with the v0.2 bundle schema for diffusion models.

    ``diffusion.compute_units`` maps each Neural Engine component to ``"neural_engine"``; the
    runtime loads components it does not list on the GPU. Entries from a previous partial
    export of the same bundle are kept while their asset still exists.
    """
    from datetime import datetime

    if pipeline_type == "flux2":
        diffusion_config = _build_flux2_config(hf_pipe, model_id, dynamic=dynamic)
    elif pipeline_type == "sana_sprint":
        diffusion_config = _build_sana_sprint_config(hf_pipe)
    elif pipeline_type == "wan":
        diffusion_config = _build_wan_config(hf_pipe, model_id, vae_tile_size=vae_tile_size)
    else:
        raise unknown_pipeline_type_error(pipeline_type)

    json_path = output_path / "metadata.json"
    compute_units = _prepare_compute_units(
        json_path, exported_assets, neural_engine_components or []
    )
    assets = _prepare_assets(json_path, exported_assets)
    if compute_units:
        diffusion_config["compute_units"] = compute_units

    metadata = {
        "metadata_version": METADATA_VERSION,
        "kind": "video-diffusion" if pipeline_type == "wan" else "diffusion",
        "name": output_path.name,
        "assets": assets,
        "diffusion": diffusion_config,
        "source": {
            "model_definition": "torch",
            "hf_model_id": model_id,
        },
        "compression": compression if compression != "none" else None,
        "compilation": {
            "date": datetime.now().astimezone().isoformat(),
            "targets": [],
        },
    }

    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata.json to {json_path}")


def _build_flux2_config(hf_pipe: Any, model_id: str, *, dynamic: bool = False) -> dict:
    vae_config = hf_pipe.vae.config
    transformer_config = hf_pipe.transformer.config

    vae_scale_power = len(vae_config.block_out_channels) - 1
    vae_spatial_scale = 2**vae_scale_power
    default_sample_size = getattr(transformer_config, "default_sample_size", 64)
    image_size = default_sample_size * vae_spatial_scale * 2

    scaling_factor = getattr(vae_config, "scaling_factor", 1.0)
    shift_factor = getattr(vae_config, "shift_factor", 0.0)
    batch_norm_eps = getattr(vae_config, "batch_norm_eps", 1e-5)
    guidance_embeds = getattr(transformer_config, "guidance_embeds", True)
    axes_dims_rope = list(getattr(transformer_config, "axes_dims_rope", [32, 32, 32, 32]))
    rope_theta = getattr(transformer_config, "rope_theta", 2000.0)

    config = {
        "type": "flux2",
        "prediction_type": "flow_matching",
        "encoder_scale_factor": scaling_factor,
        "decoder_scale_factor": scaling_factor,
        "decoder_shift_factor": shift_factor,
        "batch_norm_eps": batch_norm_eps,
        "guidance_embeds": guidance_embeds,
        "image_size": image_size,
        "default_guidance_scale": 1.0,
        "default_steps": 4,
        "rope_axes_dims": axes_dims_rope,
        "rope_theta": rope_theta,
    }
    if dynamic:
        # Read from the Dim bounds so the runtime can reject an unsupported size or
        # reference count up front instead of failing inside the graph.
        from coreai_models.diffusion.flux2 import (
            FLUX2_IMAGE_SIZE_MULTIPLE,
            FLUX2_IMG_SEQ_MAX,
            FLUX2_MAX_IMAGE_SIZE,
            FLUX2_MIN_IMAGE_SIZE,
            FLUX2_MIN_REFERENCE_SIZE,
        )

        config["dynamic_shapes"] = True
        config["min_image_size"] = FLUX2_MIN_IMAGE_SIZE
        config["max_image_size"] = FLUX2_MAX_IMAGE_SIZE
        config["image_size_multiple"] = FLUX2_IMAGE_SIZE_MULTIPLE
        config["max_image_tokens"] = FLUX2_IMG_SEQ_MAX
        config["min_reference_size"] = FLUX2_MIN_REFERENCE_SIZE
    return config


def _build_sana_sprint_config(hf_pipe: Any) -> dict:
    from coreai_models.diffusion.sana import (
        TEXT_SEQUENCE_LENGTH,
        sana_prompt_prefix,
        sana_text_input_length,
    )

    vae_config = hf_pipe.vae.config
    vae_spatial_scale = 2 ** (len(vae_config.encoder_block_out_channels) - 1)
    # SanaSprintPipeline.__call__ defaults; the runtime converts these TrigFlow angles to σ.
    call_defaults = inspect.signature(type(hf_pipe).__call__).parameters

    return {
        "type": "sana-sprint",
        "prediction_type": "flow_matching",
        "decoder_scale_factor": vae_config.scaling_factor,
        "image_size": hf_pipe.transformer.config.sample_size * vae_spatial_scale,
        "default_guidance_scale": call_defaults["guidance_scale"].default,
        "default_steps": call_defaults["num_inference_steps"].default,
        "max_timesteps": call_defaults["max_timesteps"].default,
        "intermediate_timesteps": call_defaults["intermediate_timesteps"].default,
        "prompt_prefix": sana_prompt_prefix(),
        "text_input_length": sana_text_input_length(hf_pipe),
        "text_sequence_length": TEXT_SEQUENCE_LENGTH,
    }


def _build_wan_config(hf_pipe: Any, model_id: str, *, vae_tile_size: int | None = None) -> dict:
    cfg = hf_pipe.transformer.config
    config = {
        "type": "wan2.1",
        "prediction_type": "flow_matching",
        "num_attention_heads": cfg.num_attention_heads,
        "attention_head_dim": cfg.attention_head_dim,
        "text_dim": cfg.text_dim,
        "z_dim": cfg.in_channels,
        "patch_size": list(cfg.patch_size) if hasattr(cfg, "patch_size") else [1, 2, 2],
        "default_steps": 50,
        "default_guidance_scale": 5.0,
        "default_shift": 3.0,
        "default_num_frames": 81,
        "default_fps": 16,
        "spatial_compression": 8,
        "temporal_compression": 4,
    }
    if vae_tile_size is not None:
        config["vae_tile_size"] = vae_tile_size
        config["vae_temporal_frames"] = 5
    return config


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------


def _resolve_compression(compression: str, component_names: list[str]) -> dict[str, dict | None]:
    """Resolve a compression string to a quantization config per pipeline component.

    A preset holds either one ``config`` for every component or ``component_configs`` keyed
    by diffusers component name. A JSON value is a coreai-opt config, or
    ``{"components": {...}}`` for the keyed form. Unlisted components stay dense.

    Args:
        compression: Preset name or JSON string.
        component_names: The pipeline's component names, e.g. ``hf_pipe.components``.
    """
    if compression in PRESETS:
        preset = PRESETS[compression]
        per_component = preset.get("component_configs")
        uniform = preset.get("config")
    else:
        try:
            parsed: dict = json.loads(compression)
        except (json.JSONDecodeError, TypeError) as e:
            available = ", ".join(list_presets())
            raise ValueError(
                f"Unknown compression value '{compression}'. "
                f"Expected a preset name ({available}) or a JSON config dict."
            ) from e
        parsed.pop("component_precision", None)  # read by `_resolve_component_precision`
        per_component = parsed.get("components")
        uniform = None if per_component is not None else parsed

    if per_component is None:
        return {name: uniform for name in component_names}

    # A misspelled name would leave that component silently dense.
    unknown = sorted(set(per_component) - set(component_names))
    if unknown:
        raise ValueError(
            f"Compression names unknown pipeline components {unknown}. "
            f"Valid: {sorted(component_names)}."
        )
    return {name: per_component.get(name) for name in component_names}


def _resolve_component_precision(compression: str, component_names: list[str]) -> dict[str, str]:
    """Per-component compute precision overrides a compression preset declares.

    A preset's ``component_precision`` (or the same key in a JSON value) overrides
    ``compute_precision`` for the named components; others keep the pipeline's precision.

    Args:
        compression: Preset name or JSON string.
        component_names: The pipeline's component names, e.g. ``hf_pipe.components``.
    """
    if compression in PRESETS:
        overrides = PRESETS[compression].get("component_precision", {})
    else:
        try:
            overrides = json.loads(compression).get("component_precision", {})
        except (json.JSONDecodeError, TypeError, AttributeError):
            overrides = {}  # `_resolve_compression` reports a malformed value

    unknown = sorted(set(overrides) - set(component_names))
    if unknown:
        raise ValueError(
            f"Precision names unknown pipeline components {unknown}. "
            f"Valid: {sorted(component_names)}."
        )
    invalid = {name: p for name, p in overrides.items() if p not in _PRECISIONS}
    if invalid:
        raise ValueError(f"Unknown precisions {invalid}. Valid: {sorted(_PRECISIONS)}.")
    return dict(overrides)


def _apply_component_precision(hf_pipe: Any, overrides: dict[str, str]) -> None:
    """Cast the named pipeline components in place, before quantization reads them."""
    for name, precision in overrides.items():
        logger.info(f"Casting {name} to {precision}")
        getattr(hf_pipe, name).to(_PRECISIONS[precision])


def _pipeline_component_name(hf_pipe: Any, weight_owner: torch.nn.Module) -> str:
    """The diffusers component name of ``weight_owner``, found by identity."""
    for name, module in hf_pipe.components.items():
        if module is weight_owner:
            return str(name)
    raise ValueError(
        f"{type(weight_owner).__name__} is not a component of {type(hf_pipe).__name__}"
    )


def _uses_int4(config: Any) -> bool:
    """Whether any spec in a coreai-opt config quantizes to a 4-bit integer dtype."""
    if not isinstance(config, dict):
        return False
    return any(
        (key == "dtype" and value in ("int4", "uint4")) or _uses_int4(value)
        for key, value in config.items()
    )


def _check_quantization_precision(
    quantization_config: dict | None, module: torch.nn.Module, component: str
) -> None:
    """Reject int4 weights quantized from bf16, which export fine but abort at runtime.

    Int scales take the weight dtype, and the Core AI GPU int4 dequant kernel asserts
    ``scale should be half or float`` on bf16 ones.
    """
    if quantization_config is None or not _uses_int4(quantization_config):
        return
    dtype = next((p.dtype for p in module.parameters() if p.is_floating_point()), None)
    if dtype == torch.bfloat16:
        raise ValueError(
            f"{component}: INT4 weights quantized from bfloat16 get bfloat16 scales, which the "
            "Core AI GPU int4 dequant kernel rejects at runtime. Export it in float16 with "
            f'`component_precision` (e.g. {{"{component}": "float16"}}), use an INT8 config for '
            "it, or pick a preset that does either (e.g. t8bit-te4bit-asym)."
        )


def _quantize_component_weights(
    wrapper: torch.nn.Module,
    weight_owner: torch.nn.Module,
    trace_inputs: tuple,
    quantization_config: dict,
    quantized_modules: set[int],
) -> None:
    """Quantize a component's weights in place before torch.export.

    Args:
        wrapper: The export wrapper, quantized in place.
        weight_owner: The shared module holding the weights, used as the identity key.
        trace_inputs: Positional inputs for the shape-discovery forward.
        quantization_config: A coreai-opt ``quantization_config`` dict.
        quantized_modules: Identity keys of weight owners already quantized, updated here.
    """
    if id(weight_owner) in quantized_modules:
        logger.info("  weights already quantized by an earlier component; reusing")
        return

    # Graph mode finalizes to an fx.GraphModule, which would replace the wrapper and its
    # forward signature. Multi-function export also re-traces one live module at eight
    # shapes, and a GraphModule is frozen at one.
    if is_compression_mode_graph(quantization_config):
        raise ValueError(
            "Diffusion pre-export quantization requires execution_mode='eager'. Graph "
            "mode returns an fx.GraphModule, which breaks the component wrappers and "
            "multi-function export."
        )

    # `quantize_pytorch_model` is shaped for the LLM path, where the trace carries KV
    # caches and activation calibration needs to know their length. Diffusion has neither,
    # and the arguments below are only read when the config sets `calibrate_activations`.
    quantize_pytorch_model(
        wrapper,
        trace_inputs,
        None,  # dynamic_shapes: eager mode ignores it
        copy.deepcopy(quantization_config),  # rewritten in place; presets are shared
        0,  # cache_seq_len: no KV cache
        (),  # state_indices: no states to reset between samples
    )
    quantized_modules.add(id(weight_owner))
