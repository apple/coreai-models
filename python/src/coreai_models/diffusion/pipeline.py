# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Diffusion export pipeline orchestration.

Exports a HuggingFace diffusion model to a set of Core AI .aimodel files — one
per component — plus tokenizer files and a pipeline.json descriptor.

Supports:
- Stable Diffusion 1.x / 2.x (UNet-based)
- Stable Diffusion 3.x (MMDiT, T5-less)
- FLUX.2 Klein (DiT-based, pre-computed RoPE)
"""

import asyncio
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from huggingface_hub import snapshot_download

from coreai_models.diffusion.components import get_component_registry
from coreai_models.diffusion.gpu import export_stateless
from coreai_models.diffusion.models import get_pipeline_type
from coreai_models.diffusion.presets import PRESETS, list_presets
from coreai_models.export.compiler import (
    apply_mlir_quantization,
)
from coreai_models.export.metadata import build_aimodel_metadata

logger = logging.getLogger(__name__)


@dataclass
class DiffusionExportConfig:
    """Configuration for a diffusion model export."""

    hf_model_id: str
    output_dir: str = "outputs"
    components: list[str] | None = None
    compute_precision: str = "float16"
    compression: str = "none"
    overwrite: bool = False


def export_diffusion(config: DiffusionExportConfig) -> dict[str, str]:
    """Export diffusion model components to Core AI format.

    Args:
        config: Export configuration.

    Returns:
        Dict mapping component name to its .aimodel path.
    """
    return asyncio.run(_async_export_diffusion(config))


async def _async_export_diffusion(config: DiffusionExportConfig) -> dict[str, str]:
    precision_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    model_dtype = precision_map.get(config.compute_precision, torch.float32)

    # 1. Determine pipeline type and load HF pipeline
    pipeline_type = get_pipeline_type(config.hf_model_id)
    hf_pipe = _load_hf_pipeline(config.hf_model_id, pipeline_type, model_dtype)

    registry = get_component_registry(hf_pipe, pipeline_type=pipeline_type)
    component_names = config.components or list(registry.keys())
    logger.info(f"Pipeline type: {pipeline_type}, components: {component_names}")

    # Resolve compression preset
    quant_config = _resolve_compression(config.compression)

    # Output goes to <output_dir>/<model-name>/
    model_subdir = config.hf_model_id.split("/")[-1]
    output_path = Path(config.output_dir) / model_subdir
    output_path.mkdir(parents=True, exist_ok=True)

    # 2. Export each component
    results: dict[str, str] = {}
    for name in component_names:
        if name not in registry:
            logger.warning(f"Unknown component '{name}', skipping. Valid: {list(registry.keys())}")
            continue

        spec = registry[name]
        mlirb_path = output_path / f"{spec.asset_name}.mlirb"
        asset_path = output_path / f"{spec.asset_name}.aimodel"

        if asset_path.exists() and not config.overwrite:
            logger.info(f"Skipping {name}: {asset_path} exists (use --overwrite)")
            results[name] = str(asset_path)
            continue

        logger.info(f"Exporting {name} -> {spec.asset_name}.aimodel")

        wrapper = spec.wrapper_fn(hf_pipe)
        dummy_inputs = spec.dummy_fn(hf_pipe)

        program = export_stateless(wrapper, dummy_inputs, spec.input_names, spec.output_names)

        # Optional MLIR quantization
        component_quant = quant_config if spec.quantizable else None
        if component_quant is not None:
            logger.info(f"Quantizing {name}...")
            program = await apply_mlir_quantization(program, component_quant)

        if asset_path.exists():
            shutil.rmtree(asset_path)
        logger.info(f"Saving {name} to {asset_path}...")
        metadata = build_aimodel_metadata(config.hf_model_id, component=spec.asset_name)
        program.save_asset(asset_path, metadata)
        del program

        # Clean up leftover .mlirb from older export runs
        if mlirb_path.exists():
            mlirb_path.unlink()

        results[name] = str(asset_path)
        logger.info(f"Exported {name} -> {asset_path}")

    # 3. Save sidecar assets (tokenizer, BN stats, etc.)
    if pipeline_type == "flux2":
        _save_flux2_sidecar_assets(hf_pipe, output_path, overwrite=config.overwrite)
    else:
        _save_tokenizer(config.hf_model_id, output_path, hf_pipe, overwrite=config.overwrite)

    # 4. Write pipeline.json
    _write_metadata_json(
        hf_pipe, config.hf_model_id, pipeline_type, output_path, config.compression, results
    )

    # Summary
    logger.info("=== Export Summary ===")
    for name, path in results.items():
        logger.info(f"  {name}: {path}")

    return results


# ---------------------------------------------------------------------------
# HF pipeline loading
# ---------------------------------------------------------------------------


def _load_hf_pipeline(model_id: str, pipeline_type: str, model_dtype: torch.dtype) -> Any:
    """Load the appropriate HuggingFace pipeline based on type."""
    logger.info(f"Loading {model_id} (type={pipeline_type}, dtype={model_dtype})...")

    if pipeline_type == "flux2":
        from diffusers import Flux2KleinPipeline

        hf_pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=model_dtype)
        # Text encoder needs float32 for token embedding precision
        hf_pipe.text_encoder = hf_pipe.text_encoder.float()
        return hf_pipe

    if pipeline_type == "sd3":
        from diffusers import StableDiffusion3Pipeline

        try:
            # T5-less path: skip text_encoder_3 / tokenizer_3 entirely. Quality cost
            # accepted; T5 can be added later by removing these kwargs.
            hf_pipe = StableDiffusion3Pipeline.from_pretrained(
                model_id,
                torch_dtype=model_dtype,
                text_encoder_3=None,
                tokenizer_3=None,
            )
            hf_pipe.vae = hf_pipe.vae.float()
            return hf_pipe
        except OSError as e:
            if "gated" in str(e).lower() or "access to model" in str(e).lower():
                raise PermissionError(
                    f"Access denied: {model_id} is a gated model. "
                    f"Accept the license at https://huggingface.co/{model_id} "
                    f"and run: hf auth login"
                ) from e
            raise

    from diffusers import StableDiffusionPipeline

    try:
        return StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=model_dtype,
            safety_checker=None,
        )
    except OSError as e:
        if "gated" in str(e).lower() or "access to model" in str(e).lower():
            raise PermissionError(
                f"Access denied: {model_id} is a gated model. "
                f"Accept the license at https://huggingface.co/{model_id} "
                f"and run: hf auth login"
            ) from e
        raise


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


def _save_tokenizer(model_id: str, output_path: Path, hf_pipe: Any, overwrite: bool) -> None:
    """Save the tokenizer subdirs the model needs.

    SD 1.x/2.x: just `tokenizer/`. SD3: also `tokenizer_2/` (CLIP-G). T5
    (`tokenizer_3/`) is skipped — paired with the T5-less load in
    `_load_hf_pipeline`.
    """
    subdirs = ["tokenizer"]
    if hasattr(hf_pipe, "tokenizer_2") and getattr(hf_pipe, "tokenizer_2", None) is not None:
        subdirs.append("tokenizer_2")

    for subdir in subdirs:
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
                        local_files_only=True,
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


def _write_metadata_json(
    hf_pipe: Any,
    model_id: str,
    pipeline_type: str,
    output_path: Path,
    compression: str,
    exported_assets: dict[str, str],
) -> None:
    """Write metadata.json with the v0.2 bundle schema for diffusion models."""
    from datetime import datetime

    if pipeline_type == "flux2":
        diffusion_config = _build_flux2_config(hf_pipe, model_id)
    else:
        diffusion_config = _build_sd_config(hf_pipe, model_id, pipeline_type)

    # Build assets map from exported component paths
    assets: dict[str, str] = {}
    for name, path_str in exported_assets.items():
        assets[name] = Path(path_str).name

    metadata = {
        "metadata_version": METADATA_VERSION,
        "kind": "diffusion",
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

    json_path = output_path / "metadata.json"
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata.json to {json_path}")


def _build_flux2_config(hf_pipe: Any, model_id: str) -> dict:
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

    return {
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


def _build_sd_config(hf_pipe: Any, model_id: str, pipeline_type: str = "sd") -> dict:
    scheduler_config = hf_pipe.scheduler.config
    vae_config = hf_pipe.vae.config

    is_sd3 = pipeline_type == "sd3"
    denoiser_config = hf_pipe.transformer.config if is_sd3 else hf_pipe.unet.config

    prediction_type = getattr(scheduler_config, "prediction_type", None) or "epsilon"
    scaling_factor = getattr(vae_config, "scaling_factor", None) or 0.18215
    shift_factor = getattr(vae_config, "shift_factor", None) or 0.0

    vae_scale_power = len(vae_config.block_out_channels) - 1
    vae_spatial_scale = 2**vae_scale_power
    image_size = denoiser_config.sample_size * vae_spatial_scale

    config: dict[str, Any] = {
        "type": "stable-diffusion-3" if is_sd3 else "stable-diffusion",
        "prediction_type": "flow" if is_sd3 else prediction_type,
        "encoder_scale_factor": scaling_factor,
        "decoder_scale_factor": scaling_factor,
        "decoder_shift_factor": shift_factor,
        "image_size": image_size,
        "default_guidance_scale": 5.0 if is_sd3 else 7.5,
        "default_steps": 28 if is_sd3 else 50,
    }

    # Include scheduler defaults for reproducibility
    config["scheduler"] = {
        "training_steps": getattr(scheduler_config, "num_train_timesteps", 1000),
        "beta_start": getattr(scheduler_config, "beta_start", 0.00085),
        "beta_end": getattr(scheduler_config, "beta_end", 0.012),
        "beta_schedule": getattr(scheduler_config, "beta_schedule", "scaled_linear"),
    }

    return config


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------


def _resolve_compression(compression: str) -> dict | None:
    """Resolve a compression string to a config dict or None."""
    if compression in PRESETS:
        config = PRESETS[compression].get("config")
        return cast(dict | None, config)
    try:
        parsed: dict = json.loads(compression)
        return parsed
    except (json.JSONDecodeError, TypeError) as e:
        available = ", ".join(list_presets())
        raise ValueError(
            f"Unknown compression value '{compression}'. "
            f"Expected a preset name ({available}) or a JSON config dict."
        ) from e
