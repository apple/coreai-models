# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""CLI entry point for coreai.llm.export."""

import argparse
import logging
from pathlib import Path

import yaml
from coreai_opt.palettization.config.palettization_config import KMeansPalettizerConfig
from coreai_opt.quantization import QuantizerConfig

from coreai_models.diffusion.models import SUPPORTED_MODELS as DIFFUSION_MODELS
from coreai_models.diffusion.presets import (
    DEFAULT_COMPRESSION_PRESET as DIFFUSION_DEFAULT,
)
from coreai_models.diffusion.presets import (
    PRESETS as DIFFUSION_PRESETS,
)
from coreai_models.export.pipeline import ExportConfig, export_model
from coreai_models.export.presets import ALL_PRESET_NAMES, IOS_PRESETS, MACOS_PRESETS, list_presets
from coreai_models.export.presets import (
    DEFAULT_IOS_COMPRESSION_PRESET as IOS_DEFAULT,
)
from coreai_models.export.presets import DEFAULT_MACOS_COMPRESSION_PRESET as MACOS_DEFAULT
from coreai_models.model_registry import try_lookup_preset, try_lookup_preset_by_hf_id
from coreai_models.models.registry import list_models as list_llm_models


def _find_repo_root() -> Path | None:
    """Walk up from this file to find the workspace root (where pyproject.toml + python/ live).

    Returns None when running from a pip-installed wheel (no enclosing repo checkout).
    """
    d = Path(__file__).resolve().parent
    while d != d.parent:
        if (d / "pyproject.toml").exists() and (d / "python").exists():
            return d
        d = d.parent
    return None


def _default_output_dir() -> str:
    """Resolve exports/ relative to the workspace root (where pyproject.toml lives)."""
    root = _find_repo_root()
    return str(root / "exports") if root is not None else "exports"


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the export CLI."""
    parser = argparse.ArgumentParser(
        prog="coreai.llm.export",
        description="Export HuggingFace models to Core AI format. "
        "Accepts a registry short-name (e.g. qwen3-0.6b) or a HuggingFace model ID.",
    )
    parser.add_argument(
        "model",
        nargs="?",
        help="Registry short-name (e.g. qwen3-0.6b) or HuggingFace model ID (e.g. Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--platform",
        choices=["macOS", "iOS"],
        default=None,
        help="Target device platform (default: macOS, or from registry preset)",
    )
    compression_group = parser.add_mutually_exclusive_group()
    compression_group.add_argument(
        "--compression",
        default=None,
        help=f"Compression preset name or 'none' (macOS default: '{MACOS_DEFAULT}', iOS "
        f"default: '{IOS_DEFAULT}'). Registry presets override these defaults.",
    )
    compression_group.add_argument(
        "--compression-config",
        default=None,
        type=Path,
        help=(
            "Path to a coreai-opt YAML config."
            "Top-level key must be 'kmeans_palettization_config' (iOS) "
            "or 'quantization_config' (macOS). See models/<family>/ in the source tree for "
            "shipped recipes. Mutually exclusive with --compression."
        ),
    )
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=None,
        help="Maximum context length (default: from registry preset or model config)",
    )
    parser.add_argument(
        "--compute-precision",
        default=None,
        help="Compute precision: float16, bfloat16, float32. "
        "Required for raw HF IDs; resolved automatically for registry short-names.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for exported model (default: <repo-root>/exports/)",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Custom output filename (without extension)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Truncate to N layers (useful for debugging)",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available compression presets and exit",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List supported model types and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved export config and exit without exporting",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--experimental",
        action="store_true",
        help="Allow exporting models without a registry preset. Requires --compute-precision.",
    )
    parser.add_argument(
        "--disable-embedding-quantization-ios",
        action="store_true",
        help=(
            "iOS only. Skip int8 quantization of the embedding table and keep it in "
            "float32. Default: False (embedding is quantized). Rejected when "
            "--platform is macOS."
        ),
    )
    return parser


def _is_hf_id(model: str) -> bool:
    return "/" in model


def _load_compression_config_object(yaml_path: Path, variant: str):  # type: ignore[no-untyped-def]
    """Load a coreai-opt YAML config and return either a prebuilt coreai-opt config object
    (palettization) or a config dict with extra keys
    (currently `coreai_models.calibrate_activations`) considered for quantization.

    The YAML's top-level must contain exactly one coreai-opt schema key
    (`kmeans_palettization_config` or `quantization_config`).

    Raises SystemExit on any validation mismatch.
    """
    try:
        with yaml_path.open() as fh:
            yaml_data = yaml.safe_load(fh)
    except FileNotFoundError as exc:
        raise SystemExit(f"compression config: file not found: {yaml_path}") from exc

    if not isinstance(yaml_data, dict):
        raise SystemExit(f"{yaml_path}: expected a YAML mapping at top level.")

    pipeline_level_options = yaml_data.pop("coreai_models", {})
    if not isinstance(pipeline_level_options, dict):
        raise SystemExit(
            f"{yaml_path}: 'coreai_models' must be a mapping, got "
            f"{type(pipeline_level_options).__name__}."
        )
    allowed_pipeline_level_keys = {"calibrate_activations"}
    unknown = set(pipeline_level_options) - allowed_pipeline_level_keys
    if unknown:
        raise SystemExit(
            f"{yaml_path}: unknown key(s) in 'coreai_models' block: "
            f"{sorted(unknown)}. Allowed: {sorted(allowed_pipeline_level_keys)}."
        )

    if len(yaml_data) != 1:
        raise SystemExit(
            f"{yaml_path}: expected exactly one coreai-opt top-level key "
            "('kmeans_palettization_config' or 'quantization_config'), "
            f"got {sorted(yaml_data)}."
        )
    top_key = next(iter(yaml_data))
    inner = yaml_data[top_key]

    if top_key == "kmeans_palettization_config":
        if pipeline_level_options:
            raise SystemExit(
                f"{yaml_path}: palettization configs do not support the 'coreai_models' block."
            )
        if variant != "iOS":
            raise SystemExit(
                f"{yaml_path}: palettization YAML requires --platform iOS (got '{variant}')."
            )
        return KMeansPalettizerConfig.from_dict({top_key: inner})

    if top_key == "quantization_config":
        if variant != "macOS":
            raise SystemExit(
                f"{yaml_path}: quantization YAML requires --platform macOS (got '{variant}')."
            )
        # Validate the coreai-opt block early so schema errors surface before
        # we merge pipeline-level options back in.
        QuantizerConfig.from_dict({top_key: inner})
        # Re-inline our extension keys into the inner dict. The downstream
        # pipeline path (pipeline.py + compression.quantize_pytorch_model)
        # pops `calibrate_activations` off the dict before rebuilding the
        # coreai-opt config, mirroring how the built-in presets path
        # carries the flag.
        merged = dict(inner)
        if "calibrate_activations" in pipeline_level_options:
            merged["calibrate_activations"] = pipeline_level_options["calibrate_activations"]
        return merged

    raise SystemExit(
        f"{yaml_path}: unknown top-level key '{top_key}'. Expected "
        "'kmeans_palettization_config' or 'quantization_config'."
    )


def _resolve_registry_compression_config(relpath: str, variant: str) -> tuple[object, str]:
    """Resolve a registry preset's `compression_config` to a loaded coreai-opt object.

    `relpath` is interpreted relative to the repo root (e.g.
    `models/qwen3/qwen3_0_6b_mixed_4bit_8bit.yaml`). The YAML lives in the
    source tree and is not shipped with the wheel, so this only works when
    running from a coreai-models checkout.
    """
    root = _find_repo_root()
    if root is None:
        raise SystemExit(
            f"Registry preset references {relpath}, but the YAML lives in "
            "the source tree which is unavailable in this install. "
            "Run from a coreai-models checkout or pass --compression-config explicitly."
        )
    yaml_path = root / relpath
    if not yaml_path.is_file():
        raise SystemExit(
            f"Registry preset references missing YAML: {yaml_path}. Expected file at {relpath}."
        )
    return _load_compression_config_object(yaml_path, variant), yaml_path.stem


def _resolve_export_config(args: argparse.Namespace) -> ExportConfig:
    """Resolve CLI args + registry preset into a final ExportConfig."""
    hf_model_id = args.model
    variant = args.platform or "macOS"
    compression = args.compression
    compute_precision = args.compute_precision
    max_context_length = args.max_context_length
    output_dir = args.output_dir or _default_output_dir()
    compression_config_object = None
    registry_compression_config: str | None = None  # repo-root-relative path, e.g. models/qwen3/...

    preset = None
    if not _is_hf_id(args.model):
        preset = try_lookup_preset(args.model, model_type="llm", variant=args.platform)
        if preset is None:
            other = try_lookup_preset(args.model, model_type="llm")
            if other is not None and args.platform:
                raise SystemExit(
                    f"Error: '{args.model}' is not available for {args.platform}. "
                    "Run --list-models to see options."
                )
            raise SystemExit(
                f"Error: '{args.model}' is not a registered short-name and doesn't look like a "
                "HuggingFace ID (expected 'org/model'). Run --list-models to see options."
            )
    else:
        # HuggingFace ID — check if we have a matching preset for defaults
        preset = try_lookup_preset_by_hf_id(args.model, model_type="llm", variant=args.platform)

    if preset is not None:
        hf_model_id = preset.hf_id
        variant = args.platform or preset.variant or "macOS"
        if compression is None and preset.compression:
            compression = preset.compression
        if compute_precision is None and preset.compute_precision:
            compute_precision = preset.compute_precision
        if max_context_length is None and preset.max_context_length:
            max_context_length = preset.max_context_length
        if args.compression is None and getattr(preset, "compression_config", None) is not None:
            registry_compression_config = preset.compression_config
    elif _is_hf_id(args.model) and not args.experimental:
        hint = ""
        if variant == "iOS":
            hint = (
                "\nThis model may not be suitable for iOS application "
                "due to its memory requirements."
            )
        raise SystemExit(
            f"Error: '{args.model}' has no registry preset. "
            "Pass --experimental to try exporting it anyway "
            "(requires --compute-precision).\n"
            "See models/README.md for supported models." + hint
        )

    if compute_precision is None:
        raise SystemExit(
            f"Error: --compute-precision is required for '{args.model}' "
            "(no registry preset found). "
            "Pass --compute-precision float16|bfloat16|float32 explicitly.\n"
            "See models/README.md for more information, including testing "
            "different quantization options and kv-cache limits."
        )

    if args.disable_embedding_quantization_ios and variant != "iOS":
        raise SystemExit(
            f"--disable-embedding-quantization-ios requires --platform iOS (got '{variant}')."
        )

    if args.compression_config is not None:
        if not args.compression_config.is_file():
            raise SystemExit(f"--compression-config: file not found: {args.compression_config}")
        compression_config_object = _load_compression_config_object(
            args.compression_config, variant
        )
        compression = args.compression_config.stem
    elif registry_compression_config is not None:
        compression_config_object, compression = _resolve_registry_compression_config(
            registry_compression_config, variant
        )
    elif compression == "none":
        pass
    elif not compression:
        compression = MACOS_DEFAULT if variant == "macOS" else IOS_DEFAULT
    elif compression in MACOS_PRESETS and variant == "iOS":
        raise RuntimeError("macOS quantization preset provided, but platform is iOS.")
    elif compression in IOS_PRESETS and variant == "macOS":
        raise RuntimeError("iOS palettization preset provided, but platform is macOS.")
    elif compression not in ALL_PRESET_NAMES and compression != "none":
        raise RuntimeError(
            f"Compression preset {compression} is not a valid compression "
            f"preset. Available: {list_presets()}"
        )

    return ExportConfig(
        hf_model_id=hf_model_id,
        variant=variant,
        compression=compression,
        max_context_length=max_context_length,
        compute_precision=compute_precision,
        output_dir=output_dir,
        output_name=args.output_name,
        num_layers=args.num_layers,
        overwrite=args.overwrite,
        compression_config_object=compression_config_object,
        disable_embedding_quantization=args.disable_embedding_quantization_ios,
    )


def main() -> None:
    """Main entry point for the export CLI."""
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.list_presets:
        print("LLM compression presets:")
        print(f"  macOS (default: {MACOS_DEFAULT})")
        for name in sorted(MACOS_PRESETS):
            desc = MACOS_PRESETS[name].get("description", "")
            print(f"    {name:40s} {desc}")
        print()
        print(f"  iOS (default: {IOS_DEFAULT})")
        for name in sorted(IOS_PRESETS):
            desc = IOS_PRESETS[name].get("description", "")
            print(f"    {name:40s} {desc}")
        print()
        print(f"Diffusion compression presets (default: {DIFFUSION_DEFAULT}):")
        print()
        for name in sorted(DIFFUSION_PRESETS):
            desc = DIFFUSION_PRESETS[name].get("description", "")
            print(f"  {name:40s} {desc}")
        return

    if args.list_models:
        print("LLM model types:")
        print()
        for name in list_llm_models():
            print(f"  {name}")
        print()
        print("Diffusion model families:")
        print()
        for name, example, _ in DIFFUSION_MODELS:
            print(f"  {name:40s} (e.g. {example})")
        return

    if not args.model:
        parser.error("model is required (unless using --list-presets or --list-models)")

    config = _resolve_export_config(args)

    if args.dry_run:
        print("Dry run — resolved export config:")
        print(f"  model:              {config.hf_model_id}")
        print(f"  platform:           {config.variant}")
        print(f"  compression:        {config.compression}")
        print(f"  compute_precision:  {config.compute_precision}")
        if config.max_context_length:
            print(f"  max_context_length: {config.max_context_length}")
        print(f"  output_dir:         {config.output_dir}")
        if config.output_name:
            print(f"  output_name:        {config.output_name}")
        if config.num_layers:
            print(f"  num_layers:         {config.num_layers}")
        print(f"  overwrite:          {config.overwrite}")
        if config.variant == "iOS":
            print(f"  disable_embedding_quantization: {config.disable_embedding_quantization}")
        return

    result = export_model(config)
    print(f"Export complete: {result}")


if __name__ == "__main__":
    main()
