# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""CLI entry point for coreai.diffusion.export."""

import argparse
import logging
import sys
from pathlib import Path

from coreai_models.diffusion.components import get_valid_components
from coreai_models.diffusion.models import get_pipeline_type
from coreai_models.diffusion.pipeline import DiffusionExportConfig, export_diffusion
from coreai_models.diffusion.presets import DEFAULT_COMPRESSION_PRESET
from coreai_models.model_registry import try_lookup_preset, try_lookup_preset_by_hf_id


def _default_output_dir() -> str:
    """Resolve exports/ relative to the workspace root (where pyproject.toml lives)."""
    d = Path(__file__).resolve().parent
    while d != d.parent:
        if (d / "pyproject.toml").exists() and (d / "python").exists():
            return str(d / "exports")
        d = d.parent
    return "exports"


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the diffusion export CLI."""
    parser = argparse.ArgumentParser(
        prog="coreai.diffusion.export",
        description="Export diffusion models to Core AI format. "
        "Accepts a registry short-name (e.g. flux2-klein-4b) or a HuggingFace model ID.",
    )
    parser.add_argument(
        "model",
        help="Registry short-name (e.g. flux2-klein-4b) or HuggingFace model ID",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for exported assets (default: <repo-root>/exports/)",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=None,
        help="Components to export (default: all). "
        "SD 1.x/2.x: text_encoder unet vae_decoder vae_encoder. "
        "SD 3.x: text_encoder text_encoder_2 transformer vae_decoder. "
        "FLUX.2: transformer text_encoder vae_decoder vae_encoder.",
    )
    parser.add_argument(
        "--compute-precision",
        default=None,
        choices=["float16", "bfloat16", "float32"],
        help="Model precision for export. "
        "Required for raw HF IDs; resolved automatically for registry short-names.",
    )
    parser.add_argument(
        "--compression",
        default=None,
        help="Compression preset name, JSON config, or 'none' (see --list-presets)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--platform",
        default=None,
        choices=["iOS", "macOS"],
        help="Target platform. iOS defaults to 512 resolution; macOS defaults to 1024.",
    )
    parser.add_argument(
        "--resolution",
        default=None,
        type=int,
        choices=[512, 1024],
        help="Output image resolution. Overrides the platform default.",
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Include half-resolution VAEs for tiled decode.",
    )
    parser.add_argument(
        "--multifunction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export FLUX.2 transformer as multi-function .aimodel with 5 functions "
        "(main, half, img2img_quarter/half/full) sharing weights. "
        "Default: enabled. Use --no-multifunction for separate single-function models.",
    )
    parser.add_argument(
        "--experimental",
        action="store_true",
        help="Allow exporting models without a registry preset. Requires --compute-precision.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved export config and exit without exporting",
    )
    parser.add_argument(
        "--include-debug-info",
        action="store_true",
        help=(
            "Embed debug information in the exported .aimodel for debugging a conversion. "
            "Default: off, which embeds minimum debug information and makes the "
            "exported asset smaller."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    return parser


def _is_hf_id(model: str) -> bool:
    return "/" in model


def main() -> None:
    """Main entry point for the diffusion export CLI."""
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- Registry resolution ---
    hf_model_id = args.model
    compression = args.compression
    compute_precision = args.compute_precision
    output_dir = args.output_dir or _default_output_dir()

    preset = None
    if not _is_hf_id(args.model):
        preset = try_lookup_preset(args.model, model_type="diffusion")
        if preset is None:
            parser.error(
                f"'{args.model}' is not a registered short-name and doesn't look like a "
                "HuggingFace ID (expected 'org/model'). "
                "Run `uv run coreai.model.registry --list-models --type diffusion` to see options."
            )
    else:
        preset = try_lookup_preset_by_hf_id(args.model, model_type="diffusion")

    if preset is not None:
        hf_model_id = preset.hf_id
        if compression is None and preset.compression:
            compression = preset.compression
        if compute_precision is None and preset.compute_precision:
            compute_precision = preset.compute_precision
    elif _is_hf_id(args.model) and not args.experimental:
        parser.error(
            f"'{args.model}' has no registry preset. "
            "Pass --experimental to try exporting it anyway "
            "(requires --compute-precision).\n"
            "See models/README.md for supported models."
        )

    if compute_precision is None:
        parser.error(
            f"--compute-precision is required for '{args.model}' "
            "(no registry preset found). "
            "Pass --compute-precision float16|bfloat16|float32 explicitly.\n"
            "See models/README.md for more information."
        )

    compression = compression if compression is not None else DEFAULT_COMPRESSION_PRESET

    pipeline_type = get_pipeline_type(hf_model_id)

    if args.components and args.platform:
        parser.error("Cannot specify both --components and --platform. Use only one.")

    if args.components:
        valid = get_valid_components(pipeline_type, multifunction=args.multifunction)
        invalid = [c for c in args.components if c not in valid]
        if invalid:
            parser.error(
                f"Invalid components for {pipeline_type}: {invalid}. Valid choices: {valid}"
            )

    # Platform-based component selection (FLUX.2 only)
    target_components: list[str] | None = None
    if args.platform and pipeline_type == "flux2":
        # Resolve effective resolution: --resolution overrides platform default
        resolution = args.resolution
        if resolution is None:
            resolution = 512 if args.platform == "iOS" else 1024

        if args.multifunction:
            # Multi-function mode: single transformer has all variants
            target_components = [
                "transformer",
                "text_encoder",
                "vae_decoder",
                "vae_encoder",
            ]
            # Always include half VAEs for img2img reference encoding
            for half in ["vae_decoder_half", "vae_encoder_half"]:
                if half not in target_components:
                    target_components.append(half)
        elif resolution == 512:
            target_components = [
                "transformer_512",
                "text_encoder",
                "vae_decoder_half",
                "vae_encoder_half",
            ]
        else:
            target_components = [
                "transformer",
                "text_encoder",
                "vae_decoder",
                "vae_encoder",
            ]

            # --low-memory adds half VAEs for tiled decode
            if args.low_memory:
                for half in ["vae_decoder_half", "vae_encoder_half"]:
                    if half not in target_components:
                        target_components.append(half)

    config = DiffusionExportConfig(
        hf_model_id=hf_model_id,
        output_dir=output_dir,
        components=args.components or target_components,
        compute_precision=compute_precision,
        compression=compression,
        overwrite=args.overwrite,
        include_debug_info=args.include_debug_info,
        multifunction=args.multifunction,
    )

    if args.dry_run:
        print("Dry run — resolved export config:")
        print(f"  model:              {config.hf_model_id}")
        print(f"  compression:        {config.compression}")
        print(f"  compute_precision:  {config.compute_precision}")
        print(f"  output_dir:         {config.output_dir}")
        if config.components:
            print(f"  components:         {', '.join(config.components)}")
        else:
            print("  components:        all")
        print(f"  multifunction:     {config.multifunction}")
        print(f"  overwrite:         {config.overwrite}")
        print(f"  include_debug_info: {config.include_debug_info}")
        return

    try:
        results = export_diffusion(config)
        failed = [k for k, v in results.items() if "FAILED" in str(v)]
        if failed:
            logging.getLogger(__name__).error(f"Failed components: {failed}")
            sys.exit(1)
        print(f"Export complete: {config.output_dir}")
    except Exception as e:
        logging.getLogger(__name__).error(f"Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
