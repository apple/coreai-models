# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""CLI entry point for ``coreai.segmentation.export``.

Currently supports SAM3 (the registry short-name ``sam3`` or the HF id
``facebook/sam3``). Two export paths share this CLI:

  * **Optimized (default)** — re-authored ANE-targeted model split into
    three independently optimizable functions (``image_encode``,
    ``text_encode``, ``detect``) with palettized encoders + fp16.
  * **Baseline (`--baseline`)** — plain ``transformers.Sam3Model``,
    single ``main`` entrypoint, mirroring the original
    ``models/sam3/export.py`` before the ANE refactor. Useful for
    parity checks against the optimized export.

Both paths produce a segmenter bundle directory containing an
``.aimodel``, a ``tokenizer/`` folder, and a ``metadata.json``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from coreai_models.model_registry import lookup_utility_model
from coreai_models.segmentation.pipeline import (
    BaselineExportConfig,
    SegmentationExportConfig,
    export_baseline,
    export_segmentation,
)

_SUPPORTED = {
    "sam3": "facebook/sam3",
    "facebook/sam3": "facebook/sam3",
}

# Defaults that differ between the two export paths. Used only when
# ``--image-size`` isn't passed explicitly so each mode picks the
# resolution it was designed for.
_OPTIMIZED_DEFAULT_IMAGE_SIZE = 336
_BASELINE_DEFAULT_IMAGE_SIZE = 1008


def _find_repo_root() -> Path | None:
    d = Path(__file__).resolve().parent
    while d != d.parent:
        if (d / "pyproject.toml").exists() and (d / "python").exists():
            return d
        d = d.parent
    return None


def _default_output_dir() -> str:
    root = _find_repo_root()
    return str(root / "exports") if root is not None else "exports"


def _resolve_hf_model_id(model: str) -> str:
    """Accept registry short-name or HF id; reject anything else."""
    if model in _SUPPORTED:
        return _SUPPORTED[model]
    preset = lookup_utility_model(model)
    if preset is not None and preset.task == "segmentation" and preset.hf_id in _SUPPORTED:
        return _SUPPORTED[preset.hf_id]
    raise SystemExit(
        f"Error: '{model}' is not a supported segmentation model. "
        f"Supported: {sorted(set(_SUPPORTED.values()))}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coreai.segmentation.export",
        description=(
            "Export segmentation models to Core AI format. By default targets "
            "Apple Neural Engine via a re-authored 3-function bundle "
            "(image_encode / text_encode / detect) with palettized encoders "
            "+ fp16. Pass --baseline to instead export the unmodified HF model "
            "as a single-entrypoint asset."
        ),
    )
    parser.add_argument(
        "model",
        help=(
            "Segmentation model. Either the registry short-name (e.g. 'sam3') "
            "or its HuggingFace id (e.g. 'facebook/sam3')."
        ),
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help=(
            "Export the plain HF model with no re-authoring or palettization. "
            "Single 'main' entrypoint; useful as a parity reference."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for the bundle (default: <repo-root>/exports/)",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help=(
            "Custom bundle directory name (default: derived from model + "
            "image-size + n-bits, or from model + dtype when --baseline)."
        ),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help=(
            "Input resolution. Defaults to 336 (optimized) or 1008 (--baseline). "
            "336 keeps the optimized path's global-attention sequence inside "
            "Neural Engine SRAM."
        ),
    )
    # ---- Optimized-only flags -------------------------------------------
    parser.add_argument(
        "--max-text-seq-len",
        type=int,
        default=32,
        help="(optimized) Static text sequence length used at export time.",
    )
    parser.add_argument(
        "--n-bits",
        type=int,
        default=None,
        choices=[2, 3, 4, 6, 8],
        help=(
            "(optimized) Uniform K-means palettization bit-width override applied "
            "to BOTH image and text encoders. Default is asymmetric: image w4, text w6."
        ),
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=None,
        help=(
            "(optimized) Uniform palettization group-size override applied to BOTH "
            "image and text encoders. Default is asymmetric: image gs32, text gs8."
        ),
    )
    # ---- Baseline-only flags --------------------------------------------
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32"],
        default="float32",
        help="(--baseline) Torch dtype to use for the model.",
    )
    # ---- Shared flags ---------------------------------------------------
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing bundle directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved export config and exit without exporting.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )
    return parser


def _resolve_image_size(args: argparse.Namespace) -> int:
    if args.image_size is not None:
        return args.image_size
    return _BASELINE_DEFAULT_IMAGE_SIZE if args.baseline else _OPTIMIZED_DEFAULT_IMAGE_SIZE


def _warn_unused_flags(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Surface flags that don't apply to the chosen mode so users notice typos.

    argparse can't natively express "this flag only applies when --baseline
    is set" without subparsers, so we just check after parsing.
    """
    parser_defaults = parser.parse_args([args.model])
    if args.baseline:
        # Optimized-only flags shouldn't be set in baseline mode.
        ignored = []
        for name in ("max_text_seq_len", "n_bits", "group_size"):
            if getattr(args, name) != getattr(parser_defaults, name):
                ignored.append(name.replace("_", "-"))
        if ignored:
            logging.warning(
                "Ignoring optimized-only flag(s) in baseline mode: %s",
                ", ".join(f"--{n}" for n in ignored),
            )
    else:
        if args.dtype != parser_defaults.dtype:
            logging.warning("Ignoring --dtype outside baseline mode (optimized path is fp16).")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    hf_model_id = _resolve_hf_model_id(args.model)
    image_size = _resolve_image_size(args)
    _warn_unused_flags(parser, args)

    if args.baseline:
        config = BaselineExportConfig(
            hf_model_id=hf_model_id,
            image_size=image_size,
            dtype=args.dtype,
            output_dir=args.output_dir or _default_output_dir(),
            output_name=args.output_name,
            overwrite=args.overwrite,
        )
        if args.dry_run:
            print("Dry run — resolved baseline export config:")
            print(f"  model:       {config.hf_model_id}")
            print(f"  image_size:  {config.image_size}")
            print(f"  dtype:       {config.dtype}")
            print(f"  output_dir:  {config.output_dir}")
            if config.output_name:
                print(f"  output_name: {config.output_name}")
            print(f"  overwrite:   {config.overwrite}")
            return
        bundle_path = export_baseline(config)
    else:
        # Resolve asymmetric defaults from SegmentationExportConfig; --n-bits / --group-size
        # are uniform overrides that apply to BOTH encoders when set.
        defaults = SegmentationExportConfig()
        image_n_bits = args.n_bits if args.n_bits is not None else defaults.image_n_bits
        text_n_bits = args.n_bits if args.n_bits is not None else defaults.text_n_bits
        image_group_size = (
            args.group_size if args.group_size is not None else defaults.image_group_size
        )
        text_group_size = (
            args.group_size if args.group_size is not None else defaults.text_group_size
        )

        config = SegmentationExportConfig(
            hf_model_id=hf_model_id,
            image_size=image_size,
            max_text_seq_len=args.max_text_seq_len,
            image_n_bits=image_n_bits,
            image_group_size=image_group_size,
            text_n_bits=text_n_bits,
            text_group_size=text_group_size,
            output_dir=args.output_dir or _default_output_dir(),
            output_name=args.output_name,
            overwrite=args.overwrite,
        )
        if args.dry_run:
            print("Dry run — resolved export config:")
            print(f"  model:             {config.hf_model_id}")
            print(f"  image_size:        {config.image_size}")
            print(f"  max_text_seq_len:  {config.max_text_seq_len}")
            print(f"  image_n_bits:      {config.image_n_bits}")
            print(f"  image_group_size:  {config.image_group_size}")
            print(f"  text_n_bits:       {config.text_n_bits}")
            print(f"  text_group_size:   {config.text_group_size}")
            print(f"  output_dir:        {config.output_dir}")
            if config.output_name:
                print(f"  output_name:       {config.output_name}")
            print(f"  overwrite:         {config.overwrite}")
            return
        bundle_path = export_segmentation(config)

    print(f"Export complete: {bundle_path}")


if __name__ == "__main__":
    main()
