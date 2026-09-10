# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""CLI entry point for ``coreai.segmentation.export``.

Currently supports SAM3 (the registry short-name ``sam3`` or the HF id
``facebook/sam3``). Three export paths share this CLI:

  * **Lite (default)** — ANE-targeted model split into three independently
    optimizable functions (``image_encode``, ``text_encode``, ``detect``)
    with palettized encoders + fp16.
  * **Full (``--full``)** — plain ``transformers.Sam3Model``,
    single ``main`` entrypoint, higher quality at the cost of size and
    on-device speed.
  * **Video (``--video``)** — ``transformers.Sam3VideoModel`` split into seven
    functions for detect-and-track. The inference session, memory ring buffer
    and tracking heuristics stay on the host.

All three paths produce a bundle directory containing an ``.aimodel``, a
``tokenizer/`` folder, and a ``metadata.json``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from coreai_models.segmentation.pipeline import (
    FullExportConfig,
    SegmentationExportConfig,
    export_full,
    export_segmentation,
)
from coreai_models.segmentation.video_pipeline import VideoExportConfig, export_video

# Accepted ``--model`` spellings → canonical HF id. Doubles as the source of
# truth for the flag's ``choices``, so adding an alias here is all it takes.
_SUPPORTED = {
    "sam3": "facebook/sam3",
    "facebook/sam3": "facebook/sam3",
}


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
    """Map an accepted ``--model`` value to its canonical HF id.

    ``--model`` derives its ``choices`` from ``_SUPPORTED``, so argparse has
    already rejected anything unknown by the time this runs.
    """
    return _SUPPORTED[model]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coreai.segmentation.export",
        description=(
            "Export segmentation models to Core AI format. By default exports "
            "the lite variant targeting iOS via a 3-function bundle "
            "(image_encode / text_encode / detect) with palettized encoders "
            "+ fp16. Pass --full to instead export the unmodified HF image "
            "model as a single-entrypoint asset, or --video for the 7-function "
            "detect-and-track bundle."
        ),
    )
    parser.add_argument(
        "--model",
        choices=sorted(_SUPPORTED),
        default="facebook/sam3",
        help=(
            "Segmentation model. Either the registry short-name (e.g. 'sam3') "
            "or its HuggingFace id (e.g. 'facebook/sam3')."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--full",
        action="store_true",
        help=(
            "Export the plain HF image model with no ANE targeting or palettization. "
            "Single 'main' entrypoint; higher quality at the cost of size and speed."
        ),
    )
    mode.add_argument(
        "--video",
        action="store_true",
        help=(
            "Export SAM3 video segmentation (Sam3VideoModel) as a 7-function bundle: "
            "image_encode, text_encode, detect, tracker_encode, tracker_step, "
            "memory_encode, tracker_mask_init."
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
            "image-size + n-bits, or from model + dtype when --full)."
        ),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help=("Input resolution. Defaults to 336 (lite) or 1008 (--full / --video). "),
    )
    # ---- Lite-only flags -------------------------------------------
    # Mode-specific flags default to None rather than their real default so
    # _warn_unused_flags can tell "user passed this" from "user left it alone",
    # even when the value passed matches the default. Resolved in main().
    parser.add_argument(
        "--max-text-seq-len",
        type=int,
        default=None,
        help="(lite, video) Static text sequence length used at export time. Default: 32.",
    )
    parser.add_argument(
        "--n-bits",
        type=int,
        default=None,
        choices=[2, 3, 4, 6, 8],
        help=(
            "(lite) Uniform K-means palettization bit-width override applied "
            "to BOTH image and text encoders. Default is asymmetric: image w4, text w6."
        ),
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=None,
        help=(
            "(lite) Uniform palettization group-size override applied to BOTH "
            "image and text encoders. Default is asymmetric: image gs32, text gs8."
        ),
    )
    # ---- Full-only flags --------------------------------------------
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32"],
        default=None,
        help=(
            "(--full, --video) Torch dtype for the model. Default: float32 (full), float16 (video)."
        ),
    )
    # ---- Video-only flags -----------------------------------------------
    parser.add_argument(
        "--spatial-slots",
        type=int,
        default=None,
        help=(
            "(--video) Fixed spatial-memory slot count for tracker_step. Must be at "
            "least max_cond_frame_num + num_maskmem - 1 (10 for facebook/sam3)."
        ),
    )
    parser.add_argument(
        "--ptr-slots",
        type=int,
        default=None,
        help=(
            "(--video) Fixed object-pointer slot count for tracker_step. Must be at "
            "least max_object_pointers_in_encoder (16 for facebook/sam3); the extra "
            "headroom absorbs pointers from accumulated conditioning frames."
        ),
    )
    # ---- Shared flags ---------------------------------------------------
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing bundle directory.",
    )
    parser.add_argument(
        "--include-debug-info",
        action="store_true",
        help=(
            "Embed debug information in the exported .aimodel for debugging a conversion. "
            "Default: off, which embeds minimum debug information and makes the "
            "exported asset smaller. Applies to both modes."
        ),
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


def _resolve_mode(args: argparse.Namespace) -> str:
    """``--full`` / ``--video`` are mutually exclusive; absent both, it's lite."""
    if args.full:
        return "full"
    if args.video:
        return "video"
    return "lite"


def _resolve_image_size(args: argparse.Namespace) -> int:
    """Pick the resolution each mode was designed for when --image-size is omitted.

    Mode-specific flags default to None so ``_warn_unused_flags`` can detect
    "passed" regardless of value; the real defaults live on the config
    dataclasses. ``@dataclass`` leaves plain defaults as class attributes, so
    they're readable without constructing a config.
    """
    if args.image_size is not None:
        return args.image_size
    return {
        "lite": SegmentationExportConfig.image_size,
        "full": FullExportConfig.image_size,
        "video": VideoExportConfig.image_size,
    }[_resolve_mode(args)]


#: Which modes each mode-specific flag actually reaches. Anything passed
#: outside its listed modes is silently ignored by the export, so warn.
_FLAG_MODES = {
    "max_text_seq_len": {"lite", "video"},
    "n_bits": {"lite"},
    "group_size": {"lite"},
    "dtype": {"full", "video"},
    "spatial_slots": {"video"},
    "ptr_slots": {"video"},
}


def _warn_unused_flags(args: argparse.Namespace) -> None:
    """Surface flags that don't apply to the chosen mode so users notice typos.

    argparse can't natively express "this flag only applies when --full is set"
    without subparsers. Every mode-specific flag defaults to ``None``, so "not
    None" means the user passed it explicitly — including when the value they
    passed happens to equal the mode's resolved default.
    """
    mode = _resolve_mode(args)
    ignored = [
        name.replace("_", "-")
        for name, modes in _FLAG_MODES.items()
        if mode not in modes and getattr(args, name, None) is not None
    ]
    if ignored:
        logging.warning(
            "Ignoring flag(s) that do not apply in %s mode: %s",
            mode,
            ", ".join(f"--{n}" for n in ignored),
        )


def _print_dry_run(label: str, config: object, fields: list[str]) -> None:
    print(f"Dry run — resolved {label} export config:")
    width = max(len(f) for f in fields) + 1
    for field in fields:
        value = getattr(config, field)
        if field == "output_name" and value is None:
            continue
        print(f"  {field + ':':<{width}} {value}")


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
    mode = _resolve_mode(args)
    _warn_unused_flags(args)
    output_dir = args.output_dir or _default_output_dir()

    if mode == "video":
        video_defaults = VideoExportConfig()
        video_config = VideoExportConfig(
            hf_model_id=hf_model_id,
            image_size=image_size,
            dtype=args.dtype or video_defaults.dtype,
            spatial_slots=args.spatial_slots or video_defaults.spatial_slots,
            ptr_slots=args.ptr_slots or video_defaults.ptr_slots,
            max_text_seq_len=args.max_text_seq_len or video_defaults.max_text_seq_len,
            output_dir=output_dir,
            output_name=args.output_name,
            overwrite=args.overwrite,
            include_debug_info=args.include_debug_info,
        )
        if args.dry_run:
            _print_dry_run(
                "video",
                video_config,
                [
                    "hf_model_id",
                    "image_size",
                    "dtype",
                    "spatial_slots",
                    "ptr_slots",
                    "max_text_seq_len",
                    "output_dir",
                    "output_name",
                    "overwrite",
                    "include_debug_info",
                ],
            )
            return
        bundle_path = export_video(video_config)
    elif mode == "full":
        full_config = FullExportConfig(
            hf_model_id=hf_model_id,
            image_size=image_size,
            dtype=args.dtype or FullExportConfig.dtype,
            output_dir=output_dir,
            output_name=args.output_name,
            overwrite=args.overwrite,
            include_debug_info=args.include_debug_info,
        )
        if args.dry_run:
            _print_dry_run(
                "full",
                full_config,
                [
                    "hf_model_id",
                    "image_size",
                    "dtype",
                    "output_dir",
                    "output_name",
                    "overwrite",
                    "include_debug_info",
                ],
            )
            return
        bundle_path = export_full(full_config)
    else:
        # Resolve asymmetric defaults from SegmentationExportConfig; --n-bits / --group-size
        # are uniform overrides that apply to BOTH encoders when set.
        lite_defaults = SegmentationExportConfig()
        image_n_bits = args.n_bits if args.n_bits is not None else lite_defaults.image_n_bits
        text_n_bits = args.n_bits if args.n_bits is not None else lite_defaults.text_n_bits
        image_group_size = (
            args.group_size if args.group_size is not None else lite_defaults.image_group_size
        )
        text_group_size = (
            args.group_size if args.group_size is not None else lite_defaults.text_group_size
        )

        lite_config = SegmentationExportConfig(
            hf_model_id=hf_model_id,
            image_size=image_size,
            max_text_seq_len=args.max_text_seq_len or lite_defaults.max_text_seq_len,
            image_n_bits=image_n_bits,
            image_group_size=image_group_size,
            text_n_bits=text_n_bits,
            text_group_size=text_group_size,
            output_dir=output_dir,
            output_name=args.output_name,
            overwrite=args.overwrite,
            include_debug_info=args.include_debug_info,
        )
        if args.dry_run:
            _print_dry_run(
                "lite",
                lite_config,
                [
                    "hf_model_id",
                    "image_size",
                    "max_text_seq_len",
                    "image_n_bits",
                    "image_group_size",
                    "text_n_bits",
                    "text_group_size",
                    "output_dir",
                    "output_name",
                    "overwrite",
                    "include_debug_info",
                ],
            )
            return
        bundle_path = export_segmentation(lite_config)

    print(f"Export complete: {bundle_path}")


if __name__ == "__main__":
    main()
