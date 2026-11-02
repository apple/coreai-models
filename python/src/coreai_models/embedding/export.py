# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Export an embedding model to a bundle directory.

  uv run coreai.embedding.export --list-models
  uv run coreai.embedding.export clip-vit-b32            # text + image towers
  uv run coreai.embedding.export clap-htsat --text-only  # text tower only
  uv run coreai.embedding.export clap-htsat --fused      # a single fused asset, including a `main`

``coreai_models.embedding.contract`` describes the bundle.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

from coreai_models._constants import DEFAULT_INCLUDE_DEBUG_INFO
from coreai_models.embedding.clip_clap import CLAP_EXPORTER, CLIP_EXPORTER
from coreai_models.embedding.contract import (
    PLATFORMS,
    SUPPORTED_COMPUTE_PRECISIONS,
    build_bundle_metadata,
    bundle_name,
    resolve_compute_precision,
    validate_block,
)
from coreai_models.embedding.export_utils import (
    EmbeddingExporter,
    resolve_snapshot,
    save_tokenizer,
)

logger = logging.getLogger(__name__)

EXPORTERS: dict[str, EmbeddingExporter] = {
    exporter.family.key: exporter
    for exporter in (
        CLAP_EXPORTER,
        CLIP_EXPORTER,
    )
}


def exporter_for_model(name: str) -> EmbeddingExporter:
    """Return the exporter whose key or HF id matches ``name`` (case-insensitive)."""
    wanted = name.lower()
    for exporter in EXPORTERS.values():
        if wanted in (exporter.family.key.lower(), exporter.family.hf_model_id.lower()):
            return exporter
    raise KeyError(f"No embedding model {name!r}; known: {sorted(EXPORTERS)}")


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


def check_options(exporter: EmbeddingExporter, *, text_only: bool, fused: bool) -> None:
    """Raise ``ValueError`` if ``fused`` is unsupported or combined with ``text_only``."""
    if fused and not exporter.fused:
        raise ValueError(f"{exporter.family.key} has no fused export")
    if fused and text_only:
        raise ValueError("a fused bundle holds every tower; drop --text-only or --fused")


def export_embedding_bundle(
    exporter: EmbeddingExporter,
    output_dir: Path,
    *,
    platform: str = "macOS",
    text_only: bool = False,
    fused: bool = False,
    compute_precision: str | None = None,
    overwrite: bool = False,
    include_debug_info: bool = DEFAULT_INCLUDE_DEBUG_INFO,
    model_dir: str | None = None,
) -> dict[str, Any]:
    """Export one bundle to ``output_dir`` and return its ``metadata.json`` content.

    ``output_dir`` receives the exporter's assets, ``tokenizer/`` and ``metadata.json``. A failed
    export removes it.

    Args:
        compute_precision: Default: the family's default on ``platform``.
        model_dir: A local Hugging Face cache snapshot (``.../snapshots/<sha>``). Default:
            download the snapshot.

    Raises:
        ValueError: on unsupported options, a ``model_dir`` outside the HF cache, or a snapshot
            that disagrees with the family.
        EmbeddingPrecisionError: if ``compute_precision`` is unsupported on ``platform``.
        FileExistsError: if ``output_dir`` exists and ``overwrite`` is False.
        FileNotFoundError: if ``model_dir`` is not a directory.
    """
    family = exporter.family
    precision = resolve_compute_precision(family, compute_precision, platform)
    check_options(exporter, text_only=text_only, fused=fused)
    out = Path(output_dir)
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"{out} exists (pass --overwrite)")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    try:
        snapshot, sha = resolve_snapshot(family.hf_model_id, model_dir)
        if not sha:
            raise ValueError(
                f"Could not read the snapshot revision from {snapshot}; pass a Hugging Face "
                "cache snapshot directory (.../snapshots/<sha>), whose name is the revision the "
                "bundle records"
            )
        exporter.check_snapshot(snapshot)
        save_tokenizer(snapshot, out / "tokenizer", family)
        assets, block = exporter.export_programs(
            snapshot,
            out,
            platform=platform,
            text_only=text_only,
            fused=fused,
            compute_precision=precision,
            include_debug_info=include_debug_info,
            revision=sha,
        )
        validate_block(block, assets)
        metadata = build_bundle_metadata(
            family,
            block,
            name=bundle_name(family, platform, text_only, fused),
            assets=assets,
            hf_revision=sha,
        )
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)  # a failed export leaves no partial bundle
        raise
    logger.info("wrote %s", out)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coreai.embedding.export",
        description="Export an embedding model to a kind=embedding bundle.",
    )
    parser.add_argument(
        "model",
        nargs="?",
        help="Registry short name (e.g. clip-vit-b32) or HuggingFace model ID",
    )
    parser.add_argument(
        "--platform",
        choices=list(PLATFORMS),
        default="macOS",
        help="Target platform; the bundle name gets -ios for iOS.",
    )
    parser.add_argument("--text-only", action="store_true", help="Export the text tower only.")
    parser.add_argument(
        "--skip-vision", dest="text_only", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--fused",
        action="store_true",
        help="Write one asset holding the towers and `main`, which returns the Hugging Face "
        "model's logits and embeddings in one call.",
    )
    parser.add_argument(
        "--compute-precision",
        choices=list(SUPPORTED_COMPUTE_PRECISIONS),
        default=None,
        help="Precision to compute in. Default: the model's first one in --list-models.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Parent directory for the bundle (default: <repo_root>/exports).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing bundle.")
    parser.add_argument("--dry-run", action="store_true", help="Print the export plan and exit.")
    parser.add_argument("--list-models", action="store_true", help="List the models and exit.")
    parser.add_argument(
        "--include-debug-info",
        action="store_true",
        help=(
            "Embed debug information in the exported .aimodel for debugging a conversion. "
            "Default: off, which embeds minimum debug information and makes the "
            "exported asset smaller."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _print_models() -> None:
    for exporter in EXPORTERS.values():
        family = exporter.family
        precisions = {p: ",".join(v) for p, v in family.precisions.items()}
        print(
            f"{family.key:20s} {family.hf_model_id:32s} {','.join(family.modalities):11s} "
            + " ".join(f"{p}:{v}" for p, v in precisions.items())
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_models:
        _print_models()
        return 0
    if args.model is None:
        parser.error("a model is required (e.g. clip-vit-b32); see --list-models")
    try:
        exporter = exporter_for_model(args.model)
    except KeyError:
        parser.error(
            f"unknown model {args.model!r}. Run `uv run coreai.embedding.export --list-models`."
        )
    family = exporter.family
    try:
        precision = resolve_compute_precision(family, args.compute_precision, args.platform)
        check_options(exporter, text_only=args.text_only, fused=args.fused)
    except ValueError as error:
        parser.error(str(error))
    name = bundle_name(family, args.platform, args.text_only, args.fused)
    out = Path(args.output_dir or _default_output_dir()) / name
    if args.dry_run:
        towers = ("text",) if args.text_only else family.modalities
        print(f"platform:  {args.platform}")
        print(f"precision: {precision}")
        print(f"towers:    {','.join(towers)}")
        print(f"fused:     {args.fused}")
        print(f"output:    {out}")
        return 0
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    export_embedding_bundle(
        exporter,
        out,
        platform=args.platform,
        text_only=args.text_only,
        fused=args.fused,
        compute_precision=precision,
        overwrite=args.overwrite,
        include_debug_info=args.include_debug_info,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
