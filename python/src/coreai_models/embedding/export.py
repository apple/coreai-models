# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Export a retrieval-embedding model to a ``kind: "embedding"`` bundle.

    uv run coreai.embedding.export embeddinggemma-2               # macOS GPU bundle: text + image
    uv run coreai.embedding.export embeddinggemma-2 --platform iOS  # Neural Engine bundle
    uv run coreai.embedding.export embeddinggemma-2 --skip-vision  # text only

The bundle contract (``metadata.json`` and the component assets) is defined in
``coreai_models.embedding.contract``.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from coreai_models._constants import DEFAULT_INCLUDE_DEBUG_INFO
from coreai_models.embedding.contract import (
    EMBEDDING_FAMILIES,
    PLATFORMS,
    EmbeddingFamily,
    EmbeddingPrecisionError,
    build_bundle_metadata,
    bundle_assets,
    bundle_name,
    check_compute_precision,
    consolidate_prompts,
    default_precision,
    embedding_block,
    encoder_functions,
    family_for_model,
    text_functions,
)
from coreai_models.embedding.export_ios import (
    export_ane_programs,
)
from coreai_models.embedding.export_macos import (
    export_gpu_programs,
)
from coreai_models.embedding.export_utils import (
    copy_tokenizer,
    read_prompts,
    resolve_snapshot,
    verify_image_config,
    verify_pooling_config,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


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


def export_embedding_bundle(
    family: EmbeddingFamily,
    output_dir: Path,
    *,
    platform: str = "macOS",
    with_image: bool = True,
    compute_precision: str | None = None,
    overwrite: bool = False,
    include_debug_info: bool = DEFAULT_INCLUDE_DEBUG_INFO,
    model_dir: str | None = None,
) -> dict[str, Any]:
    """Export one bundle to ``output_dir`` and return its ``metadata.json`` content.

    macOS writes a GPU bundle (``text_encoder``, plus ``image_encoder`` / ``token_table`` /
    ``encoder`` with images); iOS writes the Neural Engine bundle (``token_table``, ``encoder``,
    plus ``image_encoder``). Both carry the tokenizer and a ``metadata.json``.
    """
    if platform not in PLATFORMS:
        raise ValueError(f"platform must be one of {list(PLATFORMS)}, got {platform!r}")
    precision = check_compute_precision(
        family, compute_precision or default_precision(family, platform), platform=platform
    )  # before touching disk
    if with_image and family.image is None:
        raise ValueError(f"{family.hf_model_id} has no image tower")
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
        verify_pooling_config(snapshot, family)
        if with_image:
            verify_image_config(snapshot, family)
        snapshot_prompts, default_prompt = read_prompts(snapshot)
        prompts = consolidate_prompts(family, snapshot_prompts)
        copy_tokenizer(snapshot, out / "tokenizer")

        if platform == "iOS":
            export_ane_programs(
                family,
                snapshot,
                out,
                with_image=with_image,
                include_debug_info=include_debug_info,
            )
        else:
            export_gpu_programs(
                family,
                snapshot,
                out,
                with_image=with_image,
                compute_precision=precision,
                include_debug_info=include_debug_info,
            )

        block = embedding_block(
            family,
            platform=platform,
            with_image=with_image,
            revision=sha[:16],
            prompts=prompts,
            default_prompt_name=default_prompt,
            compute_precision=precision,
        )
        metadata = build_bundle_metadata(
            family,
            block,
            name=bundle_name(family, platform, with_image),
            assets=bundle_assets(platform, with_image),
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
        description="Export a retrieval-embedding model to a kind=embedding bundle.",
    )
    parser.add_argument(
        "model",
        nargs="?",
        help="Registry short name (e.g. embeddinggemma-2) or HuggingFace model ID",
    )
    parser.add_argument(
        "--platform",
        choices=list(PLATFORMS),
        default="macOS",
        help="Target platform. macOS: GPU bundle (float32 or bfloat16). iOS: Neural Engine "
        "bundle (float16); also runs on a Mac.",
    )
    parser.add_argument(
        "--skip-vision",
        action="store_true",
        help="Export the text tower only (skip the image encoder).",
    )
    parser.add_argument(
        "--compute-precision",
        choices=["float32", "bfloat16", "float16"],
        default=None,
        help="Default: float32 on macOS, float16 on iOS.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Parent directory for the bundle (default: <repo_root>/exports).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the export plan and exit.")
    parser.add_argument("--list-models", action="store_true")
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
    for family in EMBEDDING_FAMILIES.values():
        towers = "text,image" if family.image else "text"
        print(f"{family.key:24s} {family.hf_model_id:32s} {towers:12s} {','.join(PLATFORMS)}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_models:
        _print_models()
        return 0
    if args.model is None:
        parser.error("a model is required (e.g. embeddinggemma-2); see --list-models")
    try:
        family = family_for_model(args.model)
    except KeyError:
        parser.error(
            f"unknown model {args.model!r}. Run `uv run coreai.embedding.export --list-models`."
        )
    platform = args.platform
    with_image = not args.skip_vision
    try:
        precision = check_compute_precision(
            family,
            args.compute_precision or default_precision(family, platform),
            platform=platform,
        )
    except EmbeddingPrecisionError as error:
        parser.error(str(error))
    out = Path(args.output_dir or _default_output_dir()) / bundle_name(family, platform, with_image)
    if args.dry_run:
        assets = bundle_assets(platform, with_image)
        towers = "text,image" if with_image else "text"
        print(f"platform:  {platform}")
        print(f"precision: {precision}")
        print(f"towers:    {towers}")
        print(f"assets:    {', '.join(dict.fromkeys(assets.values()))}")
        if platform == "macOS":
            print(f"text functions:    {[f.name for f in text_functions(family, platform)]}")
        if with_image or platform == "iOS":
            print(f"encoder functions: {[f.name for f in encoder_functions(family, platform)]}")
        print(f"output:    {out}")
        return 0
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    export_embedding_bundle(
        family,
        out,
        platform=platform,
        with_image=with_image,
        compute_precision=precision,
        overwrite=args.overwrite,
        include_debug_info=args.include_debug_info,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
