# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""CLI entry point for ``coreai.segmentation.run``.

Loads an exported segmenter bundle (produced by
``coreai.segmentation.export``), runs the three-function pipeline
``image_encode -> text_encode -> detect`` against an input image and a
text prompt, and writes an annotated PNG with the predicted masks
overlaid.

Pure-Python inference: uses ``coreai.runtime`` to load and run the
``.aimodel`` asset on the host machine. No Swift / iOS pipeline involved.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _find_repo_root() -> Path | None:
    d = Path(__file__).resolve().parent
    while d != d.parent:
        if (d / "pyproject.toml").exists() and (d / "python").exists():
            return d
        d = d.parent
    return None


def _default_exports_dir() -> Path:
    root = _find_repo_root()
    return (root / "exports") if root is not None else Path("exports")


def _resolve_bundle(bundle: str | None, model: str) -> Path:
    """Resolve a bundle path: explicit ``--bundle`` wins, otherwise look in exports/.

    For the registry short-name path we accept any directory whose name
    starts with ``<model>_reauthored_`` so the user doesn't need to remember
    the exact image-size / n-bits suffix.
    """
    if bundle is not None:
        path = Path(bundle)
        if not path.exists():
            raise SystemExit(f"--bundle directory not found: {path}")
        return path

    exports_dir = _default_exports_dir()
    if not exports_dir.exists():
        raise SystemExit(
            f"Default exports dir {exports_dir} does not exist. "
            "Run `coreai.segmentation.export <model>` first or pass --bundle."
        )

    prefix = f"{model.lower()}_reauthored_"
    candidates = sorted(
        p for p in exports_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)
    )
    if not candidates:
        raise SystemExit(
            f"No bundle starting with {prefix!r} in {exports_dir}. "
            "Run `coreai.segmentation.export <model>` first or pass --bundle."
        )
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise SystemExit(
            f"Multiple bundles match {prefix!r} in {exports_dir}: {names}. Pass --bundle."
        )
    return candidates[0]


def _find_asset(bundle_dir: Path) -> Path:
    """Locate the .aimodel inside a segmenter bundle directory."""
    candidates = sorted(bundle_dir.glob("*.aimodel"))
    if not candidates:
        raise SystemExit(f"No .aimodel found inside {bundle_dir}")
    if len(candidates) > 1:
        # Prefer the one whose stem matches the bundle name.
        for c in candidates:
            if c.stem == bundle_dir.name:
                return c
    return candidates[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coreai.segmentation.run",
        description=(
            "Run a pre-exported segmentation model (loaded via coreai.runtime) on "
            "a single image + prompt and write the annotated result to disk."
        ),
    )
    parser.add_argument(
        "model",
        help="Registry short-name (e.g. 'sam3'); used to locate the bundle in exports/.",
    )
    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Text prompt for segmentation (e.g. 'flower').",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="Explicit path to the segmenter bundle directory. Overrides exports/ lookup.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <bundle>/<image-stem>_<prompt>.png).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection score threshold for masks (default: 0.5).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=336,
        help="Input resolution to feed the model (must match the export).",
    )
    parser.add_argument(
        "--max-text-seq-len",
        type=int,
        default=32,
        help="Static text sequence length (must match the export).",
    )
    parser.add_argument(
        "--compute-unit",
        choices=["neural-engine", "gpu", "cpu"],
        default="neural-engine",
        help="Which compute unit to specialize for (default: neural-engine).",
    )
    parser.add_argument(
        "--hf-model-id",
        default="facebook/sam3",
        help="HuggingFace id used to load the processor (default: facebook/sam3).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.image.exists():
        parser.error(f"--image not found: {args.image}")

    bundle_dir = _resolve_bundle(str(args.bundle) if args.bundle is not None else None, args.model)
    asset_path = _find_asset(bundle_dir)
    output_path = (
        args.output
        if args.output is not None
        else bundle_dir / f"{args.image.stem}_{args.prompt.replace(' ', '_')}.png"
    )

    asyncio.run(
        _run(
            asset_path=asset_path,
            image_path=args.image,
            prompt=args.prompt,
            output_path=output_path,
            threshold=args.threshold,
            image_size=args.image_size,
            max_text_seq_len=args.max_text_seq_len,
            compute_unit=args.compute_unit,
            hf_model_id=args.hf_model_id,
        )
    )


async def _run(
    asset_path: Path,
    image_path: Path,
    prompt: str,
    output_path: Path,
    threshold: float,
    image_size: int,
    max_text_seq_len: int,
    compute_unit: str,
    hf_model_id: str,
) -> None:
    # Inline imports — keep `--help` cheap.
    import numpy as np
    import torch
    import torch.nn.functional as F
    import transformers
    from coreai.runtime import (
        AIModel,
        ComputeUnitKind,
        NDArray,
        SpecializationOptions,
    )
    from PIL import Image

    image = Image.open(image_path).convert("RGB")

    pixel_values, input_ids, attention_mask = _prepare_inputs(
        image=image,
        prompt=prompt,
        image_size=image_size,
        max_seq_len=max_text_seq_len,
        hf_model_id=hf_model_id,
        np=np,
        torch=torch,
        F=F,
        transformers=transformers,
    )

    options = _specialization_options(SpecializationOptions, ComputeUnitKind, compute_unit)

    logger.info("Loading %s and specializing for %s", asset_path, compute_unit)
    model = await AIModel.load(asset_path, specialization_options=options)

    img_fn = model.load_function("image_encode")
    txt_fn = model.load_function("text_encode")
    det_fn = model.load_function("detect")
    start = time.time()
    #    logger.info("Running image_encode...")
    vision_out = await img_fn({"pixel_values": NDArray(pixel_values)})
    #    logger.info("Running text_encode...")
    text_out = await txt_fn(
        {"input_ids": NDArray(input_ids), "attention_mask": NDArray(attention_mask)}
    )
    #    logger.info("Running detect...")
    decode_out = await det_fn(
        {
            "backbone_features": vision_out["backbone_features"],
            "text_features": text_out["text_features"],
        }
    )
    duration = time.time() - start
    logger.info(f"SAM3 multi-function took {duration} seconds.")

    pred_masks = decode_out["pred_masks"].numpy()[0]
    pred_logits = decode_out["pred_logits"].numpy()[0]

    _save_overlay(image, pred_masks, pred_logits, threshold, prompt, output_path, np=np)
    print(f"Wrote {output_path}")


def _specialization_options(SpecializationOptions, ComputeUnitKind, compute_unit: str):
    kinds = {
        "neural-engine": ComputeUnitKind.neural_engine(),
        "gpu": ComputeUnitKind.gpu(),
        "cpu": ComputeUnitKind.cpu(),
    }
    return SpecializationOptions.from_preferred_compute_unit_kind(
        compute_unit_kind=kinds[compute_unit],
    )


def _prepare_inputs(
    *,
    image,
    prompt: str,
    image_size: int,
    max_seq_len: int,
    hf_model_id: str,
    np,
    torch,
    F,
    transformers,
):
    """Run the HF processor, resize to ``image_size``, pad text to ``max_seq_len``.

    Returns ``(pixel_values_fp16, input_ids_int32, attention_mask_int32)`` as
    contiguous numpy arrays ready to wrap in ``NDArray``.
    """
    processor = transformers.Sam3Processor.from_pretrained(hf_model_id)
    inputs = processor([image], text=[prompt], return_tensors="pt")

    pixel_values = (
        F.interpolate(
            inputs["pixel_values"],
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
        .numpy()
        .astype(np.float16)
    )

    input_ids = inputs["input_ids"].numpy().astype(np.int32)
    attention_mask = inputs["attention_mask"].numpy().astype(np.int32)
    if input_ids.shape[1] < max_seq_len:
        pad_len = max_seq_len - input_ids.shape[1]
        input_ids = np.pad(input_ids, ((0, 0), (0, pad_len)))
        attention_mask = np.pad(attention_mask, ((0, 0), (0, pad_len)))
    elif input_ids.shape[1] > max_seq_len:
        input_ids = input_ids[:, :max_seq_len]
        attention_mask = attention_mask[:, :max_seq_len]

    return (
        np.ascontiguousarray(pixel_values),
        np.ascontiguousarray(input_ids),
        np.ascontiguousarray(attention_mask),
    )


def _save_overlay(
    image, pred_masks, pred_logits, threshold: float, prompt: str, output_path: Path, *, np
) -> None:
    """Write a PNG with detected masks tinted onto the image."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    scores = 1.0 / (1.0 + np.exp(-pred_logits))
    valid = np.where(scores > threshold)[0]
    h, w = np.array(image).shape[:2]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    overlay_color = (0.12, 0.56, 1.0, 0.4)
    for idx in valid:
        mask_small = (pred_masks[idx] > 0).astype(np.uint8) * 255
        mask = np.array(Image.fromarray(mask_small).resize((w, h))) > 127
        overlay = np.zeros((h, w, 4))
        overlay[mask] = overlay_color
        ax.imshow(overlay)
    ax.set_title(f"{prompt} ({len(valid)} detections)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
