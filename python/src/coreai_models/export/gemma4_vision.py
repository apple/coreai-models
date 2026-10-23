# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Gemma4 vision-encoder export (adds ``vision.aimodel`` to a text bundle).

The Gemma4 iOS text decoder exports to a ``kind:"llm"`` ladder bundle
(``models/gemma4/export.py``). This module adds the *vision* half: it re-authors
the bespoke RoPE ViT + projector (``models/ios/gemma4_vision.py``) into a ladder of
stateless, static-shape ``encode_image_<size>`` graphs (one per canvas-size bucket,
mirroring the text ladder's one-program-per-context-bucket shape), saves them as
``vision.aimodel`` alongside the text ladder, and upgrades ``metadata.json`` to
``kind:"vlm"`` with the ``vision`` block (including the bucket list) the Swift
``VisionConfig`` consumes.

Called as a post-step from ``models/gemma4/export.py`` when ``--include-vision`` is
set. The vision graphs are device-agnostic; Neural-Engine residency is a
compile-time property, exactly like the text ladder.

Why a ladder instead of one fixed size: a single canvas forces every image to be
squashed/stretched to that exact resolution regardless of its own aspect ratio and
size, which both wastes compute on small images and blurs detail out of large ones.
Instead, the runner letterboxes (aspect-preserving resize + zero-pad) an image into
the smallest bucket that fits it without upscaling, and passes an ``attn_mask``
excluding the padding patches from attention (see ``Gemma4VisionEncoderForiOS``'s
docstring for why padding is always pool-block-aligned, so no patch's pooled output
is ever a mix of real and padding content).
"""

import json
import logging
import shutil
from pathlib import Path

import coreai_torch
import torch
from transformers import AutoConfig

from coreai_models.export.metadata import build_aimodel_metadata
from coreai_models.export.mlir_ops import register_custom_torch_lowering, remove_functionalization
from coreai_models.models.ios.gemma4_vision import Gemma4VisionEncoderForiOS

logger = logging.getLogger(__name__)

# Gemma4 pixels: the runner's ImagePreprocessor already divides u8 by 255
# (`scale = rescaleFactor/255`), so rescale_factor must be 1.0 to land in [0,1] -- the
# graph then centers via 2*(x-0.5). (Setting it to 1/255 double-divides -> near-black.)
# No per-channel mean/std normalization.
_RESCALE_FACTOR = 1.0
_IMAGE_MEAN = (0.0, 0.0, 0.0)
_IMAGE_STD = (1.0, 1.0, 1.0)

# Shipping canvas-size ladder (px). Each must be a multiple of
# `patch_size * pooling_kernel_size` (48 for E2B/E4B: patch 16, pool 3) so an
# aspect-preserving letterbox pads only in whole pool-blocks. `--vision-image-sizes`
# overrides this for a narrower/dev export.
SHIPPING_IMAGE_SIZE_LADDER = (384, 768, 1152)


def _validated_image_sizes(
    image_sizes: tuple[int, ...], *, patch_size: int, pooling_kernel_size: int
) -> tuple[int, ...]:
    """Sort ascending and check every bucket is a multiple of
    ``patch_size * pooling_kernel_size`` -- required so an aspect-preserving
    letterbox (see ``Gemma4VisionEncoderForiOS``'s docstring) can always pad in
    whole pool-blocks. Factored out of ``export_gemma4_vision_encoder`` so it's
    unit-testable without a checkpoint download.
    """
    if not image_sizes:
        raise ValueError("image_sizes must not be empty.")
    if patch_size <= 0 or pooling_kernel_size <= 0:
        raise ValueError(
            f"patch_size and pooling_kernel_size must be positive "
            f"(got {patch_size} and {pooling_kernel_size})."
        )
    if len(set(image_sizes)) != len(image_sizes):
        raise ValueError(f"--vision-image-sizes lists a size more than once: {image_sizes}.")
    block = patch_size * pooling_kernel_size
    for size in image_sizes:
        if size <= 0:
            raise ValueError(f"--vision-image-sizes: {size} is not positive.")
        if size % block != 0:
            raise ValueError(
                f"--vision-image-sizes: {size} is not a multiple of "
                f"patch_size*pooling_kernel_size ({patch_size}*{pooling_kernel_size}={block})."
            )
    return tuple(sorted(image_sizes))


def validate_image_sizes_for_model(
    hf_model_id: str, image_sizes: tuple[int, ...]
) -> tuple[int, ...]:
    """Check ``image_sizes`` against the checkpoint's vision geometry, from its config alone, so a
    bad ladder fails before the text export rather than after it."""
    vision_config = AutoConfig.from_pretrained(hf_model_id).vision_config
    return _validated_image_sizes(
        image_sizes,
        patch_size=vision_config.patch_size,
        pooling_kernel_size=vision_config.pooling_kernel_size,
    )


def encoder_function_name(image_size: int) -> str:
    """Name of the ``vision.aimodel`` function for one canvas size. The Swift runtime derives the
    same name from the size (``VisionConfig.encoderFunctionName(imageSize:)``)."""
    return f"encode_image_{image_size}"


def _write_vision_asset(
    export_module: torch.nn.Module,
    encoder: Gemma4VisionEncoderForiOS,
    image_sizes: tuple[int, ...],
    *,
    target_dtype: torch.dtype,
    vision_path: Path,
    hf_model_id: str,
    overwrite: bool,
) -> None:
    """Trace one static program per canvas size and save them as one multi-function asset.

    One TorchConverter accumulates every size's traced program as its own named entrypoint,
    mirroring how models/gemma4/export.py's text ladder registers one extend_<ctx>/prompt_opt_<ctx>
    pair per context bucket. A single ``to_coreai()`` + ``save_asset()`` then produces one asset
    that shares the (already palettized) weight tensors across every function.
    """
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    register_custom_torch_lowering(converter)
    decomp_table = coreai_torch.get_decomp_table()

    for size in image_sizes:
        if encoder.image_size != size:
            encoder.resize_to(size)
        pixel_values = torch.zeros(1, 3, size, size, dtype=target_dtype)
        attn_mask = torch.zeros(1, encoder.num_patches, 1, encoder.num_patches, dtype=target_dtype)
        with torch.no_grad():
            out = export_module(pixel_values, attn_mask)
        logger.info(
            f"Vision encoder size {size}: input [1,3,{size},{size}] -> "
            f"output {tuple(out.shape)} {out.dtype} ({encoder.num_soft_tokens} soft tokens)"
        )

        entrypoint = encoder_function_name(size)
        logger.info(f"Tracing {entrypoint}...")
        with torch.no_grad():
            exported = torch.export.export(
                export_module,
                args=(),
                kwargs={"pixel_values": pixel_values, "attn_mask": attn_mask},
                dynamic_shapes=None,
            ).run_decompositions(decomp_table)
        remove_functionalization(exported)
        converter.add_exported_program(
            exported,
            input_names=["pixel_values", "attn_mask"],
            output_names=["image_features"],
            entrypoint_name=entrypoint,
        )

    coreai_program = converter.to_coreai()

    if vision_path.exists():
        if not overwrite:
            raise FileExistsError(f"{vision_path} exists. Use --overwrite.")
        shutil.rmtree(vision_path)

    logger.info(f"Saving {vision_path} ({len(image_sizes)} size(s): {list(image_sizes)})...")
    coreai_program.save_asset(vision_path, build_aimodel_metadata(hf_model_id, component="vision"))


def export_gemma4_vision_encoder(
    hf_model_id: str,
    bundle_path: Path,
    *,
    image_sizes: tuple[int, ...] = SHIPPING_IMAGE_SIZE_LADDER,
    overwrite: bool = False,
    target_dtype: torch.dtype = torch.float16,
    compression: str = "4bit_weight_palettized_group32",
) -> str:
    """Export the Gemma4 vision encoder's size ladder into an existing text bundle.

    Args:
        hf_model_id: The multimodal checkpoint (e.g. ``google/gemma-4-E2B-it``).
        bundle_path: The text bundle directory produced by the LLM export
            (contains ``<name>.aimodel``, the PLE safetensors, and ``metadata.json``).
        image_sizes: Canvas-size buckets to export, smallest first (default
            :data:`SHIPPING_IMAGE_SIZE_LADDER`). Each must be a multiple of
            ``patch_size * pooling_kernel_size``.
        overwrite: Replace an existing ``vision.aimodel``.
        target_dtype: Vision-graph weight/compute dtype (fp16 for the runner).
        compression: iOS palettization preset applied to the tower weights (default
            ``4bit_weight_palettized_group32``, matching the text decoder). ``"none"``
            or ``"fp16"`` ships the fp16 tower uncompressed.

    Returns:
        The relative asset filename (``"vision.aimodel"``).
    """
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}. Export the text decoder first.")
    meta_path = bundle_path / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found. Export the text decoder first (it writes metadata.json, "
            "which this function patches at the end -- failing here avoids losing the time spent "
            "building/palettizing/exporting the vision tower only to crash writing it back)."
        )

    # Config-only load (no weights) for the checkpoint's own special-token ids and
    # patch/pool geometry -- these vary per checkpoint, so they must not be
    # hardcoded (they happen to match Gemma4Config's defaults today, but a
    # future/custom checkpoint could differ). Validating image_sizes against this
    # cheap, weights-free load -- rather than after Gemma4VisionEncoderForiOS.from_hf
    # downloads and processes the full tower below -- means a bad size fails fast.
    hf_config = AutoConfig.from_pretrained(hf_model_id)
    image_token_id = hf_config.image_token_id
    boi_token_id = hf_config.boi_token_id
    eoi_token_id = hf_config.eoi_token_id
    pad_token_id = getattr(hf_config.text_config, "pad_token_id", None)
    image_sizes = _validated_image_sizes(
        image_sizes,
        patch_size=hf_config.vision_config.patch_size,
        pooling_kernel_size=hf_config.vision_config.pooling_kernel_size,
    )

    logger.info(
        f"Building Gemma4 vision encoder from {hf_model_id} "
        f"(image_sizes={list(image_sizes)}, smallest bucket first)..."
    )
    encoder = Gemma4VisionEncoderForiOS.from_hf(
        hf_model_id, image_size=image_sizes[0], target_dtype=target_dtype
    )
    patch_size = encoder.patch_size
    pool_k = encoder.pool_k

    # Palettize once against the smallest bucket's shapes -- the tower's weights
    # (what gets palettized) don't depend on which bucket is currently baked via
    # resize_to, so the calibration/compression pass runs a single time and every
    # bucket's traced program shares the resulting (palettized) weight tensors.
    export_module: torch.nn.Module = encoder
    if compression not in ("none", "fp16", "", None):
        from coreai_models.export.compression import palettize_pytorch_model
        from coreai_models.export.presets import get_preset

        calib_size = image_sizes[0]
        calib_pixels = torch.zeros(1, 3, calib_size, calib_size, dtype=target_dtype)
        calib_mask = torch.zeros(1, encoder.num_patches, 1, encoder.num_patches, dtype=target_dtype)
        palettization_cfg = get_preset(compression)["torch_palettization_config"]
        logger.info(f"Palettizing vision tower ({compression})...")
        export_module = palettize_pytorch_model(
            encoder, (calib_pixels, calib_mask), palettization_cfg
        )

    _write_vision_asset(
        export_module,
        encoder,
        image_sizes,
        target_dtype=target_dtype,
        vision_path=bundle_path / "vision.aimodel",
        hf_model_id=hf_model_id,
        overwrite=overwrite,
    )

    _patch_metadata(
        bundle_path,
        patch_size=patch_size,
        pooling_kernel_size=pool_k,
        image_sizes=image_sizes,
        image_token_id=image_token_id,
        boi_token_id=boi_token_id,
        eoi_token_id=eoi_token_id,
        pad_token_id=pad_token_id,
    )
    logger.info("Vision export complete; metadata.json upgraded to kind:vlm.")
    return "vision.aimodel"


def _patch_metadata(
    bundle_path: Path,
    *,
    patch_size: int,
    pooling_kernel_size: int,
    image_sizes: tuple[int, ...],
    image_token_id: int,
    boi_token_id: int | None,
    eoi_token_id: int | None,
    pad_token_id: int | None,
) -> None:
    """Upgrade the text bundle's metadata.json to a VLM manifest."""
    meta_path = bundle_path / "metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)

    metadata["kind"] = "vlm"
    metadata.setdefault("assets", {})["vision"] = "vision.aimodel"
    # Every size's function lives in the one vision asset; register all of them.
    lang = metadata.setdefault("language", {})
    fn_map = lang.setdefault("function_map", {})
    fn_map["vision"] = [encoder_function_name(size) for size in image_sizes]
    # Top-level `vision` block consumed by Swift VisionConfig (snake_case keys). `image_size` and
    # `image_token_count` describe the *largest* size for callers that only look at a single
    # scalar (e.g. a quick metadata inspection); `image_sizes` is what the runner uses to pick a
    # canvas, deriving each size's function name and soft-token count from it.
    largest = max(image_sizes)
    vision: dict[str, object] = {
        "image_size": largest,
        "patch_size": patch_size,
        "pooling_kernel_size": pooling_kernel_size,
        "image_token_count": (largest // (patch_size * pooling_kernel_size)) ** 2,
        "image_token_id": image_token_id,
        "image_mean": list(_IMAGE_MEAN),
        "image_std": list(_IMAGE_STD),
        "rescale_factor": _RESCALE_FACTOR,
        "image_sizes": list(image_sizes),
    }
    if boi_token_id is not None:
        vision["boi_token_id"] = boi_token_id
    if eoi_token_id is not None:
        vision["eoi_token_id"] = eoi_token_id
    if pad_token_id is not None:
        vision["pad_token_id"] = pad_token_id
    metadata["vision"] = vision

    # Serialize to a sibling temp file and replace, so a failed write leaves metadata.json intact.
    tmp_path = meta_path.with_name(meta_path.name + ".tmp")
    try:
        with open(tmp_path, "w") as f:
            json.dump(metadata, f, indent=2)
        tmp_path.replace(meta_path)
    finally:
        tmp_path.unlink(missing_ok=True)
