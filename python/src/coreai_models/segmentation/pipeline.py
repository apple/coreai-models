# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Segmentation export pipeline.

The optimized SAM3 export targets the Apple Neural Engine. The HF
``Sam3Model`` is replaced with a re-authored model
(``coreai_models.models.ios.sam3.Sam3Reauthored``) split into three
independently optimizable functions:

  * ``image_encode``  — vision backbone (palettized + fp16)
  * ``text_encode``   — text encoder + projection (palettized + fp16)
  * ``detect``        — FPN + DETR + mask decoder + scoring (fp16)

The output is a single ``.aimodel`` bundle directory containing the
asset, the HF tokenizer, and a ``metadata.json`` (segmenter bundle,
schema 0.2) — the same shape as ``models/sam3/export.py`` produced
previously, but now with three entrypoints instead of one ``main``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Three-function wrapper modules
# ---------------------------------------------------------------------------


class ImageEncoderModule(nn.Module):
    """Wraps the re-authored vision backbone for ``image_encode``."""

    def __init__(self, image_encoder: nn.Module) -> None:
        super().__init__()
        self.image_encoder = image_encoder

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(pixel_values)


class TextEncoderModule(nn.Module):
    """Wraps the re-authored text encoder + projection for ``text_encode``."""

    def __init__(self, text_encoder: nn.Module, text_projection: nn.Module) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.text_projection = text_projection

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # attention_mask is taken so the function signature matches what the
        # iOS-side ImageSegmenter feeds in; SAM3 itself doesn't use it.
        del attention_mask
        text_hidden = self.text_encoder(input_ids)
        return self.text_projection(text_hidden)


class DetectorModule(nn.Module):
    """Wraps FPN + DETR encoder/decoder + mask decoder + scoring for ``detect``."""

    def __init__(self, sam3_reauth: nn.Module) -> None:
        super().__init__()
        self.fpn = sam3_reauth.fpn
        self.detr_encoder = sam3_reauth.detr_encoder
        self.detr_decoder = sam3_reauth.detr_decoder
        self.scoring = sam3_reauth.scoring
        self.mask_decoder = sam3_reauth.mask_decoder
        self.register_buffer("spatial_shapes", sam3_reauth.spatial_shapes.clone())

    def forward(
        self, backbone_features: torch.Tensor, text_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fpn_hidden_states, fpn_position_encoding = self.fpn(backbone_features)
        fpn_hidden_states_trimmed = fpn_hidden_states[:-1]
        fpn_position_encoding_trimmed = fpn_position_encoding[:-1]

        vision_level2 = fpn_hidden_states_trimmed[-1]
        B = vision_level2.shape[0]
        vision_bc1s = vision_level2.reshape(B, 256, 1, -1)
        pos_level2 = fpn_position_encoding_trimmed[-1]
        pos_bc1s = pos_level2.reshape(1, 256, 1, -1).expand(B, -1, -1, -1)

        encoder_output = self.detr_encoder(
            vision_feats=vision_bc1s,
            text_feats=text_features,
            vision_pos=pos_bc1s,
        )
        final_hidden_states, pred_boxes, presence_logits = self.detr_decoder(
            vision_features=encoder_output,
            text_features=text_features,
            vision_pos=pos_bc1s,
            spatial_shapes=self.spatial_shapes,
        )
        pred_logits = (
            self.scoring(
                decoder_hidden_states=final_hidden_states.unsqueeze(0),
                text_features=text_features,
            )
            .squeeze(-1)
            .squeeze(0)
        )
        mask_outputs = self.mask_decoder(
            decoder_queries=final_hidden_states,
            backbone_features=list(fpn_hidden_states_trimmed),
            encoder_hidden_states=encoder_output,
            prompt_features=text_features,
        )
        return mask_outputs["pred_masks"], pred_boxes, pred_logits, presence_logits


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SegmentationExportConfig:
    """Configuration for a segmentation model export."""

    hf_model_id: str = "facebook/sam3"
    image_size: int = 336
    max_text_seq_len: int = 32
    n_bits: int = 4
    group_size: int = 16
    output_dir: str = "exports"
    output_name: str | None = None
    overwrite: bool = False


def _bundle_name(config: SegmentationExportConfig) -> str:
    if config.output_name is not None:
        return config.output_name
    safe = Path(config.hf_model_id).name.lower()
    return f"{safe}_reauthored_{config.image_size}_w{config.n_bits}_static"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def export_segmentation(config: SegmentationExportConfig) -> str:
    """Export the optimized re-authored SAM3 model to a Core AI bundle.

    Returns the path to the bundle directory.
    """
    return asyncio.run(_async_export_segmentation(config))


async def _async_export_segmentation(config: SegmentationExportConfig) -> str:
    # Imports kept inline so `--list-presets` / `--help` don't pay the cost.
    import coreai_torch
    import transformers
    from coreai_opt import ExportBackend
    from coreai_opt.casting import cast_to_16_bit_precision
    from coreai_opt.palettization import (
        KMeansPalettizer,
        KMeansPalettizerConfig,
        ModuleKMeansPalettizerConfig,
        PalettizationSpec,
    )
    from coreai_opt.palettization.spec import PerGroupedChannelGranularity

    from coreai_models.export.metadata import build_aimodel_metadata
    from coreai_models.models.ios.sam3 import Sam3Reauthored

    bundle_dir, asset_path = _resolve_paths(config)
    _prepare_bundle_dir(bundle_dir, config.overwrite)

    image_size = config.image_size
    grid = image_size // 14

    logger.info("Loading re-authored SAM3 (%s, image_size=%d)...", config.hf_model_id, image_size)
    sam3_reauth = Sam3Reauthored.from_pretrained(
        model_id=config.hf_model_id,
        image_size=image_size,
    )
    sam3_reauth.eval()

    pal_spec = PalettizationSpec(
        n_bits=config.n_bits,
        granularity=PerGroupedChannelGranularity(axis=0, group_size=config.group_size),
        enable_per_channel_scale=True,
    )
    pal_config = KMeansPalettizerConfig(
        global_config=ModuleKMeansPalettizerConfig(op_state_spec={"weight": pal_spec}),
    )

    pixel_ref = torch.randn(1, 3, image_size, image_size)
    ids_ref = torch.randint(0, 49408, (1, config.max_text_seq_len), dtype=torch.int32)
    mask_ref = torch.ones(1, config.max_text_seq_len, dtype=torch.int32)
    backbone_ref = torch.randn(1, 1024, 1, grid * grid)
    text_feat_ref = torch.randn(1, 256, 1, config.max_text_seq_len)

    logger.info(
        "Palettizing image encoder (%d-bit, group_size=%d)...", config.n_bits, config.group_size
    )
    img_enc = ImageEncoderModule(sam3_reauth.image_encoder)
    img_enc.eval()
    img_palettizer = KMeansPalettizer(img_enc, pal_config)
    img_enc = img_palettizer.prepare(example_inputs=(pixel_ref,))
    img_enc = img_palettizer.finalize(backend=ExportBackend.CoreAI)

    logger.info("Palettizing text encoder...")
    txt_enc = TextEncoderModule(sam3_reauth.text_encoder, sam3_reauth.text_projection)
    txt_enc.eval()
    txt_palettizer = KMeansPalettizer(txt_enc, pal_config)
    txt_enc = txt_palettizer.prepare(example_inputs=(ids_ref, mask_ref))
    txt_enc = txt_palettizer.finalize(backend=ExportBackend.CoreAI)

    det = DetectorModule(sam3_reauth)
    det.eval()

    logger.info("Exporting image_encode...")
    img_program = torch.export.export(img_enc, args=(pixel_ref,))
    img_program = img_program.run_decompositions(coreai_torch.get_decomp_table())
    img_program = cast_to_16_bit_precision(img_program)

    logger.info("Exporting text_encode...")
    txt_program = torch.export.export(txt_enc, args=(ids_ref, mask_ref))
    txt_program = txt_program.run_decompositions(coreai_torch.get_decomp_table())
    txt_program = cast_to_16_bit_precision(txt_program)

    logger.info("Exporting detect...")
    det_program = torch.export.export(det, args=(backbone_ref, text_feat_ref))
    det_program = det_program.run_decompositions(coreai_torch.get_decomp_table())
    det_program = cast_to_16_bit_precision(det_program)

    logger.info("Converting to Core AI...")
    converter = coreai_torch.TorchConverter()
    converter.add_exported_program(
        img_program,
        entrypoint_name="image_encode",
        input_names=["pixel_values"],
        output_names=["backbone_features"],
    )
    converter.add_exported_program(
        txt_program,
        entrypoint_name="text_encode",
        input_names=["input_ids", "attention_mask"],
        output_names=["text_features"],
    )
    converter.add_exported_program(
        det_program,
        entrypoint_name="detect",
        input_names=["backbone_features", "text_features"],
        output_names=["pred_masks", "pred_boxes", "pred_logits", "presence_logits"],
    )
    coreai_program = converter.to_coreai()
    coreai_program.optimize()

    metadata = build_aimodel_metadata(config.hf_model_id)
    coreai_program.save_asset(asset_path, metadata)
    logger.info("Saved Core AI asset to %s", asset_path)

    _write_tokenizer(bundle_dir / "tokenizer", config.hf_model_id, transformers)
    _write_bundle_metadata(bundle_dir, asset_path.name)
    return str(bundle_dir)


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def _resolve_paths(config: SegmentationExportConfig) -> tuple[Path, Path]:
    name = _bundle_name(config)
    bundle_dir = Path(config.output_dir) / name
    asset_path = bundle_dir / f"{name}.aimodel"
    return bundle_dir, asset_path


def _prepare_bundle_dir(bundle_dir: Path, overwrite: bool) -> None:
    if bundle_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{bundle_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)


def _write_tokenizer(dest: Path, hf_model_id: str, transformers_module) -> None:
    logger.info("Saving tokenizer from %s to %s", hf_model_id, dest)
    tokenizer = transformers_module.AutoTokenizer.from_pretrained(hf_model_id)
    tokenizer.save_pretrained(str(dest))


def _write_bundle_metadata(bundle_dir: Path, asset_filename: str) -> None:
    name = bundle_dir.name
    metadata = {
        "metadata_version": "0.2",
        "kind": "segmenter",
        "name": name,
        "assets": {"main": asset_filename},
    }
    metadata_path = bundle_dir / "metadata.json"
    with open(metadata_path, "w") as fh:
        json.dump(metadata, fh, indent=2)
    logger.info("Wrote bundle metadata to %s", metadata_path)


# ---------------------------------------------------------------------------
# Plain HF Sam3Model export (no re-authoring, no ANE optimizations)
# ---------------------------------------------------------------------------


@dataclass
class BaselineExportConfig:
    """Configuration for the plain ``transformers.Sam3Model`` export.

    Mirrors what the original ``models/sam3/export.py`` produced before the
    ANE-targeted re-authoring landed: a single-entrypoint ``main`` asset
    that takes ``(pixel_values, input_ids)`` and returns the five raw
    ``Sam3Model`` outputs.
    """

    hf_model_id: str = "facebook/sam3"
    image_size: int = 1008
    dtype: str = "float32"  # "float16" | "float32"
    output_dir: str = "exports"
    output_name: str | None = None
    overwrite: bool = False


class _Sam3BaselineModule(nn.Module):
    """Wrap ``transformers.Sam3Model`` and return the five detection tensors."""

    def __init__(self, model_id: str) -> None:
        super().__init__()
        import transformers

        self._model = transformers.Sam3Model.from_pretrained(model_id)

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor):
        outputs = self._model(pixel_values=pixel_values, input_ids=input_ids)
        return (
            outputs.pred_masks,
            outputs.pred_boxes,
            outputs.pred_logits,
            outputs.presence_logits,
            outputs.semantic_seg,
        )


def _baseline_bundle_name(config: BaselineExportConfig) -> str:
    if config.output_name is not None:
        return config.output_name
    safe = Path(config.hf_model_id).name.lower()
    return f"{safe}_{config.dtype}"


def _baseline_resolve_paths(config: BaselineExportConfig) -> tuple[Path, Path]:
    name = _baseline_bundle_name(config)
    bundle_dir = Path(config.output_dir) / name
    asset_path = bundle_dir / f"{name}.aimodel"
    return bundle_dir, asset_path


def export_baseline(config: BaselineExportConfig) -> str:
    """Export the plain HF ``Sam3Model`` to a Core AI bundle.

    Returns the path to the bundle directory.
    """
    return asyncio.run(_async_export_baseline(config))


async def _async_export_baseline(config: BaselineExportConfig) -> str:
    # Inline imports — keep `--help` cheap.
    import coreai_torch
    import transformers
    from coreai_torch import get_decomp_table

    from coreai_models.export.metadata import build_aimodel_metadata

    dtype_map = {"float16": torch.float16, "float32": torch.float32}
    if config.dtype not in dtype_map:
        raise ValueError(f"Invalid dtype {config.dtype!r}; expected one of {sorted(dtype_map)}.")
    torch_dtype = dtype_map[config.dtype]

    bundle_dir, asset_path = _baseline_resolve_paths(config)
    _prepare_bundle_dir(bundle_dir, config.overwrite)

    logger.info(
        "Loading HF %s (image_size=%d, dtype=%s)...",
        config.hf_model_id,
        config.image_size,
        config.dtype,
    )
    model = _Sam3BaselineModule(model_id=config.hf_model_id)
    model.eval()
    model.to(torch_dtype)

    processor = transformers.Sam3Processor.from_pretrained(config.hf_model_id)
    text_inputs = processor.tokenizer(["dummy"], return_tensors="pt")
    example_inputs = {
        "pixel_values": torch.randn(1, 3, config.image_size, config.image_size).to(torch_dtype),
        "input_ids": text_inputs["input_ids"].to(torch.int32),
    }

    logger.info("Running torch.export...")
    with torch.autocast(device_type="cpu", dtype=torch_dtype):
        exported = torch.export.export(model, args=(), kwargs=example_inputs)
    exported = exported.run_decompositions(get_decomp_table())

    logger.info("Converting to Core AI...")
    converter = coreai_torch.TorchConverter().add_exported_program(
        exported_program=exported,
        input_names=["pixel_values", "input_ids"],
        output_names=[
            "pred_masks",
            "pred_boxes",
            "pred_logits",
            "presence_logits",
            "semantic_seg",
        ],
    )
    coreai_program = converter.to_coreai()
    coreai_program.optimize()

    metadata = build_aimodel_metadata(config.hf_model_id)
    coreai_program.save_asset(asset_path, metadata)
    logger.info("Saved Core AI asset to %s", asset_path)

    _write_tokenizer(bundle_dir / "tokenizer", config.hf_model_id, transformers)
    _write_bundle_metadata(bundle_dir, asset_path.name)
    return str(bundle_dir)
