#!/usr/bin/env python3
# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Export the Qwen3-VL vision encoder (448×448 fixed input, static shapes).

Adds vision.aimodel to an existing VLM bundle directory:
  exports/qwen3_vl_2b.llmasset/vision.aimodel

The exported model uses fully-static shapes (all position embeddings and cu_seqlens
pre-computed as constant buffers), making it compatible with torch.export.

  Input:  pixel_values   float32 [1, 3, 448, 448]  (CLIP-normalized NCHW, runner-produced)
  Output: image_features float16 [1, 196, 2048]    (batch × 196 visual tokens × text-hidden-dim)

Usage:
    cd <repo-root>
    uv run python python/export_vision_encoder_224.py [--bundle-path exports/qwen3_vl_2b.llmasset]
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration as HFModel,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionModel,
)

from coreai_models.export.macos import export_to_coreai
from coreai_models.export.metadata import build_aimodel_metadata

HF_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
BUNDLE_NAME = "qwen3_vl_2b"

# Fixed shapes for 448×448 images
PATCH_SIZE = 16
IMAGE_SIZE = 448
SPATIAL_MERGE_SIZE = 2
TEMPORAL_PATCH_SIZE = 2  # Qwen frames-per-image (single image → duplicated)
CHANNELS = 3
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 784
PATCH_DIM = TEMPORAL_PATCH_SIZE * CHANNELS * PATCH_SIZE * PATCH_SIZE  # 1536
NUM_VISUAL_TOKENS = (IMAGE_SIZE // PATCH_SIZE // SPATIAL_MERGE_SIZE) ** 2  # 196

# Pre-computed grid parameters (constant for 448×448)
GRID_T, GRID_H, GRID_W = 1, IMAGE_SIZE // PATCH_SIZE, IMAGE_SIZE // PATCH_SIZE  # 1, 28, 28


class StaticVisionEncoder(nn.Module):
    """Vision encoder with pre-computed static position embeddings for 448×448 inputs.

    Avoids all data-dependent operations (linspace, repeat_interleave, etc.)
    by baking in the constant values at init time.

    Accepts raw CHW pixel values (the layout the Swift runner's ImagePreprocessor
    produces) and reproduces the Qwen image-processor patchify internally, so the
    runner needs no Qwen-specific preprocessing beyond resize + normalize.

    Input:  pixel_values  float32 [1, 3, 448, 448]   (CLIP-normalized, NCHW)
    Output: image_features float32 [196, text_hidden]
    """

    def __init__(self, visual_model: Qwen3VLVisionModel) -> None:
        super().__init__()
        self.patch_embed = visual_model.patch_embed
        self.blocks = visual_model.blocks
        self.merger = visual_model.merger

        # Pre-compute constant tensors for fixed 448×448 grid
        grid_thw = torch.tensor([[GRID_T, GRID_H, GRID_W]], dtype=torch.int32)

        with torch.no_grad():
            # Position embeddings [NUM_PATCHES, vision_hidden]
            pos_embeds = visual_model.fast_pos_embed_interpolate(grid_thw)
            self.register_buffer("pos_embeds", pos_embeds)

            # Rotary position embeddings
            rotary_pos_emb = visual_model.rot_pos_emb(grid_thw)  # [NUM_PATCHES, rot_dim/2]
            seq_len = rotary_pos_emb.shape[0]
            rotary_flat = rotary_pos_emb.reshape(seq_len, -1)
            emb = torch.cat([rotary_flat, rotary_flat], dim=-1)  # [NUM_PATCHES, rot_dim]
            self.register_buffer("rot_cos", emb.cos())
            self.register_buffer("rot_sin", emb.sin())

            # cu_seqlens for variable-length attention: [0, NUM_PATCHES]
            # For single image batch: [0, GRID_T * GRID_H * GRID_W]
            total_patches = GRID_T * GRID_H * GRID_W  # 784
            cu = torch.tensor([0, total_patches], dtype=torch.int32)
            self.register_buffer("cu_seqlens", cu)

    @staticmethod
    def _patchify(pixel_values: torch.Tensor) -> torch.Tensor:
        """Turn NCHW pixels into Qwen's pre-patchified [NUM_PATCHES, PATCH_DIM].

        Reproduces the exact reshape/permute of Qwen2/3-VL's image processor
        (transpose order ``(0,3,6,4,7,2,1,5,8)``) so the resulting patch order
        matches both the precomputed ``pos_embeds`` and the merger's 2×2
        spatial-merge grouping. The single image is duplicated across the
        temporal dimension, matching the processor's last-frame repeat.
        """
        # [1, 3, 448, 448] → [3, 448, 448] → [temporal, 3, 448, 448]
        x = pixel_values.reshape(CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
        x = x.unsqueeze(0).repeat(TEMPORAL_PATCH_SIZE, 1, 1, 1)
        # split H,W into (grid, merge, patch) and T into (grid_t, temporal)
        x = x.reshape(
            GRID_T,
            TEMPORAL_PATCH_SIZE,
            CHANNELS,
            GRID_H // SPATIAL_MERGE_SIZE,
            SPATIAL_MERGE_SIZE,
            PATCH_SIZE,
            GRID_W // SPATIAL_MERGE_SIZE,
            SPATIAL_MERGE_SIZE,
            PATCH_SIZE,
        )
        x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        return x.reshape(NUM_PATCHES, PATCH_DIM)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [1, 3, 448, 448] (NCHW) → patchify → [NUM_PATCHES, PATCH_DIM]
        patches = self._patchify(pixel_values)
        hidden_states = self.patch_embed(patches)  # [NUM_PATCHES, vision_hidden]
        hidden_states = hidden_states + self.pos_embeds

        position_embeddings = (self.rot_cos, self.rot_sin)

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=self.cu_seqlens,
                position_embeddings=position_embeddings,
            )

        # merger pixel_shuffle → [NUM_VISUAL_TOKENS, text_hidden]
        return self.merger(hidden_states)


class BatchedF16VisionEncoder(nn.Module):
    """Conform the encoder output to the runner contract shared with embed/main.

    StaticVisionEncoder emits f32 [NUM_VISUAL_TOKENS, text_hidden]; PR #65 expects
    f16/bf16 [1, image_token_count, hidden] (a leading batch dim, like embed.aimodel).
    The vision math stays in f32; only the final result is batched and cast to f16.
    """

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.encoder(pixel_values)
        if isinstance(out, tuple):
            out = out[0]
        return out.unsqueeze(0).to(torch.float16)


def _patch_fast_pos_embed_interpolate():
    """Monkeypatch to use Python ints — needed for the init-time pre-computation."""

    def patched(self, grid_thw):
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for _t, h, w in zip(grid_ts.tolist(), grid_hs.tolist(), grid_ws.tolist(), strict=False):
            h, w = int(h), int(w)
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)
            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor
            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side
            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=self.pos_embed.weight.device)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype, device=self.pos_embed.weight.device
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        hw_pairs = [
            (int(h), int(w)) for h, w in zip(grid_hs.tolist(), grid_ws.tolist(), strict=False)
        ]
        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in hw_pairs])

        merge_size = self.config.spatial_merge_size
        patch_pos_embeds_permute = []
        for pos_embed, t, (h, w) in zip(patch_pos_embeds, grid_ts.tolist(), hw_pairs, strict=False):
            t = int(t)
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        return torch.cat(patch_pos_embeds_permute)

    Qwen3VLVisionModel.fast_pos_embed_interpolate = patched


async def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s"
    )

    _patch_fast_pos_embed_interpolate()

    bundle_path = Path(args.bundle_path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}. Run export_qwen3vl.py first.")

    # ---- 1. Text hidden size (projection target) from the HF config ----
    text_hidden = AutoConfig.from_pretrained(HF_MODEL_ID).text_config.hidden_size  # 2048

    # ---- 2. Load HF model (vision part only) ----
    logging.info(f"Loading {HF_MODEL_ID} for vision encoder extraction...")
    hf_model = HFModel.from_pretrained(HF_MODEL_ID, dtype=torch.float32)
    hf_model = hf_model.eval()

    wrapper = StaticVisionEncoder(hf_model.model.visual).eval()
    del hf_model

    # ---- 3. Validate output shape before export ----
    pixel_shape = (1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        test_input = torch.randn(*pixel_shape, dtype=torch.float32)
        test_out = wrapper(test_input)
        # merger returns (hidden_states, deepstack_features) in newer transformers
        if isinstance(test_out, tuple):
            test_out = test_out[0]
        logging.info(
            f"Vision encoder output {tuple(test_out.shape)}; "
            f"expected [{NUM_VISUAL_TOKENS}, {text_hidden}]"
        )

    # ---- 4. Wrap merger to handle tuple output ----
    if isinstance(wrapper(torch.randn(*pixel_shape)), tuple):
        original_merger = wrapper.merger

        class MergerWrapper(nn.Module):
            def __init__(self, merger):
                super().__init__()
                self.merger = merger

            def forward(self, x):
                out = self.merger(x)
                return out[0] if isinstance(out, tuple) else out

        wrapper.merger = MergerWrapper(original_merger)

    # ---- 5. Export ----
    export_module = BatchedF16VisionEncoder(wrapper).eval()

    # Final-shape sanity check (batched + f16) before export.
    with torch.no_grad():
        final_out = export_module(torch.randn(*pixel_shape, dtype=torch.float32))
        logging.info(
            f"Export module output: {tuple(final_out.shape)} {final_out.dtype} "
            f"(expected (1, {NUM_VISUAL_TOKENS}, {text_hidden}) torch.float16)"
        )

    pixel_values = torch.randn(*pixel_shape, dtype=torch.float32)
    reference_inputs = {"pixel_values": pixel_values}

    logging.info(
        f"Exporting vision encoder "
        f"(input: {list(pixel_shape)} → output: [1,{NUM_VISUAL_TOKENS},{text_hidden}] f16)..."
    )
    program = export_to_coreai(
        export_module,
        reference_inputs,
        dynamic_shapes=None,
        input_names=("pixel_values",),
        output_names=("image_features",),
    )
    logging.info("Optimizing AIProgram...")
    program.optimize()

    # ---- 6. Save vision.aimodel ----
    vision_path = bundle_path / "vision.aimodel"
    if vision_path.exists() and not args.overwrite:
        raise FileExistsError(f"{vision_path} exists. Use --overwrite.")
    elif vision_path.exists():
        import shutil

        shutil.rmtree(vision_path)

    logging.info(f"Saving to {vision_path}...")
    build_meta = build_aimodel_metadata(HF_MODEL_ID)
    await asyncio.to_thread(program.save_asset, vision_path, build_meta)

    # ---- 7. Patch metadata.json ----
    with open(bundle_path / "metadata.json") as f:
        metadata = json.load(f)
    metadata["assets"]["vision"] = "vision.aimodel"
    with open(bundle_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logging.info("Updated metadata.json with vision asset")
    print(f"\nVision encoder export done: {vision_path.resolve()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bundle-path",
        default=f"exports/{BUNDLE_NAME}.llmasset",
        help="Path to existing VLM bundle directory",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
