# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""SAM3 **video** segmentation export pipeline.

``transformers.Sam3VideoModel.forward`` takes a mutable
``Sam3VideoInferenceSession`` (dicts / sets / an object registry that renumbers
itself mid-video) rather than tensors, so it is not traceable. What *is* traceable
is the set of tensor kernels it drives. This module exports those as seven entrypoints
in one ``.aimodel``; the session, the memory ring buffer and every tracking heuristic
stay on the host.

| Function            | Runs           | Purpose                                        |
|---------------------|----------------|------------------------------------------------|
| ``image_encode``    | 1x / frame     | ViT backbone, shared by detector *and* tracker |
| ``text_encode``     | 1x / prompt    | CLIP text + projection (cached for the video)  |
| ``detect``          | Px / frame     | FPN + DETR + mask decoder + scoring            |
| ``tracker_encode``  | 1x / frame     | tracker FPN neck + conv_s0/conv_s1             |
| ``tracker_step``    | Mx / frame     | memory attention + SAM mask decoder            |
| ``memory_encode``   | Mx / frame     | encode a predicted mask into spatial memory    |
| ``tracker_mask_init`` | new objects  | seed a track from a detection mask             |

``tracker_step`` takes a **fixed-capacity** memory bank plus an additive key
mask, because HF builds its memory with a variable-length ``torch.cat``
(``modeling_sam3_tracker_video.py:2523``) whose length changes every frame and
per object. Capacity is exact for the spatial half (``_select_closest_cond_frames``
caps conditioning frames at ``max_cond_frame_num``, and there are exactly
``num_maskmem - 1`` recent slots); the object-pointer half is a budget, since
HF's conditioning-frame pointer branch is uncapped.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from coreai_models._constants import DEFAULT_INCLUDE_DEBUG_INFO
from coreai_models.segmentation.pipeline import _prepare_bundle_dir, _write_tokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class VideoExportConfig:
    """Configuration for the ``transformers.Sam3VideoModel`` export.

    ``spatial_slots`` is exact rather than tunable in practice: HF admits at
    most ``max_cond_frame_num`` conditioning memories plus ``num_maskmem - 1``
    recent ones, and the default (4 + 6) is validated against the loaded config
    at export time. ``ptr_slots`` *is* a budget — HF's conditioning-frame
    object-pointer branch has no cap, so the runtime drops the temporally
    furthest pointer on overflow.
    """

    hf_model_id: str = "facebook/sam3"
    image_size: int = 1008
    dtype: str = "float16"  # "float16" | "float32"
    spatial_slots: int = 10
    ptr_slots: int = 24
    max_text_seq_len: int = 32
    output_dir: str = "exports"
    output_name: str | None = None
    overwrite: bool = False
    include_debug_info: bool = DEFAULT_INCLUDE_DEBUG_INFO


#: Additive mask value for padded memory slots. Matches the convention used by
#: the lite SAM3 text encoder — ``-inf`` produces NaNs once a row is fully
#: masked, which happens on frame 1 where only one memory slot is populated.
MASK_NEG = -40000.0


#: ``entrypoint -> (input_names, output_names)``. Single source of truth: the
#: converter, the example inputs and both runtime backends read it, so an
#: argument can't drift between the traced graph and the host that calls it.
ENTRYPOINT_IO: dict[str, tuple[list[str], list[str]]] = {
    "image_encode": (["pixel_values"], ["last_hidden_state"]),
    "text_encode": (["input_ids", "attention_mask"], ["text_features"]),
    "detect": (
        ["last_hidden_state", "text_features", "attention_mask"],
        ["pred_masks", "pred_boxes", "pred_logits", "presence_logits", "semantic_seg"],
    ),
    "tracker_encode": (
        ["last_hidden_state"],
        [
            "vision_feat_0",
            "vision_feat_1",
            "vision_feat_2",
            "vision_pos_0",
            "vision_pos_1",
            "vision_pos_2",
        ],
    ),
    "tracker_step": (
        [
            "vision_feat_0",
            "vision_feat_1",
            "vision_feat_2",
            "vision_pos_2",
            "spatial_memory",
            "spatial_memory_pos",
            "spatial_tpos_idx",
            "spatial_valid",
            "object_pointers",
            "ptr_tpos",
            "ptr_valid",
        ],
        ["pred_masks", "high_res_masks", "object_pointer", "object_score_logits"],
    ),
    "memory_encode": (
        ["vision_feat_2", "mask_logits", "object_score_logits", "binarize_mask"],
        ["maskmem_features", "maskmem_pos_enc"],
    ),
    "tracker_mask_init": (
        ["vision_feat_0", "vision_feat_1", "vision_feat_2", "mask_input"],
        ["object_pointer"],
    ),
}


def _bundle_name(config: VideoExportConfig) -> str:
    if config.output_name is not None:
        return config.output_name
    safe = Path(config.hf_model_id).name.lower()
    return f"{safe}_video_{config.dtype}"


# ---------------------------------------------------------------------------
# Masked memory attention
# ---------------------------------------------------------------------------
#
# `Sam3TrackerVideoRoPEAttention.forward` hard-codes `attention_mask=None`
# (modeling_sam3_tracker_video.py:901) and the layer above it has no mask
# argument at all. Rather than subclass and re-register modules, these helpers
# re-run the same math over the *existing* module's projections, so no weights
# are copied and parity is by construction.


def _rope_attention(
    attn: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    num_k_exclude_rope: int = 0,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """`Sam3TrackerVideoRoPEAttention.forward` with an additive key mask.

    Uses SDPA rather than the eager path: the cross-attention here is
    (H*W) queries against (spatial_slots * H*W + ptr tokens) keys, so a
    materialized score matrix is the difference between fitting and not.
    """
    from transformers.models.sam3_tracker_video.modeling_sam3_tracker_video import (
        apply_rotary_pos_emb_2d,
    )

    batch_size, point_batch_size = query.shape[:2]
    new_shape = (batch_size * point_batch_size, -1, attn.num_attention_heads, attn.head_dim)

    query = attn.q_proj(query).view(*new_shape).transpose(1, 2)
    key = attn.k_proj(key).view(*new_shape).transpose(1, 2)
    value = attn.v_proj(value).view(*new_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query, key = apply_rotary_pos_emb_2d(
        query,
        key,
        cos,
        sin,
        repeat_freqs_k=attn.rope_k_repeat,
        num_k_exclude_rope=num_k_exclude_rope,
    )

    attn_output = F.scaled_dot_product_attention(
        query.to(value.dtype),
        key.to(value.dtype),
        value,
        attn_mask=attention_mask,
        scale=attn.scaling,
    )
    attn_output = attn_output.transpose(1, 2).reshape(
        batch_size, point_batch_size, -1, attn.num_attention_heads * attn.head_dim
    )
    return attn.o_proj(attn_output)


def _memory_attention_layer(
    layer: nn.Module,
    queries: torch.Tensor,
    keys: torch.Tensor,
    key_point_embedding: torch.Tensor,
    rope_position_embeddings: tuple[torch.Tensor, torch.Tensor],
    num_k_exclude_rope: int,
    key_mask: torch.Tensor | None,
) -> torch.Tensor:
    """`Sam3TrackerVideoMemoryAttentionLayer.forward` threading `key_mask` to cross-attn.

    Self-attention is unmasked: every query token is a real vision token.
    """
    query = layer.layer_norm1(queries)
    query = _rope_attention(layer.self_attn, query, query, query, rope_position_embeddings)
    queries = queries + query

    query = layer.layer_norm2(queries)
    query = _rope_attention(
        layer.cross_attn_image,
        query,
        keys + key_point_embedding,
        keys,
        rope_position_embeddings,
        num_k_exclude_rope=num_k_exclude_rope,
        attention_mask=key_mask,
    )
    queries = queries + query

    query = layer.layer_norm3(queries)
    query = layer.linear2(layer.activation(layer.linear1(query)))
    return queries + query


def _memory_attention(
    memory_attention: nn.Module,
    current_vision_features: torch.Tensor,
    current_vision_position_embeddings: torch.Tensor,
    memory: torch.Tensor,
    memory_position_embeddings: torch.Tensor,
    num_object_pointer_tokens: int,
    key_mask: torch.Tensor,
) -> torch.Tensor:
    """`Sam3TrackerVideoMemoryAttention.forward` with a fixed-length masked memory."""
    output = current_vision_features + 0.1 * current_vision_position_embeddings

    output = output.transpose(0, 1)
    memory = memory.transpose(0, 1).unsqueeze(1)
    memory_position_embeddings = memory_position_embeddings.transpose(0, 1).unsqueeze(1)
    rope_position_embeddings = memory_attention.rotary_emb()

    for layer in memory_attention.layers:
        output = _memory_attention_layer(
            layer,
            queries=output.unsqueeze(1) if output.ndim == 3 else output,
            keys=memory,
            key_point_embedding=memory_position_embeddings,
            rope_position_embeddings=rope_position_embeddings,
            num_k_exclude_rope=num_object_pointer_tokens,
            key_mask=key_mask,
        )

    return memory_attention.layer_norm(output).transpose(0, 1)


# ---------------------------------------------------------------------------
# Entrypoint wrappers
# ---------------------------------------------------------------------------


def resize_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Bilinear mask resize, requesting antialiasing only when it does something.

    HF passes ``antialias=True`` unconditionally, which lowers to
    ``aten._upsample_bilinear2d_aa`` — an op the Core AI converter has no
    lowering for. The flag is only meaningful when downsampling; measured on
    the shapes this export uses (1008->1152, 63->288, 288->1008) the two agree
    to 6e-5, well under fp16 resolution. A genuine downsample such as
    1008->288 differs by ~2.4, so those are kept on the host instead of being
    quietly approximated here.
    """
    if size[0] <= mask.shape[-2] or size[1] <= mask.shape[-1]:
        raise ValueError(
            f"resize_mask only handles upsampling; {tuple(mask.shape[-2:])} -> {size} "
            "would need antialiasing, which has no Core AI lowering. Do it on the host."
        )
    return F.interpolate(mask.float(), size=size, mode="bilinear", align_corners=False).to(
        mask.dtype
    )


class ImageEncodeModule(nn.Module):
    """``image_encode`` — ViT backbone only.

    The FPN neck is deliberately left to ``detect`` (mirroring the lite split,
    where ``image_encode`` returns ``backbone_features`` and ``DetectorModule``
    owns the FPN): emitting the 288x288 and 144x144 neck levels across the
    function boundary would cost ~113 MB of fp16 IO per frame.
    """

    def __init__(self, video_model: nn.Module) -> None:
        super().__init__()
        self.backbone = video_model.detector_model.vision_encoder.backbone

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.backbone(pixel_values).last_hidden_state


class TextEncodeModule(nn.Module):
    """``text_encode`` — CLIP text encoder + projection, run once per prompt per video."""

    def __init__(self, video_model: nn.Module) -> None:
        super().__init__()
        self.detector = video_model.detector_model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.detector.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        ).pooler_output


class DetectModule(nn.Module):
    """``detect`` — detector FPN neck + DETR encoder/decoder + scoring + mask decoder.

    The video path never passes ``input_boxes``, so ``geometry_encoder`` is dead
    code here and is excluded from the traced graph.
    """

    def __init__(self, video_model: nn.Module, grid: int) -> None:
        super().__init__()
        self.detector = video_model.detector_model
        self.grid = grid

    def forward(
        self,
        last_hidden_state: torch.Tensor,
        text_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        from transformers.modeling_outputs import BaseModelOutputWithPooling
        from transformers.models.sam3.modeling_sam3 import Sam3VisionEncoderOutput

        batch_size = last_hidden_state.shape[0]
        spatial = last_hidden_state.view(batch_size, self.grid, self.grid, -1).permute(0, 3, 1, 2)
        fpn_hidden_states, fpn_position_encoding = self.detector.vision_encoder.neck(spatial)

        vision_embeds = Sam3VisionEncoderOutput(
            last_hidden_state=last_hidden_state,
            fpn_hidden_states=fpn_hidden_states,
            fpn_position_encoding=fpn_position_encoding,
        )
        text_embeds = BaseModelOutputWithPooling(
            last_hidden_state=text_features, pooler_output=text_features
        )

        outputs = self.detector(
            vision_embeds=vision_embeds,
            text_embeds=text_embeds,
            attention_mask=attention_mask,
        )
        return (
            outputs.pred_masks,
            outputs.pred_boxes,
            outputs.pred_logits,
            outputs.presence_logits,
            outputs.semantic_seg,
        )


class TrackerEncodeModule(nn.Module):
    """``tracker_encode`` — ``Sam3VideoModel.get_vision_features_for_tracker`` (`:545`).

    Includes the ``conv_s0``/``conv_s1`` mask-decoder projections, which HF
    precomputes here so they aren't re-run per SAM click.
    """

    def __init__(self, video_model: nn.Module, grid: int) -> None:
        super().__init__()
        self.video_model = video_model
        self.grid = grid

    def forward(self, last_hidden_state: torch.Tensor) -> tuple[torch.Tensor, ...]:
        from transformers.models.sam3.modeling_sam3 import Sam3VisionEncoderOutput

        vision_embeds = Sam3VisionEncoderOutput(last_hidden_state=last_hidden_state)
        feats, pos = self.video_model.get_vision_features_for_tracker(vision_embeds=vision_embeds)
        return (*feats, *pos)


class TrackerStepModule(nn.Module):
    """``tracker_step`` — memory-conditioned propagation for **one** object.

    Batch 1 is faithful, not a simplification: HF loops
    ``for obj_idx in range(num_objects)`` (`modeling_sam3_tracker_video.py:1790`)
    precisely because per-object prompts differ.

    Reimplements ``_prepare_memory_conditioned_features`` (`:2425`) over a fixed
    slot layout, then defers to the stock ``_single_frame_forward``. The two
    positional-encoding pieces HF computes while assembling memory — the
    per-slot temporal embedding and the object-pointer sine PE + projection —
    are folded in here so the host never needs model weights.
    """

    def __init__(self, video_model: nn.Module, spatial_slots: int, ptr_slots: int) -> None:
        super().__init__()
        self.tracker = video_model.tracker_model
        self.spatial_slots = spatial_slots
        self.ptr_slots = ptr_slots
        self.num_splits = self.tracker.hidden_dim // self.tracker.mem_dim
        self.num_ptr_tokens = ptr_slots * self.num_splits
        # Both are Python bools off the config, so they bake into the graph.
        self.multimask_output = self.tracker._use_multimask(False, None)
        self.temporal_ptr_pe = self.tracker.config.enable_temporal_pos_encoding_for_object_pointers

    def forward(
        self,
        vision_feat_0: torch.Tensor,
        vision_feat_1: torch.Tensor,
        vision_feat_2: torch.Tensor,
        vision_pos_2: torch.Tensor,
        spatial_memory: torch.Tensor,
        spatial_memory_pos: torch.Tensor,
        spatial_tpos_idx: torch.Tensor,
        spatial_valid: torch.Tensor,
        object_pointers: torch.Tensor,
        ptr_tpos: torch.Tensor,
        ptr_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        from transformers.models.sam3_tracker_video.modeling_sam3_tracker_video import (
            get_1d_sine_pe,
        )

        tracker = self.tracker
        mem_dim = tracker.mem_dim
        height, width = tracker.backbone_feature_sizes[-1]
        hw = height * width

        # --- spatial memory: add the per-slot temporal embedding, then flatten slots
        # HF indexes `memory_temporal_positional_encoding[offset - 1]` while
        # concatenating (`:2309`); conditioning frames use offset 0 and are
        # handed index 0 by the host, matching `[(0, out) for out in ...]`.
        temporal_pe = tracker.memory_temporal_positional_encoding[spatial_tpos_idx]  # (S,1,1,D)
        spatial_pos = spatial_memory_pos + temporal_pe
        spatial_mem = spatial_memory.reshape(self.spatial_slots * hw, 1, mem_dim)
        spatial_pos = spatial_pos.reshape(self.spatial_slots * hw, 1, mem_dim)

        # --- object pointers: sine PE over normalized temporal offsets, projected
        if self.temporal_ptr_pe:
            sine_pe = get_1d_sine_pe(ptr_tpos, dim=tracker.hidden_dim).to(object_pointers.dtype)
            ptr_pos = tracker.temporal_positional_encoding_projection_layer(sine_pe)  # (P, mem_dim)
            ptr_pos = ptr_pos.unsqueeze(1)  # (P, 1, mem_dim)
        else:
            ptr_pos = object_pointers.new_zeros(self.ptr_slots, 1, mem_dim)

        ptrs = object_pointers.reshape(self.ptr_slots, 1, self.num_splits, mem_dim)
        ptrs = ptrs.permute(0, 2, 1, 3).reshape(self.num_ptr_tokens, 1, mem_dim)
        ptr_pos = ptr_pos.repeat_interleave(self.num_splits, dim=0)

        memory = torch.cat([spatial_mem, ptrs], dim=0)
        memory_pos = torch.cat([spatial_pos, ptr_pos], dim=0)

        # --- additive key mask over the same layout
        spatial_key_valid = spatial_valid.reshape(self.spatial_slots, 1).expand(-1, hw).reshape(-1)
        ptr_key_valid = ptr_valid.repeat_interleave(self.num_splits)
        key_valid = torch.cat([spatial_key_valid, ptr_key_valid], dim=0)
        key_mask = ((1.0 - key_valid) * MASK_NEG).reshape(1, 1, 1, -1)

        pix_feat = _memory_attention(
            tracker.memory_attention,
            current_vision_features=vision_feat_2,
            current_vision_position_embeddings=vision_pos_2,
            memory=memory,
            memory_position_embeddings=memory_pos,
            num_object_pointer_tokens=self.num_ptr_tokens,
            key_mask=key_mask,
        )
        pix_feat = pix_feat.squeeze(1).transpose(1, 2).view(1, tracker.hidden_dim, height, width)

        high_res_features = [
            feat.permute(1, 2, 0).view(1, feat.size(2), *size)
            for feat, size in zip(
                (vision_feat_0, vision_feat_1), tracker.backbone_feature_sizes[:-1], strict=True
            )
        ]

        outputs = tracker._single_frame_forward(
            image_embeddings=high_res_features + [pix_feat],
            multimask_output=self.multimask_output,
        )
        return (
            outputs.pred_masks,
            outputs.high_res_masks,
            outputs.object_pointer,
            outputs.object_score_logits,
        )


class MemoryEncodeModule(nn.Module):
    """``memory_encode`` — ``_encode_new_memory`` (`:2658`) for one object.

    HF batches this across objects (`_batch_encode_memories`), but every op is
    elementwise along the batch dim, so per-object calls are bit-identical.

    ``is_mask_from_pts`` cannot be baked in: HF picks it per *call*, and
    `_batch_encode_memories:2752` sets it from ``any(...)`` over the batch — so
    adding one new object flips every object on that frame from
    sigmoid-scaling to hard binarization. It becomes a 0/1 input instead, with
    both branches evaluated and selected by `torch.where`.

    The mask arrives **already at the memory-encoder input size**. Upstream,
    `_encode_new_memory` resizes whatever it is handed, and it is handed two
    different resolutions: `_tracker_update_memories:1202` passes the tracker's
    low-res masks (288) while `_batch_encode_memories` passes image-resolution
    ones (1008). A static graph can only accept one, so the host resizes — and
    that also keeps it a single interpolation, matching HF, instead of
    round-tripping through an intermediate size.
    """

    def __init__(self, video_model: nn.Module) -> None:
        super().__init__()
        self.tracker = video_model.tracker_model

    def forward(
        self,
        vision_feat_2: torch.Tensor,
        mask_logits: torch.Tensor,
        object_score_logits: torch.Tensor,
        binarize_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tracker = self.tracker
        config = tracker.config
        height, width = tracker.backbone_feature_sizes[-1]

        pix_feat = vision_feat_2.permute(1, 2, 0).view(1, tracker.hidden_dim, height, width)

        binarized = (mask_logits > 0).to(mask_logits.dtype)
        smoothed = torch.sigmoid(mask_logits)
        mask_for_mem = binarize_mask * binarized + (1.0 - binarize_mask) * smoothed
        mask_for_mem = (
            mask_for_mem * config.sigmoid_scale_for_mem_enc + config.sigmoid_bias_for_mem_enc
        )

        maskmem_features, maskmem_pos_enc = tracker.memory_encoder(pix_feat, mask_for_mem)

        if tracker.occlusion_spatial_embedding_parameter is not None:
            is_obj_appearing = (object_score_logits > 0).to(maskmem_features.dtype)
            maskmem_features = maskmem_features + (1 - is_obj_appearing[..., None]) * (
                tracker.occlusion_spatial_embedding_parameter[..., None, None].expand(
                    *maskmem_features.shape
                )
            )

        # HF downcasts spatial memory to bf16 here (`:2709`) purely to save host
        # memory; the fp16 asset already carries more mantissa than that, so the
        # cast is dropped rather than emulated.
        maskmem_features = maskmem_features.flatten(2).permute(2, 0, 1)
        maskmem_pos_enc = maskmem_pos_enc.flatten(2).permute(2, 0, 1)
        return maskmem_features, maskmem_pos_enc


class TrackerMaskInitModule(nn.Module):
    """``tracker_mask_init`` — the weighted half of ``_use_mask_as_output`` (`:2136`).

    The path a detection takes when it becomes a new tracked object. Upstream
    this function also derives ``pred_masks`` / ``high_res_masks`` /
    ``object_score_logits``, but those are pure resampling and scaling of the
    input mask — no weights — and one of the resamples is a true antialiased
    downsample with no Core AI lowering.

    Takes the mask at detector resolution rather than image resolution: it
    arrives that way from ``det_out["mask"]``, and upsampling inside the graph
    keeps a 1008x1008 tensor off the function boundary.
    """

    def __init__(self, video_model: nn.Module, image_size: int) -> None:
        super().__init__()
        self.tracker = video_model.tracker_model
        self.image_size = image_size

    def forward(
        self,
        vision_feat_0: torch.Tensor,
        vision_feat_1: torch.Tensor,
        vision_feat_2: torch.Tensor,
        mask_input: torch.Tensor,
    ) -> torch.Tensor:
        tracker = self.tracker
        height, width = tracker.backbone_feature_sizes[-1]

        pix_feat = vision_feat_2.permute(1, 2, 0).view(1, tracker.hidden_dim, height, width)
        high_res_features = [
            feat.permute(1, 2, 0).view(1, feat.size(2), *size)
            for feat, size in zip(
                (vision_feat_0, vision_feat_1), tracker.backbone_feature_sizes[:-1], strict=True
            )
        ]

        mask_at_image = resize_mask(mask_input, (self.image_size, self.image_size))
        # Pre-size to `mask_input_size` so `_single_frame_forward` skips its own
        # antialiased resize (`:2057`) and takes the mask straight through.
        input_masks = resize_mask(
            tracker.mask_downsample(mask_at_image), tuple(tracker.prompt_encoder.mask_input_size)
        )

        object_pointer = tracker._single_frame_forward(
            input_masks=input_masks,
            image_embeddings=high_res_features + [pix_feat],
        ).object_pointer

        # `_use_mask_as_output:2178-2183` — occlusion gating keyed off the raw
        # mask, not the decoder's own object score.
        is_obj_appearing = torch.any(mask_input.flatten(1).float() > 0.0, dim=1)[..., None]
        is_obj_appearing = is_obj_appearing.to(object_pointer.dtype)
        object_pointer = is_obj_appearing * object_pointer
        return object_pointer + (1 - is_obj_appearing) * tracker.no_object_pointer


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def export_video(config: VideoExportConfig) -> str:
    """Export SAM3 video segmentation to a Core AI bundle.

    Returns the path to the bundle directory.
    """
    return asyncio.run(_async_export_video(config))


def _validate_slots(config: VideoExportConfig, tracker_config) -> None:
    """The spatial slot count is dictated by the checkpoint, not chosen.

    ``_gather_memory_frame_outputs`` yields at most ``max_cond_frame_num``
    conditioning memories (``_select_closest_cond_frames`` truncates) plus
    exactly ``num_maskmem - 1`` recent ones. Sizing the bank any smaller
    silently drops memory HF would have attended to.
    """
    required = tracker_config.max_cond_frame_num + tracker_config.num_maskmem - 1
    if config.spatial_slots < required:
        raise ValueError(
            f"spatial_slots={config.spatial_slots} is below the {required} slots this "
            f"checkpoint can populate (max_cond_frame_num="
            f"{tracker_config.max_cond_frame_num} + num_maskmem-1="
            f"{tracker_config.num_maskmem - 1})."
        )
    if config.ptr_slots < tracker_config.max_object_pointers_in_encoder:
        raise ValueError(
            f"ptr_slots={config.ptr_slots} is below max_object_pointers_in_encoder="
            f"{tracker_config.max_object_pointers_in_encoder}."
        )


async def _async_export_video(config: VideoExportConfig) -> str:
    # Inline imports — keep `--help` and `--dry-run` cheap.
    import coreai_torch
    import transformers
    from coreai_opt.casting import cast_to_16_bit_precision

    from coreai_models.export.metadata import build_aimodel_metadata

    dtype_map = {"float16": torch.float16, "float32": torch.float32}
    if config.dtype not in dtype_map:
        raise ValueError(f"Invalid dtype {config.dtype!r}; expected one of {sorted(dtype_map)}.")
    torch_dtype = dtype_map[config.dtype]

    bundle_dir, asset_path = _resolve_paths(config)
    _prepare_bundle_dir(bundle_dir, config.overwrite)

    logger.info("Loading %s (image_size=%d)...", config.hf_model_id, config.image_size)
    model = transformers.Sam3VideoModel.from_pretrained(config.hf_model_id)
    if config.image_size != model.config.image_size:
        raise ValueError(
            f"--image-size {config.image_size} does not match the checkpoint's "
            f"{model.config.image_size}. Resizing needs position-embedding "
            f"interpolation that this export does not yet do."
        )
    model.eval()
    model.to(torch_dtype)

    tracker_config = model.config.tracker_config
    _validate_slots(config, tracker_config)

    modules = build_modules(model, config)
    example_inputs = build_example_inputs(config, torch_dtype, **model_geometry(model, config))

    converter = coreai_torch.TorchConverter(
        mode=(
            coreai_torch.TorchConverter.Mode.DEBUG
            if config.include_debug_info
            else coreai_torch.TorchConverter.Mode.RELEASE
        )
    )
    for name, module in modules.items():
        input_names, output_names = ENTRYPOINT_IO[name]
        logger.info("Exporting %s...", name)
        module.eval()
        program = torch.export.export(module, args=example_inputs[name])
        program = program.run_decompositions(coreai_torch.get_decomp_table())
        program = cast_to_16_bit_precision(program)
        converter.add_exported_program(
            program,
            entrypoint_name=name,
            input_names=input_names,
            output_names=output_names,
        )

    logger.info("Converting to Core AI...")
    coreai_program = converter.to_coreai()
    coreai_program.optimize()

    metadata = build_aimodel_metadata(config.hf_model_id, component="video segmentation")
    coreai_program.save_asset(asset_path, metadata)
    logger.info("Saved Core AI asset to %s", asset_path)

    # Metadata before tokenizer, so a flaky HF fetch can't leave an unloadable bundle.
    _write_bundle_metadata(bundle_dir, asset_path.name, config)
    _write_tokenizer(bundle_dir / "tokenizer", config.hf_model_id)
    return str(bundle_dir)


def model_geometry(model, config: VideoExportConfig) -> dict:
    """Shapes every entrypoint is traced against, derived from the loaded model."""
    vision_config = model.config.detector_config.vision_config
    tracker = model.tracker_model
    feat_sizes = tracker.backbone_feature_sizes
    return {
        "grid": config.image_size // vision_config.backbone_config.patch_size,
        "hidden": vision_config.backbone_config.hidden_size,
        "text_dim": model.config.detector_config.detr_encoder_config.hidden_size,
        "mem_dim": tracker.mem_dim,
        "tracker_hidden": tracker.hidden_dim,
        "feat_sizes": feat_sizes,
        "hw": feat_sizes[-1][0] * feat_sizes[-1][1],
        "mask_size": model.config.low_res_mask_size,
        # `_encode_new_memory` feeds the mask downsampler at 4x the prompt
        # encoder's mask size, so that is the graph's fixed input resolution.
        "mem_size": tracker.prompt_encoder.mask_input_size[0] * 4,
        "conv_s_channels": [
            tracker.mask_decoder.conv_s0.out_channels,
            tracker.mask_decoder.conv_s1.out_channels,
        ],
    }


def build_modules(model, config: VideoExportConfig) -> dict[str, nn.Module]:
    """One ``nn.Module`` per entrypoint, sharing the loaded model's weights."""
    grid = (
        config.image_size // model.config.detector_config.vision_config.backbone_config.patch_size
    )
    return {
        "image_encode": ImageEncodeModule(model),
        "text_encode": TextEncodeModule(model),
        "detect": DetectModule(model, grid),
        "tracker_encode": TrackerEncodeModule(model, grid),
        "tracker_step": TrackerStepModule(model, config.spatial_slots, config.ptr_slots),
        "memory_encode": MemoryEncodeModule(model),
        "tracker_mask_init": TrackerMaskInitModule(model, config.image_size),
    }


def build_example_inputs(
    config: VideoExportConfig,
    dtype: torch.dtype,
    *,
    grid: int,
    hidden: int,
    text_dim: int,
    mem_dim: int,
    tracker_hidden: int,
    feat_sizes,
    hw: int,
    mask_size: int,
    mem_size: int,
    conv_s_channels: list[int],
) -> dict[str, tuple]:
    """Static positional example inputs per entrypoint, ordered per ``ENTRYPOINT_IO``.

    Shared with the parity harness so probe shapes can't drift from traced ones.
    """
    size = config.image_size
    seq = config.max_text_seq_len
    slots = config.spatial_slots
    ptrs = config.ptr_slots

    lhs = torch.randn(1, grid * grid, hidden, dtype=dtype)
    text_features = torch.randn(1, seq, text_dim, dtype=dtype)
    attention_mask = torch.ones(1, seq, dtype=torch.int32)

    vf0 = torch.randn(feat_sizes[0][0] * feat_sizes[0][1], 1, conv_s_channels[0], dtype=dtype)
    vf1 = torch.randn(feat_sizes[1][0] * feat_sizes[1][1], 1, conv_s_channels[1], dtype=dtype)
    vf2 = torch.randn(hw, 1, tracker_hidden, dtype=dtype)
    vp2 = torch.randn(hw, 1, tracker_hidden, dtype=dtype)

    return {
        "image_encode": (torch.randn(1, 3, size, size, dtype=dtype),),
        "text_encode": (torch.randint(0, 49408, (1, seq), dtype=torch.int32), attention_mask),
        "detect": (lhs, text_features, attention_mask),
        "tracker_encode": (lhs,),
        "tracker_step": (
            vf0,
            vf1,
            vf2,
            vp2,
            torch.randn(slots, hw, 1, mem_dim, dtype=dtype),
            torch.randn(slots, hw, 1, mem_dim, dtype=dtype),
            torch.zeros(slots, dtype=torch.int32),
            torch.ones(slots, dtype=dtype),
            torch.randn(ptrs, 1, tracker_hidden, dtype=dtype),
            torch.zeros(ptrs, dtype=torch.float32),
            torch.ones(ptrs, dtype=dtype),
        ),
        "memory_encode": (
            vf2,
            torch.randn(1, 1, mem_size, mem_size, dtype=dtype),
            # (1, 1, 1), not (1, 1): this is fed straight from `tracker_step` /
            # `tracker_mask_init`, both of which emit the mask decoder's
            # `object_score_logits` with a trailing singleton.
            torch.randn(1, 1, 1, dtype=dtype),
            torch.zeros((), dtype=dtype),
        ),
        # Detector-resolution mask: it comes straight off `det_out["mask"]`.
        "tracker_mask_init": (
            vf0,
            vf1,
            vf2,
            torch.randint(0, 2, (1, 1, mask_size, mask_size)).to(dtype),
        ),
    }


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def _resolve_paths(config: VideoExportConfig) -> tuple[Path, Path]:
    name = _bundle_name(config)
    bundle_dir = Path(config.output_dir) / name
    return bundle_dir, bundle_dir / f"{name}.aimodel"


def _write_bundle_metadata(
    bundle_dir: Path, asset_filename: str, config: VideoExportConfig
) -> None:
    """Write the bundle manifest.

    ``runtime`` carries the slot geometry because the host has to pack memory to
    exactly the shapes the graph was traced with; deriving it from the HF config
    at load time would silently break if the export used non-default slots.
    """
    metadata = {
        "metadata_version": "0.2",
        "kind": "video_segmenter",
        "name": bundle_dir.name,
        "assets": {"main": asset_filename},
        "runtime": {
            "image_size": config.image_size,
            "spatial_slots": config.spatial_slots,
            "ptr_slots": config.ptr_slots,
            "max_text_seq_len": config.max_text_seq_len,
        },
    }
    metadata_path = bundle_dir / "metadata.json"
    with open(metadata_path, "w") as fh:
        json.dump(metadata, fh, indent=2)
    logger.info("Wrote bundle metadata to %s", metadata_path)
