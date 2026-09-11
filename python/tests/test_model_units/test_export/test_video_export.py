# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Unit tests for the SAM3 video export wrappers.

Everything here runs on a randomly-initialized, heavily downscaled config
(112x112, 2 backbone layers, 1 memory-attention layer) so no weights are
downloaded and the whole file runs in seconds. Numerical parity against the
real checkpoint is the parity harness's job
(``models/sam3_video/run_video_parity.py``); what these tests pin is the
*structure*: that the fixed-slot memory bank is mathematically equivalent to
HF's variable-length one, and that every entrypoint is traceable.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from coreai_models.segmentation.video_pipeline import (  # noqa: E402
    ENTRYPOINT_IO,
    MASK_NEG,
    VideoExportConfig,
    _memory_attention,
    _validate_slots,
    build_example_inputs,
    build_modules,
    model_geometry,
)

IMAGE_SIZE = 112
SPATIAL_SLOTS = 10
PTR_SLOTS = 6


def _tiny_config() -> transformers.Sam3VideoConfig:
    config = transformers.Sam3VideoConfig()

    detector = config.detector_config
    backbone = detector.vision_config.backbone_config
    backbone.hidden_size = 64
    backbone.num_hidden_layers = 2
    backbone.num_attention_heads = 2
    backbone.intermediate_size = 128
    backbone.window_size = 4
    backbone.global_attn_indexes = [1]

    detector.text_config.hidden_size = 64
    detector.text_config.num_hidden_layers = 2
    detector.text_config.num_attention_heads = 2
    detector.text_config.intermediate_size = 128
    detector.text_config.projection_dim = 64

    detector.detr_encoder_config.num_hidden_layers = 1
    detector.detr_decoder_config.num_hidden_layers = 1
    detector.detr_decoder_config.num_queries = 8

    config.tracker_config.memory_attention_num_layers = 1
    config.image_size = IMAGE_SIZE
    # The detector's mask head emits at 4x the patch grid; the stock 288 belongs
    # to the 1008 config and would make detection masks *larger* than the frame.
    config.low_res_mask_size = (IMAGE_SIZE // backbone.patch_size) * 4
    return config


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    model = transformers.Sam3VideoModel(_tiny_config())
    model.eval()
    return model


@pytest.fixture(scope="module")
def tiny_export_config() -> VideoExportConfig:
    return VideoExportConfig(
        image_size=IMAGE_SIZE,
        spatial_slots=SPATIAL_SLOTS,
        ptr_slots=PTR_SLOTS,
        dtype="float32",
    )


# --- fixed-slot memory bank ------------------------------------------------


@pytest.mark.parametrize("valid_slots", [1, 4, SPATIAL_SLOTS])
def test_padded_memory_matches_variable_length_memory(tiny_model, valid_slots):
    """The whole static-shape scheme rests on this.

    HF concatenates only the memories that exist, so the KV length changes
    every frame. We always send ``spatial_slots`` slots and mask the empty
    ones. Padding must be a no-op — including in the RoPE key path, where
    ``repeat_freqs_k`` derives its repeat factor from the key length and so
    sees a *different* number for the padded tensor.
    """
    torch.manual_seed(1)
    tracker = tiny_model.tracker_model
    memory_attention = tracker.memory_attention
    hw = tracker.backbone_feature_sizes[-1][0] * tracker.backbone_feature_sizes[-1][1]
    mem_dim = tracker.mem_dim
    num_ptr_tokens = PTR_SLOTS * (tracker.hidden_dim // tracker.mem_dim)

    vision = torch.randn(hw, 1, tracker.hidden_dim)
    vision_pos = torch.randn(hw, 1, tracker.hidden_dim)

    spatial = torch.randn(SPATIAL_SLOTS, hw, 1, mem_dim)
    spatial_pos = torch.randn(SPATIAL_SLOTS, hw, 1, mem_dim)
    pointers = torch.randn(num_ptr_tokens, 1, mem_dim)
    pointer_pos = torch.randn(num_ptr_tokens, 1, mem_dim)

    # Reference: exactly the memories HF would have concatenated.
    reference = _memory_attention(
        memory_attention,
        current_vision_features=vision,
        current_vision_position_embeddings=vision_pos,
        memory=torch.cat(
            [spatial[:valid_slots].reshape(valid_slots * hw, 1, mem_dim), pointers], dim=0
        ),
        memory_position_embeddings=torch.cat(
            [spatial_pos[:valid_slots].reshape(valid_slots * hw, 1, mem_dim), pointer_pos], dim=0
        ),
        num_object_pointer_tokens=num_ptr_tokens,
        key_mask=None,
    )

    # Actual: all slots sent, the unpopulated ones masked out. Fill the padded
    # region with large garbage so a masking bug can't hide behind small values.
    padded = spatial.clone()
    padded[valid_slots:] = 1e3
    padded_pos = spatial_pos.clone()
    padded_pos[valid_slots:] = 1e3

    valid = torch.zeros(SPATIAL_SLOTS)
    valid[:valid_slots] = 1.0
    key_valid = torch.cat(
        [valid.reshape(SPATIAL_SLOTS, 1).expand(-1, hw).reshape(-1), torch.ones(num_ptr_tokens)]
    )
    key_mask = ((1.0 - key_valid) * MASK_NEG).reshape(1, 1, 1, -1)

    actual = _memory_attention(
        memory_attention,
        current_vision_features=vision,
        current_vision_position_embeddings=vision_pos,
        memory=torch.cat([padded.reshape(SPATIAL_SLOTS * hw, 1, mem_dim), pointers], dim=0),
        memory_position_embeddings=torch.cat(
            [padded_pos.reshape(SPATIAL_SLOTS * hw, 1, mem_dim), pointer_pos], dim=0
        ),
        num_object_pointer_tokens=num_ptr_tokens,
        key_mask=key_mask,
    )

    torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-4)


def test_fully_masked_spatial_memory_does_not_produce_nans(tiny_model):
    """``-inf`` would NaN here; ``MASK_NEG`` is why the constant is finite.

    Not reachable in the SAM3 video flow (an object always has its seeding
    frame in memory), but a silent NaN would be far worse than a wrong number.
    """
    torch.manual_seed(2)
    tracker = tiny_model.tracker_model
    hw = tracker.backbone_feature_sizes[-1][0] * tracker.backbone_feature_sizes[-1][1]
    mem_dim = tracker.mem_dim
    total = SPATIAL_SLOTS * hw

    output = _memory_attention(
        tracker.memory_attention,
        current_vision_features=torch.randn(hw, 1, tracker.hidden_dim),
        current_vision_position_embeddings=torch.randn(hw, 1, tracker.hidden_dim),
        memory=torch.zeros(total, 1, mem_dim),
        memory_position_embeddings=torch.zeros(total, 1, mem_dim),
        num_object_pointer_tokens=0,
        key_mask=torch.full((1, 1, 1, total), MASK_NEG),
    )
    assert torch.isfinite(output).all()


# --- entrypoint contract ---------------------------------------------------


def test_entrypoint_io_covers_every_module(tiny_model, tiny_export_config):
    modules = build_modules(tiny_model, tiny_export_config)
    assert set(modules) == set(ENTRYPOINT_IO)


def test_example_input_count_matches_declared_input_names(tiny_model, tiny_export_config):
    """A mismatch here means the converter would label arguments wrongly, which
    the runtime only discovers as a shape error deep inside a graph."""
    examples = build_example_inputs(
        tiny_export_config, torch.float32, **model_geometry(tiny_model, tiny_export_config)
    )
    for name, (input_names, _) in ENTRYPOINT_IO.items():
        assert len(examples[name]) == len(input_names), name


@pytest.mark.parametrize("entrypoint", sorted(ENTRYPOINT_IO))
def test_every_entrypoint_is_traceable(tiny_model, tiny_export_config, entrypoint):
    """``torch.export`` is where data-dependent control flow surfaces."""
    torch.manual_seed(3)
    modules = build_modules(tiny_model, tiny_export_config)
    examples = build_example_inputs(
        tiny_export_config, torch.float32, **model_geometry(tiny_model, tiny_export_config)
    )
    module = modules[entrypoint].eval()
    program = torch.export.export(module, args=examples[entrypoint])

    _, output_names = ENTRYPOINT_IO[entrypoint]
    outputs = program.module()(*examples[entrypoint])
    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
    assert len(outputs) == len(output_names), entrypoint


def test_memory_encode_binarize_flag_selects_the_branch(tiny_model, tiny_export_config):
    """``is_mask_from_pts`` is a per-call decision upstream — batching one new
    object flips it for every object on the frame — so it has to stay an input,
    and the two branches must actually differ."""
    torch.manual_seed(4)
    module = build_modules(tiny_model, tiny_export_config)["memory_encode"].eval()
    vision, mask, scores, _ = build_example_inputs(
        tiny_export_config, torch.float32, **model_geometry(tiny_model, tiny_export_config)
    )["memory_encode"]

    with torch.inference_mode():
        smoothed, _ = module(vision, mask, scores, torch.zeros(()))
        binarized, _ = module(vision, mask, scores, torch.ones(()))
    assert not torch.allclose(smoothed, binarized)


# --- slot validation -------------------------------------------------------


def test_spatial_slots_below_checkpoint_capacity_is_rejected(tiny_model):
    """Undersizing the bank silently drops memories HF would have attended to,
    so it fails the export instead of degrading quality invisibly."""
    tracker_config = tiny_model.config.tracker_config
    required = tracker_config.max_cond_frame_num + tracker_config.num_maskmem - 1
    with pytest.raises(ValueError, match="below the"):
        _validate_slots(VideoExportConfig(spatial_slots=required - 1), tracker_config)


def test_ptr_slots_below_encoder_capacity_is_rejected(tiny_model):
    tracker_config = tiny_model.config.tracker_config
    with pytest.raises(ValueError, match="ptr_slots"):
        _validate_slots(
            VideoExportConfig(ptr_slots=tracker_config.max_object_pointers_in_encoder - 1),
            tracker_config,
        )


def test_default_slots_satisfy_the_shipped_checkpoint(tiny_model):
    _validate_slots(VideoExportConfig(), tiny_model.config.tracker_config)
