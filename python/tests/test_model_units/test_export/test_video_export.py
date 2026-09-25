# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Unit tests for the SAM3 video export wrappers.

Everything here runs on a randomly-initialized, heavily downscaled config
(112x112, 2 backbone layers, 1 memory-attention layer) so no weights are
downloaded and the whole file runs in seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from coreai_models.segmentation.video_pipeline import (  # noqa: E402
    _TRACKER_MEMORY_FIELDS,
    _TRACKING_FIELDS,
    ENTRYPOINT_IO,
    MASK_NEG,
    VideoExportConfig,
    _apply_image_size,
    _bundle_name,
    _memory_attention,
    _tracking_metadata,
    _validate_image_size,
    _validate_slots,
    build_example_inputs,
    build_modules,
    model_geometry,
)

IMAGE_SIZE = 112
SPATIAL_SLOTS = 10
PTR_SLOTS = 6

#: Shared with the Swift runtime's `VideoSegmenterBundle metadata` suite, which drives the
#: same keys through `parameters()`. It is the one written-down list joining the field
#: tuples below to Swift's `Tracking.CodingKeys`; the two are otherwise maintained
#: independently and drift with no CI signal.
TRACKING_KEYS_FIXTURE = (
    Path(__file__).parents[4] / "swift/Tests/VideoSegmenterTests/Resources/tracking_keys.json"
)


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


# --- bundle metadata -------------------------------------------------------


def test_tracking_metadata_matches_the_shared_fixture(tiny_model):
    """Pins the exporter's emitted keys to the fixture the Swift runtime reads.

    Without this the two key lists -- ``_TRACKING_FIELDS`` here and ``Tracking.CodingKeys``
    on the Swift side -- are maintained by hand with nothing joining them. Adding a field
    here and forgetting Swift produces no error at all: every field in the bundle's
    ``tracking`` block is optional by design, so the runtime silently ignores the new key
    and keeps its own default for a threshold the checkpoint meant to override.
    """
    assert TRACKING_KEYS_FIXTURE.is_file(), f"missing fixture at {TRACKING_KEYS_FIXTURE}"
    fixture = json.loads(TRACKING_KEYS_FIXTURE.read_text())
    assert set(_tracking_metadata(tiny_model.config)) == set(fixture)


def test_tracking_metadata_emits_every_declared_field(tiny_model):
    """``_tracking_metadata`` reads each field behind a ``hasattr`` guard, so a field
    renamed upstream drops out of the bundle silently. ``transformers`` is a floating
    ``>=5.5.0,<6.0``, so a routine dependency bump can trigger that with no code change."""
    tracking = _tracking_metadata(tiny_model.config)
    assert set(tracking) == set(_TRACKING_FIELDS) | set(_TRACKER_MEMORY_FIELDS)


# --- input resolution ------------------------------------------------------


@pytest.mark.parametrize("size", [336, 672, 1008])
def test_validate_image_size_accepts_window_aligned_sizes(size):
    _validate_image_size(size, patch_size=14, window_size=24)


@pytest.mark.parametrize("size", [504, 448, 350])
def test_validate_image_size_rejects_window_misaligned_sizes(size):
    """These divide by the patch size but not the window, so `window_partition` would pad
    and silently change the token count the graph was traced for."""
    with pytest.raises(ValueError, match="attention window"):
        _validate_image_size(size, patch_size=14, window_size=24)


@pytest.mark.parametrize(
    ("size", "mask_size", "feats"),
    [
        (1008, 288, [[288, 288], [144, 144], [72, 72]]),
        (672, 192, [[192, 192], [96, 96], [48, 48]]),
        (336, 96, [[96, 96], [48, 48], [24, 24]]),
    ],
)
def test_apply_image_size_retargets_every_derived_geometry(size, mask_size, feats):
    """`low_res_mask_size` is the one field `image_size`'s setter does not touch; left at
    288 the detector would emit masks larger than the frame."""
    config = transformers.Sam3VideoConfig()
    _apply_image_size(config, size)

    assert config.image_size == size
    assert config.low_res_mask_size == mask_size
    assert config.tracker_config.vision_config.backbone_feature_sizes == feats
    assert config.detector_config.vision_config.backbone_feature_sizes == feats


def _backbone_at(image_size: int, *, layers: int):
    """The ViT backbone alone, retargeted to ``image_size``.

    `_apply_image_size` works on the full video config, but these tests only read the
    backbone, and building the whole `Sam3VideoModel` to reach it is ~30x slower.
    """
    from transformers.models.sam3.modeling_sam3 import Sam3ViTModel

    config = transformers.Sam3VideoConfig()
    _apply_image_size(config, image_size)
    backbone_config = config.detector_config.vision_config.backbone_config
    backbone_config.num_hidden_layers = layers
    return Sam3ViTModel(backbone_config)


def test_apply_image_size_leaves_position_embeddings_alone():
    """The property that makes this whole change cheap.

    Position embeddings are stored at `pretrain_image_size` (336, a 24x24 grid) and tiled up
    at runtime, so the checkpoint tensor is the same shape at every resolution and 336 lands
    on the tiling identity. If this ever stops holding, resizing needs real interpolation.
    """
    shapes = {
        tuple(_backbone_at(size, layers=1).embeddings.position_embeddings.shape)
        for size in (1008, 672, 336)
    }
    assert shapes == {(1, 576, 1024)}


def test_apply_image_size_rebuilds_global_attention_rope():
    """RoPE tables are non-persistent buffers, so they rebuild at the new grid rather than
    loading from the checkpoint -- which is why no checkpoint tensor changes shape."""
    backbone = _backbone_at(336, layers=8)

    # Layer 7 is global (`global_attn_indexes`), the rest windowed. At 336 the grid equals
    # the window, so both collapse to the same 576-position table.
    assert tuple(backbone.layers[7].rotary_emb.rope_embeddings_cos.shape) == (576, 64)
    assert tuple(backbone.layers[0].rotary_emb.rope_embeddings_cos.shape) == (576, 64)


def test_fill_hole_area_scales_with_mask_area():
    """It is an absolute pixel area the host applies at `low_res_mask_size`, so a threshold
    tuned at 288^2 would cover 9x more of the mask at 96^2."""
    stock = _tracking_metadata(transformers.Sam3VideoConfig())["fill_hole_area"]
    assert stock == 16

    scaled = {}
    for size in (672, 336):
        config = transformers.Sam3VideoConfig()
        _apply_image_size(config, size)
        scaled[size] = _tracking_metadata(config)["fill_hole_area"]

    assert scaled[672] == round(stock * (192 / 288) ** 2)
    assert scaled[336] == round(stock * (96 / 288) ** 2)


def test_bundle_name_distinguishes_resolution():
    """The default keeps the name it has always had, so existing bundles don't collide."""
    assert _bundle_name(VideoExportConfig()) == "sam3_video_float16"
    assert _bundle_name(VideoExportConfig(image_size=336)) == "sam3_video_336_float16"
    assert _bundle_name(VideoExportConfig(image_size=672)) == "sam3_video_672_float16"


@pytest.mark.parametrize("entrypoint", sorted(ENTRYPOINT_IO))
def test_every_entrypoint_is_traceable_at_a_second_resolution(entrypoint):
    """The 7-function contract has to hold at more than the one size it was written for."""
    torch.manual_seed(4)
    size = IMAGE_SIZE * 2  # grid 16, still a whole number of the tiny config's 4-patch window
    config = _tiny_config()
    _apply_image_size(config, size)
    model = transformers.Sam3VideoModel(config).eval()

    export_config = VideoExportConfig(
        image_size=size, spatial_slots=SPATIAL_SLOTS, ptr_slots=PTR_SLOTS, dtype="float32"
    )
    modules = build_modules(model, export_config)
    examples = build_example_inputs(
        export_config, torch.float32, **model_geometry(model, export_config)
    )

    program = torch.export.export(modules[entrypoint].eval(), args=examples[entrypoint])
    outputs = program.module()(*examples[entrypoint])
    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
    assert len(outputs) == len(ENTRYPOINT_IO[entrypoint][1]), entrypoint
