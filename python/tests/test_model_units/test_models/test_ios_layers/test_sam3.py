# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""SAM3 lite text-padding mask parity with HF ``Sam3Model`` submodules (#381).

HF masks prompt padding in four places: DETR encoder and decoder text cross-attention,
mask-decoder prompt cross-attention and the scoring head's mean pool. Each parity test builds
the HF submodule from the default ``Sam3Config`` with random weights (no download), converts
it, and compares both with a mask that leaves most of the 32 slots as padding. The last test
checks that ``DetectorModule`` ignores whatever sits in the padded slots.
"""

import copy

import pytest
import torch
import transformers
import transformers.models.sam3.modeling_sam3 as hf

from coreai_models.models.ios.sam3 import Sam3Lite
from coreai_models.models.ios.sam3.detr import (
    DETRDecoderReauthored,
    DETREncoderReauthored,
)
from coreai_models.models.ios.sam3.fpn import FPNNeckReauthored
from coreai_models.models.ios.sam3.mask_decoder import (
    DotProductScoringReauthored,
    MaskDecoderReauthored,
)
from coreai_models.models.ios.sam3.sam3_reauthored import text_mask_bias
from coreai_models.segmentation.pipeline import DetectorModule

_SEQ = 32
_NUM_VALID = 4  # SOT, two words, EOT
_GRID = 8
_C = 256


@pytest.fixture(scope="module")
def config():
    config = transformers.Sam3Config()
    config._attn_implementation = "eager"
    for sub in (config.detr_encoder_config, config.detr_decoder_config, config.mask_decoder_config):
        sub._attn_implementation = "eager"
    return config


def _mask() -> torch.Tensor:
    mask = torch.zeros(1, _SEQ, dtype=torch.long)
    mask[:, :_NUM_VALID] = 1
    return mask


def _bsc_to_bc1s(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 1).unsqueeze(2)


def _bc1s_to_bsc(x: torch.Tensor) -> torch.Tensor:
    return x.squeeze(2).permute(0, 2, 1)


def _assert_close(lite: torch.Tensor, ref: torch.Tensor) -> None:
    torch.testing.assert_close(lite, ref, atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_detr_encoder_masks_text_padding(config):
    torch.manual_seed(0)
    ref = hf.Sam3DetrEncoder(config.detr_encoder_config).eval()
    lite = DETREncoderReauthored.from_hf_encoder(ref).eval()

    vision = torch.randn(1, _C, _GRID, _GRID)
    pos = torch.randn(1, _C, _GRID, _GRID)
    text = torch.randn(1, _SEQ, _C)
    mask = _mask()

    expected = ref(
        vision_features=[vision],
        text_features=text,
        vision_pos_embeds=[pos],
        text_mask=mask.bool(),
    ).last_hidden_state
    bias, _ = text_mask_bias(mask, torch.float32)
    actual = lite(
        vision_feats=vision.reshape(1, _C, 1, -1),
        text_feats=_bsc_to_bc1s(text),
        vision_pos=pos.reshape(1, _C, 1, -1),
        text_cross_attn_mask=bias,
    )
    _assert_close(_bc1s_to_bsc(actual), expected)


@torch.no_grad()
def test_detr_decoder_masks_text_padding(config):
    torch.manual_seed(0)
    ref = hf.Sam3DetrDecoder(config.detr_decoder_config).eval()
    lite = DETRDecoderReauthored.from_hf_decoder(ref, spatial_h=_GRID, spatial_w=_GRID).eval()

    vision = torch.randn(1, _GRID * _GRID, _C)
    pos = torch.randn(1, _GRID * _GRID, _C)
    text = torch.randn(1, _SEQ, _C)
    mask = _mask()

    out = ref(
        vision_features=vision,
        text_features=text,
        vision_pos_encoding=pos,
        text_mask=mask.bool(),
        spatial_shapes=torch.tensor([[_GRID, _GRID]]),
    )
    bias, _ = text_mask_bias(mask, torch.float32)
    hidden, _, presence = lite(
        vision_features=_bsc_to_bc1s(vision),
        text_features=_bsc_to_bc1s(text),
        vision_pos=_bsc_to_bc1s(pos),
        text_cross_attn_mask=bias,
    )
    _assert_close(_bc1s_to_bsc(hidden), out.intermediate_hidden_states[-1])
    _assert_close(presence.reshape(-1), out.presence_logits[-1].reshape(-1))


@torch.no_grad()
def test_mask_decoder_masks_prompt_padding(config):
    torch.manual_seed(0)
    ref = hf.Sam3MaskDecoder(config.mask_decoder_config).eval()
    lite = MaskDecoderReauthored.from_hf_mask_decoder(ref).eval()

    num_queries = 16
    queries = torch.randn(1, num_queries, _C)
    backbone = [torch.randn(1, _C, _GRID * s, _GRID * s) for s in (4, 2, 1)]
    encoder = torch.randn(1, _GRID * _GRID, _C)
    prompt = torch.randn(1, _SEQ, _C)
    mask = _mask()

    out = ref(
        decoder_queries=queries,
        backbone_features=[f.clone() for f in backbone],
        encoder_hidden_states=encoder,
        prompt_features=prompt,
        prompt_mask=mask.bool(),
    )
    bias, _ = text_mask_bias(mask, torch.float32)
    actual = lite(
        decoder_queries=_bsc_to_bc1s(queries),
        backbone_features=backbone,
        encoder_hidden_states=_bsc_to_bc1s(encoder),
        prompt_features=_bsc_to_bc1s(prompt),
        prompt_attention_mask=bias,
    )
    _assert_close(actual["pred_masks"], out.pred_masks)
    _assert_close(actual["semantic_seg"], out.semantic_seg)


@torch.no_grad()
def test_scoring_pools_only_real_tokens(config):
    torch.manual_seed(0)
    ref = hf.Sam3DotProductScoring(config).eval()
    lite = DotProductScoringReauthored.from_hf_scoring(ref).eval()

    num_queries = 16
    queries = torch.randn(1, 1, num_queries, _C)
    text = torch.randn(1, _SEQ, _C)
    mask = _mask()

    expected = ref(decoder_hidden_states=queries, text_features=text, text_mask=mask.bool())
    _, valid = text_mask_bias(mask, torch.float32)
    actual = lite(
        decoder_hidden_states=_bsc_to_bc1s(queries[0]).unsqueeze(0),
        text_features=_bsc_to_bc1s(text),
        text_valid=valid,
    )
    _assert_close(actual, expected)


@pytest.fixture(scope="module")
def detector(config):
    torch.manual_seed(0)
    lite = Sam3Lite(image_size=_GRID * 14)
    lite.fpn = FPNNeckReauthored.from_hf_fpn(
        hf.Sam3VisionNeck(config.vision_config), grid_h=_GRID, grid_w=_GRID
    )
    lite.detr_encoder = DETREncoderReauthored.from_hf_encoder(
        hf.Sam3DetrEncoder(config.detr_encoder_config)
    )
    lite.detr_decoder = DETRDecoderReauthored.from_hf_decoder(
        hf.Sam3DetrDecoder(config.detr_decoder_config), spatial_h=_GRID, spatial_w=_GRID
    )
    lite.mask_decoder = MaskDecoderReauthored.from_hf_mask_decoder(
        hf.Sam3MaskDecoder(config.mask_decoder_config)
    )
    lite.scoring = DotProductScoringReauthored.from_hf_scoring(hf.Sam3DotProductScoring(config))
    return DetectorModule(lite).eval()


@torch.no_grad()
def test_detector_ignores_padded_text_slots(detector, config):
    """``detect`` outputs must not depend on what sits in the padded text slots."""
    torch.manual_seed(1)
    backbone = torch.randn(1, config.vision_config.backbone_config.hidden_size, 1, _GRID * _GRID)
    text = torch.randn(1, _C, 1, _SEQ)
    noisy = text.clone()
    noisy[..., _NUM_VALID:] = torch.randn_like(noisy[..., _NUM_VALID:]) * 10
    mask = _mask().float().reshape(1, 1, 1, _SEQ)

    for a, b in zip(detector(backbone, text, mask), detector(backbone, noisy, mask), strict=True):
        _assert_close(a, b)

    # Guard against a mask that does nothing: with every slot marked real, the noise shows.
    ones = torch.ones_like(mask)
    logits_clean = detector(backbone, text, ones)[2]
    logits_noisy = detector(backbone, noisy, ones)[2]
    assert (logits_clean - logits_noisy).abs().max() > 1e-2


@torch.no_grad()
def test_detector_masks_padding_in_fp16(detector, config):
    """``detect`` ships fp16: the ``-40000`` bias must stay finite and still zero out padding."""
    half = copy.deepcopy(detector).half()
    torch.manual_seed(1)
    backbone = torch.randn(1, config.vision_config.backbone_config.hidden_size, 1, _GRID * _GRID)
    text = torch.randn(1, _C, 1, _SEQ)
    noisy = text.clone()
    noisy[..., _NUM_VALID:] = torch.randn_like(noisy[..., _NUM_VALID:]) * 10
    mask = _mask().float().reshape(1, 1, 1, _SEQ)

    clean_out = half(backbone.half(), text.half(), mask.half())
    noisy_out = half(backbone.half(), noisy.half(), mask.half())
    for a, b in zip(clean_out, noisy_out, strict=True):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, atol=0, rtol=0)
