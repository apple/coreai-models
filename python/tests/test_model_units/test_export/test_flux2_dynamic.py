# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the FLUX.2 dynamic-shape export (``--dynamic``).

CPU-only, toy modules. Nothing here downloads weights; the conversion tests run the Core AI
converter on tiny randomly-initialized models.
"""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from coreai_models.converter import export_stateless
from coreai_models.diffusion import export
from coreai_models.diffusion.components import (
    _FLUX2_TRANSFORMER_INPUT_NAMES,
    FLUX2_DYNAMIC_COMPONENTS,
    FLUX2_MULTIFUNCTION_COMPONENTS,
    get_component_registry,
)
from coreai_models.diffusion.flux2 import (
    FLUX2_IMG_SEQ_MAX,
    FLUX2_IMG_SEQ_MIN,
    FLUX2_MIN_IMAGE_SIZE,
    FLUX2_MIN_REFERENCE_SIZE,
    Flux2TransformerWrapper,
    Flux2VAEDecoderWrapper,
    Flux2VAEEncoderWrapper,
    _dummy_flux2_transformer_impl,
    _patch_attn_split_for_dynamic,
    flux2_transformer_dynamic_shapes,
    flux2_vae_decoder_dynamic_shapes,
    flux2_vae_encoder_dynamic_shapes,
)
from coreai_models.diffusion.pipeline import _build_flux2_config


def _tiny_transformer() -> nn.Module:
    from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel

    torch.manual_seed(0)
    return Flux2Transformer2DModel(
        in_channels=8,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=32,
        timestep_guidance_channels=16,
        axes_dims_rope=(4, 4, 4, 4),
        guidance_embeds=False,
    ).eval()


def _tiny_vae() -> nn.Module:
    from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2

    torch.manual_seed(0)
    return AutoencoderKLFlux2(
        block_out_channels=(8, 8, 8, 8), layers_per_block=1, latent_channels=4, norm_num_groups=4
    ).eval()


def test_dynamic_transformer_converts() -> None:
    """The symbolic image sequence needs the slice split to get through the converter."""
    model = _tiny_transformer().half()
    side = math.isqrt(FLUX2_IMG_SEQ_MIN)  # trace at the Dim's lower bound
    inputs = _dummy_flux2_transformer_impl(SimpleNamespace(transformer=model), grid_size=side)
    export_stateless(
        Flux2TransformerWrapper(model, dynamic=True),
        inputs,
        _FLUX2_TRANSFORMER_INPUT_NAMES,
        ("output",),
        dynamic_shapes=flux2_transformer_dynamic_shapes(),
    )


def test_dynamic_vae_encoder_converts() -> None:
    """A free size Dim fails torch.export's divisibility guards; the spec must not."""
    image = torch.randn(1, 3, FLUX2_MIN_REFERENCE_SIZE, FLUX2_MIN_REFERENCE_SIZE).half()
    export_stateless(
        Flux2VAEEncoderWrapper(_tiny_vae().half()),
        (image,),
        ("image",),
        ("latent_params",),
        dynamic_shapes=flux2_vae_encoder_dynamic_shapes(),
    )


def test_dynamic_vae_decoder_converts() -> None:
    side = FLUX2_MIN_IMAGE_SIZE // 8
    z = torch.randn(1, 4, side, side).half()
    export_stateless(
        Flux2VAEDecoderWrapper(_tiny_vae().half(), dynamic=True),
        (z,),
        ("z",),
        ("image",),
        dynamic_shapes=flux2_vae_decoder_dynamic_shapes(),
    )


def test_dynamic_vae_decoder_keeps_stock_upsample() -> None:
    """The static decoder's ``repeat_interleave`` upsample patch fails ANE region formation
    under symbolic sizes. That only shows on device and still converts, so check the patch
    directly: a dynamic decoder must leave ``Upsample2D.forward`` alone."""
    from diffusers.models.upsampling import Upsample2D

    def patched_upsamplers(dynamic: bool) -> list[bool]:
        # The patch mutates the decoder in place, so each wrapper gets a fresh VAE.
        wrapper = Flux2VAEDecoderWrapper(_tiny_vae(), dynamic=dynamic)
        return [
            "forward" in m.__dict__
            for m in wrapper.vae.decoder.modules()
            if isinstance(m, Upsample2D)
        ]

    static = patched_upsamplers(dynamic=False)
    assert static and all(static)  # the patch applies, so the check below can see it
    assert not any(patched_upsamplers(dynamic=True))


def test_dynamic_registry_takes_precedence_over_multifunction() -> None:
    assert get_component_registry(None, "flux2", multifunction=True, dynamic=True) is (
        FLUX2_DYNAMIC_COMPONENTS
    )
    assert get_component_registry(None, "flux2", multifunction=True) is (
        FLUX2_MULTIFUNCTION_COMPONENTS
    )


def _run_export_cli(
    monkeypatch: pytest.MonkeyPatch, *args: str, model: str = "flux2-klein-4b"
) -> None:
    monkeypatch.setattr("sys.argv", ["coreai.diffusion.export", model, *args])
    export.main()


def test_dynamic_rejects_ios(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        _run_export_cli(monkeypatch, "--dynamic", "--platform", "iOS", "--dry-run")
    assert "--dynamic is not supported with --platform iOS" in capsys.readouterr().err


def test_dynamic_explains_wan_is_already_dynamic(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        _run_export_cli(
            monkeypatch, "--dynamic", "--dry-run", model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        )
    assert "always exported with a dynamic frame count" in capsys.readouterr().err


def test_dynamic_allows_macos(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _run_export_cli(monkeypatch, "--dynamic", "--platform", "macOS", "--dry-run")
    assert "dynamic:           True" in capsys.readouterr().out


def _flux2_pipe_stub() -> SimpleNamespace:
    return SimpleNamespace(
        vae=SimpleNamespace(config=SimpleNamespace(block_out_channels=[1, 2, 3, 4])),
        transformer=SimpleNamespace(config=SimpleNamespace()),
    )


def test_dynamic_metadata_carries_runtime_bounds() -> None:
    config = _build_flux2_config(_flux2_pipe_stub(), "m", dynamic=True)
    assert config["dynamic_shapes"] is True
    assert config["max_image_tokens"] == FLUX2_IMG_SEQ_MAX
    for key in ("min_image_size", "max_image_size", "image_size_multiple", "min_reference_size"):
        assert isinstance(config[key], int)


def test_static_metadata_has_no_dynamic_keys() -> None:
    config = _build_flux2_config(_flux2_pipe_stub(), "m")
    assert "dynamic_shapes" not in config
    assert "max_image_tokens" not in config


def test_slice_split_processor_matches_stock() -> None:
    """The dynamic patch only swaps ``split_with_sizes`` for slices; output must not move."""
    stock = _tiny_transformer()
    patched = copy.deepcopy(stock)
    assert _patch_attn_split_for_dynamic(patched) == 1  # the one double-stream block
    assert _patch_attn_split_for_dynamic(patched) == 0  # idempotent

    img_seq, txt_seq = 12, 5
    img_ids = torch.zeros(1, img_seq, 4)
    img_ids[0, :, 1] = torch.arange(img_seq, dtype=torch.float32)
    txt_ids = torch.zeros(1, txt_seq, 4)
    txt_ids[0, :, 3] = torch.arange(txt_seq, dtype=torch.float32)
    inputs = {
        "hidden_states": torch.randn(1, img_seq, 8),
        "encoder_hidden_states": torch.randn(1, txt_seq, 32),
        "timestep": torch.tensor([0.5]),
        "img_ids": img_ids,
        "txt_ids": txt_ids,
    }
    with torch.no_grad():
        expected = stock(**inputs).sample
        actual = patched(**inputs).sample
    assert torch.equal(actual, expected)
