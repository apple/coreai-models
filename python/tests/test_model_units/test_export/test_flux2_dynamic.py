# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the FLUX.2 dynamic-shape export (``--dynamic``) and multi-reference ids.

CPU-only, toy modules. Nothing here downloads weights or runs an export.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from coreai_models.diffusion import export
from coreai_models.diffusion.components import (
    FLUX2_DYNAMIC_COMPONENTS,
    FLUX2_MULTIFUNCTION_COMPONENTS,
    get_component_registry,
)
from coreai_models.diffusion.flux2 import (
    FLUX2_IMG_SEQ_MAX,
    REFERENCE_TOKEN_TIME_OFFSET,
    _dummy_flux2_transformer_img2img,
    _patch_attn_split_for_dynamic,
)
from coreai_models.diffusion.pipeline import _build_flux2_config


class _ConfiguredModule(nn.Module):
    """Stand-in transformer: the img2img dummy reads only ``config`` and a parameter dtype."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(
            axes_dims_rope=[32, 32, 32, 32], in_channels=128, joint_attention_dim=7680
        )


def test_each_reference_gets_its_own_time_coordinate() -> None:
    """Reference i sits at T=10*(i+1), matching diffusers ``_prepare_image_ids``."""
    pipe = SimpleNamespace(transformer=_ConfiguredModule())
    hidden_states, *_, img_ids, _ = _dummy_flux2_transformer_img2img(
        pipe, noise_grid=4, ref_grids=(2, 3)
    )
    assert hidden_states.shape[1] == img_ids.shape[1] == 16 + 4 + 9

    times = img_ids[0, :, 0]
    assert torch.all(times[:16] == 0)
    assert torch.all(times[16:20] == REFERENCE_TOKEN_TIME_OFFSET)
    assert torch.all(times[20:] == 2 * REFERENCE_TOKEN_TIME_OFFSET)
    # H/W restart at 0 for each reference grid.
    assert img_ids[0, 20, 1:3].tolist() == [0, 0]
    assert img_ids[0, -1, 1:3].tolist() == [2, 2]


@pytest.mark.parametrize("name", list(FLUX2_DYNAMIC_COMPONENTS))
def test_dynamic_shapes_align_with_input_names(name: str) -> None:
    """``export_stateless`` binds the spec positionally, so a length mismatch misbinds."""
    spec = FLUX2_DYNAMIC_COMPONENTS[name]
    if getattr(spec, "dynamic_shapes_fn", None) is None:
        pytest.skip(f"{name} is static")
    assert len(spec.dynamic_shapes_fn()) == len(spec.input_names)


def test_transformer_sequence_bound_fits_three_full_references_at_1024() -> None:
    """The README promises three full-size references at 1024 output."""
    assert FLUX2_IMG_SEQ_MAX >= 4096 + 3 * 4096


def test_dynamic_registry_takes_precedence_over_multifunction() -> None:
    assert get_component_registry(None, "flux2", multifunction=True, dynamic=True) is (
        FLUX2_DYNAMIC_COMPONENTS
    )
    assert get_component_registry(None, "flux2", multifunction=True) is (
        FLUX2_MULTIFUNCTION_COMPONENTS
    )


def _run_export_cli(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    monkeypatch.setattr("sys.argv", ["coreai.diffusion.export", "flux2-klein-4b", *args])
    export.main()


def test_dynamic_rejects_ios(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        _run_export_cli(monkeypatch, "--dynamic", "--platform", "iOS", "--dry-run")
    assert "--dynamic is not supported with --platform iOS" in capsys.readouterr().err


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
    from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel

    torch.manual_seed(0)
    stock = Flux2Transformer2DModel(
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
