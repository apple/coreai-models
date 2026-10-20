# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Neural Engine EG2 image encoder (``models/ios/embedding_gemma2_vision.py``) vs macOS."""

from pathlib import Path

import torch
import torch.nn.functional as F

import coreai_models.models.ios.embedding_gemma2_vision as ios_vision
from coreai_models.models.macos.embedding_gemma2_vision import (
    EmbeddingGemma2ImageEncoder,
    pad_to_budget,
    patch_position_ids,
)
from tests.test_model_units.test_models._embedding_gemma2_common import TINY_VISION_CONFIG as TINY


def _pair():
    torch.manual_seed(0)
    ref = EmbeddingGemma2ImageEncoder(TINY)
    with torch.no_grad():
        for name, p in ref.named_parameters():
            if name.endswith("norm.weight") or "layernorm" in name:
                p.copy_(1.0 + 0.5 * torch.rand_like(p))
            else:
                p.normal_(0.0, 0.3)
    sd = {
        (k if k.startswith("embed_vision.") else "vision_tower." + k): v
        for k, v in ref.state_dict().items()
    }
    ane = ios_vision.EmbeddingGemma2ImageEncoder(TINY)
    for layer in ane.layers:
        layer.self_attn.q_chunk = 25  # several ragged query chunks at P = 108
    ios_vision.load_hf_state_dict(ane, sd)
    return ref.eval(), ane.eval()


def _inputs(grids):
    pvs, poss, ns = [], [], []
    for gh, gw in grids:
        n = gh * gw
        pv = torch.rand(n, TINY.patch_dim)
        pv, pos = pad_to_budget(pv, patch_position_ids(gh, gw), TINY.max_patches)
        pvs.append(pv[None])
        poss.append(pos[None])
        ns.append(n // TINY.pooling_kernel_size**2)
    return pvs, poss, ns


def test_one_hot_is_exact_including_padding() -> None:
    grid = torch.arange(840, dtype=torch.float16)
    coord = torch.tensor([[-1.0], [0.0], [3.0], [839.0]], dtype=torch.float16)
    oh = ios_vision.one_hot_f16(coord, grid)
    assert oh[0].sum() == 0
    for row, c in zip(oh[1:], (0, 3, 839), strict=True):
        assert row.sum() == 1 and row[c] == 1


def test_matches_macos_reference_fp32() -> None:
    ref, ane = _pair()
    pvs, poss, ns = _inputs([(6, 9), (12, 9), (3, 18), (3, 3)])
    for pv, pos, n in zip(pvs, poss, ns, strict=True):
        with torch.no_grad():
            want = ref(pv, pos)
            got = ane(pv, pos.float())
        assert got.shape == want.shape == (1, TINY.max_soft_tokens, TINY.text_hidden_size)
        cos = F.cosine_similarity(got[0, :n].double(), want[0, :n].double(), -1)
        assert cos.min() > 1 - 1e-6, (n, cos.min())
        torch.testing.assert_close(got, want, atol=2e-4, rtol=2e-4)
        assert (got[0, n:] == 0).all()  # padding slots are exact zeros


def test_fp16_stays_finite_with_large_activations() -> None:
    _, ane = _pair()
    ane = ane.half()
    pvs, poss, _ = _inputs([(6, 9)])
    with torch.no_grad():
        # Scale the residual stream far past sqrt(fp16 max) = 255.9 via the input projection.
        ane.input_proj.weight.mul_(400.0)
        out = ane(pvs[0].half(), poss[0].half())
    assert torch.isfinite(out).all() and out.abs().max() > 0


def test_torch_export_static() -> None:
    _, ane = _pair()
    pvs, poss, _ = _inputs([(6, 9)])
    ep = torch.export.export(ane, args=(pvs[0], poss[0].float()))
    torch.testing.assert_close(ep.module()(pvs[0], poss[0].float()), ane(pvs[0], poss[0].float()))


def test_real_snapshot_loads_and_pools_finite(embeddinggemma2_snapshot: Path) -> None:
    budget = 12
    ane = ios_vision.load_image_encoder(
        str(embeddinggemma2_snapshot), dtype=torch.float16, max_soft_tokens=budget
    )
    gh, gw = 6, 9
    n_soft = gh * gw // 9
    pv = torch.rand(gh * gw, ane.config.patch_dim)
    pv, pos = pad_to_budget(pv, patch_position_ids(gh, gw), budget * 9)
    with torch.no_grad():
        out = ane(pv[None].half(), pos[None].half())
    assert out.shape == (1, budget, 512)
    assert torch.isfinite(out).all()
    assert (out[0, n_soft:] == 0).all()
