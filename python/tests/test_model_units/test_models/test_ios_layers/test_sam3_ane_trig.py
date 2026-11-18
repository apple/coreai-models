# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""The SAM3 DETR decoder's box sine encoding uses no ``sin``/``cos`` ops.

The M2 (H14) ANE can't run ``sin``/``cos``, so MPSGraph used to put decoder layers 1-4 on the
GPU, and that ANE -> GPU -> ANE ``detect`` aborted on M2 Ultra/Max. ``sin_0_2pi``/``cos_0_2pi``
are multiply/add polynomials. These tests check they track ``torch.sin``/``torch.cos`` on the
encoding's whole range, that ``encode_boxes`` still matches the original formula, and that its
exported graph holds no trig ops.
"""

import math

import pytest
import torch

from coreai_models.models.ios.sam3.detr import SinePositionEmbeddingReauthored
from coreai_models.primitives.ios.trig import cos_0_2pi, sin_0_2pi

# fp16 floor: the angle itself is rounded to fp16, and half an ulp of x near 2*pi is ~2e-3.
_TOL = {torch.float32: (1e-4, 2e-5), torch.float16: (4e-3, 6e-4)}  # (max, mean) abs error


def _angles(dtype: torch.dtype) -> torch.Tensor:
    # In fp16 the 200k-point grid hits every representable angle above ~3e-5 (tiny angles are
    # covered by the small-angle test). 6.28125 = fp16(2*pi), the encoding's largest angle.
    return torch.linspace(0, 2 * math.pi, 200_001, dtype=torch.float64).to(dtype).clamp(max=6.28125)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(("approx", "exact"), [(sin_0_2pi, torch.sin), (cos_0_2pi, torch.cos)])
def test_trig_matches_torch_on_0_2pi(approx, exact, dtype):
    x = _angles(dtype)
    err = (approx(x).double() - exact(x.double())).abs()
    max_tol, mean_tol = _TOL[dtype]
    assert err.max().item() <= max_tol
    assert err.mean().item() <= mean_tol


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_sin_keeps_relative_precision_for_small_angles(dtype):
    """Most encoding channels divide by a large ``dim_t``; their tiny angles must survive."""
    x = torch.logspace(-6, -1, 2001, dtype=torch.float64).to(dtype)
    rel = (sin_0_2pi(x).double() / x.double().sin() - 1).abs()
    assert rel.max().item() <= (1e-4 if dtype == torch.float32 else 3e-3)


def _encode_boxes_reference(boxes: torch.Tensor, num_pos_feats: int) -> torch.Tensor:
    """The pre-change ``encode_boxes`` (``torch.sin``/``torch.cos``), same fp16 constants."""
    scale = torch.tensor(2.0 * math.pi, dtype=torch.float16).to(boxes.dtype)
    temperature = torch.tensor(10000.0, dtype=torch.float16).to(boxes.dtype)
    dim_t = torch.arange(num_pos_feats).to(boxes.dtype)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)
    parts = []
    for i in (1, 0, 2, 3):
        pos = boxes[:, :, i, None] * scale / dim_t
        parts.append(torch.stack((pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3).flatten(2))
    return torch.cat(parts, dim=2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_encode_boxes_matches_sin_cos_formula(dtype):
    torch.manual_seed(0)
    boxes = torch.rand(2, 200, 4)
    boxes[0, :2] = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])  # range ends
    boxes = boxes.to(dtype)
    encoding = SinePositionEmbeddingReauthored(num_pos_feats=128)

    actual = encoding.encode_boxes(boxes)
    expected = _encode_boxes_reference(boxes, 128)

    assert actual.shape == expected.shape == (2, 200, 512)
    assert actual.dtype == dtype
    err = (actual.double() - expected.double()).abs()
    max_tol, mean_tol = _TOL[dtype]
    assert err.max().item() <= max_tol
    assert err.mean().item() <= mean_tol


class _EncodeBoxes(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoding = SinePositionEmbeddingReauthored(num_pos_feats=128)

    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        return self.encoding.encode_boxes(boxes)


def test_encode_boxes_graph_has_no_trig_or_negation():
    """``sin``, ``cos`` and unary ``neg`` don't appear in H14 ANE regions; keep them out."""
    exported = torch.export.export(_EncodeBoxes().half(), (torch.rand(1, 200, 4).half(),))
    targets = {str(node.target) for node in exported.graph.nodes if node.op == "call_function"}
    banned = {
        t
        for t in targets
        if t.split(".")[:2] in (["aten", "sin"], ["aten", "cos"], ["aten", "neg"])
    }
    assert not banned, banned
