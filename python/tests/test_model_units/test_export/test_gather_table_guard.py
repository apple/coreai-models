# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Embedding tables a GPU gather reads incorrectly are flagged at export (#313)."""

import logging

import pytest
import torch

from coreai_models.export.pipeline import warn_on_oversized_gather_tables
from coreai_models.models.macos.gemma3n import ChunkedEmbedding

# Gemma 3n per-layer table shapes; meta tensors keep these free to build.
E4B_SHAPE = (262144, 35 * 256)  # 2.35e9 elements, over 2**31
E2B_SHAPE = (262144, 30 * 256)  # 2.01e9 elements, under 2**31


def _model(**modules: torch.nn.Module) -> torch.nn.Module:
    return torch.nn.ModuleDict(modules)


def test_warns_on_oversized_embedding(caplog: pytest.LogCaptureFixture) -> None:
    with torch.device("meta"):
        model = _model(per_layer=torch.nn.Embedding(*E4B_SHAPE))
    with caplog.at_level(logging.WARNING):
        warn_on_oversized_gather_tables(model)
    assert "'per_layer'" in caplog.text
    assert str(E4B_SHAPE[0] * E4B_SHAPE[1]) in caplog.text


def test_silent_below_limit(caplog: pytest.LogCaptureFixture) -> None:
    with torch.device("meta"):
        model = _model(per_layer=torch.nn.Embedding(*E2B_SHAPE))
    with caplog.at_level(logging.WARNING):
        warn_on_oversized_gather_tables(model)
    assert caplog.text == ""


def test_silent_for_chunked_embedding(caplog: pytest.LogCaptureFixture) -> None:
    """The fix itself must not trip the guard."""
    with torch.device("meta"):
        model = _model(per_layer=ChunkedEmbedding(*E4B_SHAPE))
    with caplog.at_level(logging.WARNING):
        warn_on_oversized_gather_tables(model)
    assert caplog.text == ""
