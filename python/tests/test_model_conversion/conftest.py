# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Test fixtures specific to ``test_model_conversion/``.

The autouse ``use_hf_impl`` fixture flips ``USE_HF_IMPL=true`` for every
test in this directory so ``coreai_models.primitives.macos.{sdpa,rope}``
take the Hugging Face lowering path -- the only path that gives bit-for-bit
parity with HF eager. ``disable_hf_impl_for_coreai`` is the per-test
opt-out for Core AI-export tests where the HF impl decomposes into where-ops
the Core AI runtime can't lower.

Also applies ``pytest.mark.flaky(reruns=5)`` to every test in this tree
and skips all tests when the HuggingFace Hub is unreachable (every test
in this directory calls ``from_pretrained``).
"""

import os
from collections.abc import Iterator

import pytest

from tests._runner_infra._deps import _hf_hub_reachable


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Apply flaky marker to all tests in this directory and subdirectories."""
    for item in items:
        item.add_marker(pytest.mark.flaky(reruns=5))


@pytest.fixture(autouse=True, scope="session")
def _skip_conversion_tests_if_offline() -> None:
    """Skip all model conversion tests when HF Hub is unreachable."""
    if not _hf_hub_reachable():
        pytest.skip("HuggingFace Hub unreachable; skipping model conversion tests")


@pytest.fixture(autouse=True, scope="module")
def use_hf_impl() -> Iterator[None]:
    """Use HuggingFace implementation for comparison tests in this module."""
    original = os.environ.get("USE_HF_IMPL")
    os.environ["USE_HF_IMPL"] = "true"
    yield
    if original is None:
        os.environ.pop("USE_HF_IMPL", None)
    else:
        os.environ["USE_HF_IMPL"] = original


@pytest.fixture
def disable_hf_impl_for_coreai() -> Iterator[None]:
    """Use vanilla SDPA (not HF impl) for Core AI tests.

    The HF impl decomposes ``F.scaled_dot_product_attention`` into where ops
    with dynamic-shaped i1 that the Core AI runtime does not support.
    """
    original = os.environ.get("USE_HF_IMPL")
    os.environ["USE_HF_IMPL"] = "false"
    yield
    if original is None:
        os.environ.pop("USE_HF_IMPL", None)
    else:
        os.environ["USE_HF_IMPL"] = original
