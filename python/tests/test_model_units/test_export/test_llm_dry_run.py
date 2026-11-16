# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""``coreai.llm.export --dry-run`` prints every parameter that shapes the export.

Config resolution only: nothing here downloads weights or runs an export.
"""

from __future__ import annotations

import pytest

from coreai_models.llm import export as llm_export


def _dry_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], *args: str
) -> str:
    monkeypatch.setattr("sys.argv", ["coreai.llm.export", "qwen3-0.6b", "--dry-run", *args])
    llm_export.main()
    return capsys.readouterr().out


def test_dry_run_lists_batched_parameters(monkeypatch, capsys):
    out = _dry_run(monkeypatch, capsys, "--attention-mask", "batched", "--dynamic-batch-size", "4")
    assert "attention_mask:     batched" in out
    assert "dynamic_batch_size: 4" in out


def test_dry_run_shows_the_default_batch_size_for_batched_exports(monkeypatch, capsys):
    out = _dry_run(monkeypatch, capsys, "--attention-mask", "batched")
    assert "attention_mask:     batched" in out
    assert "dynamic_batch_size: 8" in out


def test_dry_run_omits_batched_parameters_by_default(monkeypatch, capsys):
    out = _dry_run(monkeypatch, capsys)
    assert "attention_mask" not in out
    assert "dynamic_batch_size" not in out
