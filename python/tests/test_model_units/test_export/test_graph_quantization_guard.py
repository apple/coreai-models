# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Graph-mode quantization is refused for models whose prefill graph needs its own trace."""

from types import SimpleNamespace

import pytest

from coreai_models.export import pipeline as export_pipeline


def test_muse_glimmer_refuses_graph_quantization(monkeypatch):
    """A flattened Muse module would derive ``prefill`` from the write-first ``main`` graph."""
    hf_config = SimpleNamespace(model_type="muse_glimmer", text_config=SimpleNamespace())
    monkeypatch.setattr(export_pipeline.AutoConfig, "from_pretrained", lambda *_, **__: hf_config)
    config = export_pipeline.ExportConfig(
        hf_model_id="meta-models/Muse-Glimmer-30B", quantization_mode="graph"
    )
    with pytest.raises(ValueError, match="Graph-mode quantization is not currently supported"):
        export_pipeline.resolve_model(config)
