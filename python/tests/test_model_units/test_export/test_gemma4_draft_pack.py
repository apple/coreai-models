# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""The Gemma 4 CPU draft-head pack (``models/gemma4/export_draft_pack.py``).

The recipe is not part of the installed package, so it is loaded by path, after ``export.py`` and
``export_drafter.py``, which it imports. Covered: the pack's tensors and metadata, the permuted
output table, the int8 attention projections, and the checkpoint checks it shares with the drafter
export.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from tests._runner_infra._deps import _HAS_COREAI, _MSG_COREAI_NOT_FOUND  # noqa: E402

pytestmark = pytest.mark.skipif(not _HAS_COREAI, reason=_MSG_COREAI_NOT_FOUND)

_GEMMA4 = Path(__file__).resolve().parents[4] / "models/gemma4"


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _GEMMA4 / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if _HAS_COREAI:
    _load("export")
    _load("export_drafter")
    export_draft_pack = _load("export_draft_pack")
    pack = export_draft_pack.pack
    int8_rows = export_draft_pack.int8_rows

VOCAB, HIDDEN, CENTROIDS, HEADS, HEAD_DIM, FFN = 8, 8, 2, 2, 4, 16


def checkpoint(layers=2, untied=False, vocab=VOCAB):
    g = torch.Generator().manual_seed(0)

    def r(*shape):
        return torch.randn(*shape, generator=g)

    t = {
        "model.embed_tokens.weight": r(vocab, HIDDEN),
        "masked_embedding.token_ordering": torch.randperm(vocab, generator=g),
        "masked_embedding.centroids.weight": r(CENTROIDS, HIDDEN),
        "pre_projection.weight": r(HIDDEN, 2 * HIDDEN),
        "post_projection.weight": r(HIDDEN, HIDDEN),
        "model.norm.weight": r(HIDDEN),
    }
    if untied:
        t["lm_head.weight"] = r(vocab, HIDDEN)
    for i in range(layers):
        p = f"model.layers.{i}."
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            t[p + name + ".weight"] = r(HIDDEN)
        t[p + "self_attn.q_norm.weight"] = r(HEAD_DIM)
        t[p + "layer_scalar"] = r(1)
        t[p + "self_attn.q_proj.weight"] = r(HEADS * HEAD_DIM, HIDDEN)
        t[p + "self_attn.o_proj.weight"] = r(HIDDEN, HEADS * HEAD_DIM)
        t[p + "mlp.gate_proj.weight"] = r(FFN, HIDDEN)
        t[p + "mlp.up_proj.weight"] = r(FFN, HIDDEN)
        t[p + "mlp.down_proj.weight"] = r(HIDDEN, FFN)
    return t


def write_checkpoint(tmp_path, t, layers=2):
    drafter = tmp_path / "assistant"
    drafter.mkdir()
    save_file(t, str(drafter / "model.safetensors"))
    text = {
        "num_hidden_layers": layers,
        "hidden_size": HIDDEN,
        "num_attention_heads": HEADS,
        "head_dim": HEAD_DIM,
        "global_head_dim": HEAD_DIM,
        "intermediate_size": FFN,
        "layer_types": ["sliding_attention"] * (layers - 1) + ["full_attention"],
        "sliding_window": 512,
        "rope_parameters": {
            "sliding_attention": {"rope_theta": 10000.0},
            "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.25},
        },
    }
    (drafter / "config.json").write_text(json.dumps({"text_config": text}))
    return drafter


def read(path):
    """The pack's tensors and its ``__metadata__``."""
    with safe_open(str(path), framework="pt") as f:
        return {k: f.get_tensor(k) for k in f.keys()}, f.metadata()  # noqa: SIM118


def test_pack_metadata_describes_the_geometry(tmp_path):
    t = checkpoint()
    path = pack(write_checkpoint(tmp_path, t), tmp_path / "out")
    tensors, meta = read(path)
    assert path.name == "draft_head.safetensors"
    assert meta["format"] == "gemma4_draft_head" and meta["version"] == "1"
    assert (meta["vocab"], meta["centroids"], meta["per_centroid"]) == ("8", "2", "4")
    assert (meta["layers"], meta["hidden"], meta["heads"]) == ("2", str(HIDDEN), str(HEADS))
    assert (meta["head_dim"], meta["full_head_dim"], meta["ffn"]) == (
        str(HEAD_DIM),
        str(HEAD_DIM),
        str(FFN),
    )
    assert meta["sliding_window"] == "512"
    assert float(meta["sliding_theta"]) == 10000.0 and float(meta["full_theta"]) == 1000000.0
    assert float(meta["full_partial_rotary_factor"]) == 0.25
    for name in ("pre_projection.weight", "model.layers.1.mlp.down_proj.weight"):
        assert tensors[name].dtype == torch.float16
        assert torch.equal(tensors[name], t[name].to(torch.float16))


def test_pack_permutes_the_untied_head(tmp_path):
    t = checkpoint(untied=True)
    tensors, _ = read(pack(write_checkpoint(tmp_path, t), tmp_path / "out"))
    expected = t["lm_head.weight"][t["masked_embedding.token_ordering"]].to(torch.float16)
    assert torch.equal(tensors["embed_perm_f16"], expected)
    order = tensors["token_ordering_i32"]
    assert torch.equal(order, t["masked_embedding.token_ordering"].to(torch.int32))


def test_int8_attention_quantizes_only_the_attention_projections(tmp_path):
    t = checkpoint()
    tensors, _ = read(pack(write_checkpoint(tmp_path, t), tmp_path / "out", int8_attention=True))
    for i in range(2):
        p = f"model.layers.{i}."
        for name in ("self_attn.q_proj.weight", "self_attn.o_proj.weight"):
            q, scale = tensors[p + name], tensors[p + name + ".scale"]
            assert q.dtype == torch.int8 and scale.shape == (q.shape[0],)
            restored = q.float() * scale.float()[:, None]
            assert torch.allclose(restored, t[p + name], atol=float(scale.max()) * 0.51)
        for name in ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"):
            assert tensors[p + name].dtype == torch.float16
            assert p + name + ".scale" not in tensors


def test_int8_rows_scale_each_row_to_its_own_largest_magnitude():
    w = torch.tensor([[1.0, -2.0, 0.5], [0.0, 0.0, 0.0], [127.0, 0.0, -127.0]])
    q, scale = int8_rows(w)
    assert q.dtype == torch.int8
    assert q[0].tolist() == [64, -127, 32] and q[1].tolist() == [0, 0, 0]
    assert q[2].tolist() == [127, 0, -127]
    assert torch.allclose(scale.float(), torch.tensor([2 / 127, 1.0, 1.0]), rtol=1e-3)


def test_rejects_vocab_not_divisible_by_centroids(tmp_path):
    t = checkpoint(vocab=9)
    with pytest.raises(SystemExit, match="not divisible"):
        pack(write_checkpoint(tmp_path, t), tmp_path / "out")


def test_rejects_a_token_ordering_that_is_not_a_permutation(tmp_path):
    t = checkpoint()
    t["masked_embedding.token_ordering"] = torch.zeros(VOCAB, dtype=torch.int64)
    with pytest.raises(SystemExit, match="not a permutation"):
        pack(write_checkpoint(tmp_path, t), tmp_path / "out")


def test_names_a_missing_layer_tensor(tmp_path):
    t = checkpoint()
    del t["model.layers.1.mlp.up_proj.weight"]
    with pytest.raises(SystemExit, match="model.layers.1.mlp.up_proj.weight"):
        pack(write_checkpoint(tmp_path, t), tmp_path / "out")
