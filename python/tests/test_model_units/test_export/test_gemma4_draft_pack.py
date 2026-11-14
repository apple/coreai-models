# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""The Gemma 4 CPU draft-head pack (``models/gemma4/export_draft_pack.py``).

The recipe is not part of the installed package, so it is loaded by path, after ``export.py`` and
``export_drafter.py``, which it imports. Covered: the pack's tensors and metadata, the permuted
output table, and the checkpoint checks it shares with the drafter
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
    export_drafter = _load("export_drafter")
    export_draft_pack = _load("export_draft_pack")
    pack = export_draft_pack.pack
    default_output_dir = export_draft_pack.default_output_dir

VOCAB, HIDDEN, CENTROIDS, HEADS, HEAD_DIM, FFN = 8, 8, 2, 2, 4, 16
# The full-attention layer is wider than the sliding ones, as in Gemma 4.
FULL_HEAD_DIM = 2 * HEAD_DIM


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
        dim = FULL_HEAD_DIM if i == layers - 1 else HEAD_DIM
        t[p + "self_attn.q_norm.weight"] = r(dim)
        t[p + "layer_scalar"] = r(1)
        t[p + "self_attn.q_proj.weight"] = r(HEADS * dim, HIDDEN)
        t[p + "self_attn.o_proj.weight"] = r(HIDDEN, HEADS * dim)
        t[p + "mlp.gate_proj.weight"] = r(FFN, HIDDEN)
        t[p + "mlp.up_proj.weight"] = r(FFN, HIDDEN)
        t[p + "mlp.down_proj.weight"] = r(HIDDEN, FFN)
    return t


def write_checkpoint(tmp_path, t, layers=2, global_head_dim=FULL_HEAD_DIM):
    drafter = tmp_path / "assistant"
    drafter.mkdir()
    save_file(t, str(drafter / "model.safetensors"))
    text = {
        "num_hidden_layers": layers,
        "hidden_size": HIDDEN,
        "num_attention_heads": HEADS,
        "head_dim": HEAD_DIM,
        "intermediate_size": FFN,
        "layer_types": ["sliding_attention"] * (layers - 1) + ["full_attention"],
        "sliding_window": 512,
        "rope_parameters": {
            "sliding_attention": {"rope_theta": 10000.0},
            "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.25},
        },
    }
    if global_head_dim is not None:
        text["global_head_dim"] = global_head_dim
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
        str(FULL_HEAD_DIM),
        str(FFN),
    )
    assert meta["sliding_window"] == "512"
    assert float(meta["sliding_theta"]) == 10000.0 and float(meta["full_theta"]) == 1000000.0
    assert float(meta["full_partial_rotary_factor"]) == 0.25


def test_pack_holds_exactly_the_heads_tensors_in_fp16(tmp_path):
    t = checkpoint()
    tensors, _ = read(pack(write_checkpoint(tmp_path, t), tmp_path / "out"))
    layer_names = [f"model.layers.{i}.{s}" for i in range(2) for s in export_drafter.LAYER_TENSORS]
    copied = ["pre_projection.weight", "post_projection.weight", "model.norm.weight", *layer_names]
    assert set(tensors) == {"embed_perm_f16", "token_ordering_i32", "centroids_f16", *copied}
    for name in copied:
        assert tensors[name].dtype == torch.float16, name
        assert torch.equal(tensors[name], t[name].to(torch.float16)), name
    centroids = t["masked_embedding.centroids.weight"].to(torch.float16)
    assert torch.equal(tensors["centroids_f16"], centroids)


def test_full_head_dim_falls_back_to_head_dim(tmp_path):
    t = checkpoint()
    path = pack(write_checkpoint(tmp_path, t, global_head_dim=None), tmp_path / "out")
    assert read(path)[1]["full_head_dim"] == str(HEAD_DIM)


def test_refuses_an_assistant_for_another_target(tmp_path):
    drafter = write_checkpoint(tmp_path, checkpoint())
    with pytest.raises(SystemExit, match="different target"):
        pack(drafter, tmp_path / "out", backbone_hidden_size=HIDDEN + 1)
    assert pack(drafter, tmp_path / "ok", backbone_hidden_size=HIDDEN).exists()


def test_default_output_dir_is_per_variant(tmp_path):
    e2b = default_output_dir("google/gemma-4-E2B-it", tmp_path)
    e4b = default_output_dir("google/gemma-4-E4B-it", tmp_path)
    assert e2b.name == "gemma-4-E2B-it" and e4b.name == "gemma-4-E4B-it" and e2b != e4b
    assert default_output_dir(None, tmp_path / "my-assistant").name == "my-assistant"


def test_scan_table_shares_the_permutation_check(tmp_path):
    t = checkpoint()
    t["masked_embedding.token_ordering"] = torch.zeros(VOCAB, dtype=torch.int64)
    with pytest.raises(SystemExit, match="not a permutation"):
        export_drafter.write_scan_table(t, 2, tmp_path / "draft_scan.bin")


def test_pack_permutes_the_untied_head(tmp_path):
    t = checkpoint(untied=True)
    tensors, _ = read(pack(write_checkpoint(tmp_path, t), tmp_path / "out"))
    expected = t["lm_head.weight"][t["masked_embedding.token_ordering"]].to(torch.float16)
    assert torch.equal(tensors["embed_perm_f16"], expected)
    order = tensors["token_ordering_i32"]
    assert torch.equal(order, t["masked_embedding.token_ordering"].to(torch.int32))


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
