# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""The Gemma 4 draft-head export (``models/gemma4/export_drafter.py``).

The recipe is not part of the installed package, so it is loaded by path, after ``export.py``,
which it imports as ``export``. Covered: the draft step's grouped-query attention and scaled
RMSNorm, the scan table, the checkpoint checks, and declaring the draft head in a target bundle.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("safetensors")

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
    export_g4 = _load("export")
    export_drafter = _load("export_drafter")
    _ios_metadata_extras = export_g4._ios_metadata_extras
    DRAFTER_AUXILIARY_KEYS = export_drafter.DRAFTER_AUXILIARY_KEYS
    DRAFTER_KIND = export_drafter.DRAFTER_KIND
    SCAN_NAME = export_drafter.SCAN_NAME
    DecoderLayer = export_drafter.DecoderLayer
    DraftStep = export_drafter.DraftStep
    RMSNorm = export_drafter.RMSNorm
    add_to_bundle = export_drafter.add_to_bundle
    check_checkpoint = export_drafter.check_checkpoint
    default_drafter = export_drafter.default_drafter
    example_inputs = export_drafter.example_inputs
    full_windows = export_drafter.full_windows
    load_drafter = export_drafter.load_drafter
    output_table = export_drafter.output_table
    resolve_drafter = export_drafter.resolve_drafter
    rotate_half = export_drafter.rotate_half
    write_scan_table = export_drafter.write_scan_table


# ---- Draft step ----


def layer_weights(hidden, heads, kv_heads, head_dim, ffn, seed):
    g = torch.Generator().manual_seed(seed)

    def r(*shape):
        return torch.randn(*shape, generator=g) * 0.05

    p = "model.layers.0."
    return {
        p + "self_attn.q_proj.weight": r(heads * head_dim, hidden),
        p + "self_attn.o_proj.weight": r(hidden, heads * head_dim),
        p + "self_attn.q_norm.weight": r(head_dim),
        p + "mlp.gate_proj.weight": r(ffn, hidden),
        p + "mlp.up_proj.weight": r(ffn, hidden),
        p + "mlp.down_proj.weight": r(hidden, ffn),
        p + "input_layernorm.weight": r(hidden),
        p + "post_attention_layernorm.weight": r(hidden),
        p + "pre_feedforward_layernorm.weight": r(hidden),
        p + "post_feedforward_layernorm.weight": r(hidden),
        p + "layer_scalar": torch.ones(1),
    }


def copied_kv_attention(layer, x, cos, sin, k, v, mask):
    """The attention before grouping: K and V repeated to every query head."""
    b, s, _ = x.shape
    q = layer.q_proj(layer.input_layernorm(x)).view(b, s, layer.num_heads, layer.head_dim)
    q = layer.q_norm(q).transpose(1, 2)
    q = q * cos + rotate_half(q) * sin
    groups = layer.num_heads // layer.num_kv_heads
    k, v = k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)
    attn = torch.matmul(v, torch.softmax(torch.matmul(q, k) + mask, dim=-1).transpose(-1, -2))
    return attn.permute(0, 3, 1, 2).reshape(b, s, layer.num_heads * layer.head_dim)


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("queries", [1, 3])
def test_grouped_attention_matches_copied_kv(kv_heads, queries):
    hidden, heads, head_dim, window = 64, 8, 16, 12
    layer = DecoderLayer(
        layer_weights(hidden, heads, kv_heads, head_dim, 96, seed=kv_heads * 10 + queries),
        0,
        heads,
        kv_heads,
    ).eval()
    g = torch.Generator().manual_seed(7)
    x = torch.randn(1, queries, hidden, generator=g)
    angles = torch.randn(1, 1, queries, head_dim // 2, generator=g)
    cos, sin = torch.cat([angles.cos()] * 2, -1), torch.cat([angles.sin()] * 2, -1)
    k = torch.randn(1, kv_heads, head_dim, window, generator=g)
    v = torch.randn(1, kv_heads, head_dim, window, generator=g)
    mask = torch.zeros(1, 1, 1, window)
    mask[..., -3:] = -40000  # padding slots, as the drafter masks empty positions

    captured = {}
    # The hook reads o_proj's input (the attention output) and leaves the output unchanged.
    layer.o_proj.register_forward_hook(lambda _m, inputs, _o: captured.update(attn=inputs[0]))
    with torch.no_grad():
        layer(x, cos, sin, k, v, mask)
        expected = copied_kv_attention(layer, x, cos, sin, k, v, mask)
    torch.testing.assert_close(captured["attn"], expected, atol=1e-5, rtol=1e-5)


def test_full_windows_are_the_targets_context_buckets():
    # The default covers the model's full context with the target's own sparse ladder, not a
    # function for every doubling of the window.
    assert full_windows(export_g4.IOS_MAX_CONTEXT_LENGTH) == [1024, 8192, 32768, 131072]
    assert full_windows(export_g4.IOS_MAX_CONTEXT_LENGTH) == export_g4.context_ladder(131072)
    assert full_windows(4096) == export_g4.context_ladder(4096) == [1024, 4096]
    assert full_windows(512) == [512]


# ---- DraftStep ----


def _step_inputs(module, window=8, full_window=16, seed=0):
    """Random float32 inputs for one draft step; the masks leave every position visible."""
    g = torch.Generator().manual_seed(seed)
    hd, full_hd = module.layers[0].head_dim, module.layers[-1].head_dim
    backbone = module.pre_projection.in_features // 2

    def r(*shape):
        return torch.randn(*shape, generator=g)

    return {
        "inputs_embeds": r(1, 1, 2 * backbone),
        "cos_sliding": r(1, 1, 1, hd),
        "sin_sliding": r(1, 1, 1, hd),
        "cos_full": r(1, 1, 1, full_hd),
        "sin_full": r(1, 1, 1, full_hd),
        "k_sliding": r(1, 1, hd, window),
        "v_sliding": r(1, 1, hd, window),
        "mask_sliding": torch.zeros(1, 1, 1, window),
        "k_full": r(1, 1, full_hd, full_window),
        "v_full": r(1, 1, full_hd, full_window),
        "mask_full": torch.zeros(1, 1, 1, full_window),
    }


def _draft_step(layers):
    return DraftStep(checkpoint(layers=layers), num_kv_heads=1).eval()


def test_draft_step_outputs_are_consistent_with_its_layers():
    module = _draft_step(layers=2)
    with torch.no_grad():
        hidden, centroid_logits, x = module(**_step_inputs(module))
    assert hidden.shape == x.shape == (1, 1, 4)
    assert centroid_logits.shape == (1, 1, 2)  # one score per centroid
    torch.testing.assert_close(hidden, module.post_projection(x))
    torch.testing.assert_close(centroid_logits, module.centroids(x))


def test_draft_step_is_deterministic():
    module = _draft_step(layers=2)
    inputs = _step_inputs(module)
    with torch.no_grad():
        first, second = module(**inputs), module(**inputs)
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)


def _output(module, **overrides):
    inputs = _step_inputs(module) | overrides
    with torch.no_grad():
        return module(**inputs)[2]


def test_draft_step_runs_the_sliding_layers_then_the_full_attention_layer():
    module = _draft_step(layers=3)
    i = _step_inputs(module)
    sliding = (
        i["cos_sliding"],
        i["sin_sliding"],
        i["k_sliding"],
        i["v_sliding"],
        i["mask_sliding"],
    )
    full = (i["cos_full"], i["sin_full"], i["k_full"], i["v_full"], i["mask_full"])
    with torch.no_grad():
        x = module.pre_projection(i["inputs_embeds"])
        for layer in module.layers[:-1]:
            x = layer(x, *sliding)
        x = module.norm(module.layers[-1](x, *full))
        torch.testing.assert_close(_output(module), x)


def test_the_last_layer_attends_to_the_full_cache_and_the_others_to_the_sliding_cache():
    # With two layers the first is sliding and the last is full attention: both caches matter.
    two = _draft_step(layers=2)
    base = _output(two)
    assert not torch.allclose(
        base, _output(two, k_full=torch.randn_like(_step_inputs(two)["k_full"]))
    )
    assert not torch.allclose(
        base, _output(two, k_sliding=torch.randn_like(_step_inputs(two)["k_sliding"]))
    )
    # With one layer it is the full-attention layer, so the sliding cache is never read.
    one = _draft_step(layers=1)
    base = _output(one)
    torch.testing.assert_close(
        base, _output(one, k_sliding=torch.randn_like(_step_inputs(one)["k_sliding"]))
    )
    assert not torch.allclose(
        base, _output(one, v_full=torch.randn_like(_step_inputs(one)["v_full"]))
    )


@pytest.mark.parametrize("cache", ["sliding", "full"])
def test_masked_positions_do_not_change_the_output(cache):
    module = _draft_step(layers=2)
    inputs = _step_inputs(module)
    width = inputs[f"k_{cache}"].shape[-1]
    keep = width // 2
    mask = torch.zeros(1, 1, 1, width)
    mask[..., keep:] = float("-inf")
    base = _output(module, **{f"mask_{cache}": mask})
    # Rewriting the masked half of K and V, as stale cache entries would be, changes nothing ...
    noisy_k = inputs[f"k_{cache}"].clone()
    noisy_v = inputs[f"v_{cache}"].clone()
    noisy_k[..., keep:] = torch.randn_like(noisy_k[..., keep:]) * 10
    noisy_v[..., keep:] = torch.randn_like(noisy_v[..., keep:]) * 10
    torch.testing.assert_close(
        base,
        _output(module, **{f"mask_{cache}": mask, f"k_{cache}": noisy_k, f"v_{cache}": noisy_v}),
    )
    # ... while the same rewrite of the visible half does.
    visible_k = inputs[f"k_{cache}"].clone()
    visible_k[..., :keep] = torch.randn_like(visible_k[..., :keep]) * 10
    assert not torch.allclose(
        base, _output(module, **{f"mask_{cache}": mask, f"k_{cache}": visible_k})
    )


def test_draft_step_traces_once_per_context_bucket():
    module = DraftStep(checkpoint(layers=2), num_kv_heads=1).half().eval()
    for full_window in (16, 32):
        program = torch.export.export(module, example_inputs(module, 1, 8, full_window))
        # The full-attention K/V follow the bucket; the sliding ones stay at the window.
        placeholders = {
            n.name: n.meta["val"].shape for n in program.graph.nodes if n.op == "placeholder"
        }
        assert tuple(placeholders["k_full"]) == (1, 1, 4, full_window)
        assert tuple(placeholders["k_sliding"]) == (1, 1, 4, 8)


# ---- Scan table ----


VOCAB, HIDDEN, CENTROIDS = 8, 4, 2


def scan_checkpoint(untied: bool) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    t = {
        "model.embed_tokens.weight": torch.randn(VOCAB, HIDDEN, generator=g),
        "masked_embedding.token_ordering": torch.randperm(VOCAB, generator=g),
        "masked_embedding.centroids.weight": torch.randn(CENTROIDS, HIDDEN, generator=g),
    }
    if untied:
        t["lm_head.weight"] = torch.randn(VOCAB, HIDDEN, generator=g)
    return t


def test_output_table_prefers_untied_head():
    t = scan_checkpoint(untied=True)
    assert output_table(t) is t["lm_head.weight"]


def test_output_table_falls_back_to_embedding():
    t = scan_checkpoint(untied=False)
    assert output_table(t) is t["model.embed_tokens.weight"]


def test_scan_table_permutes_the_untied_head(tmp_path):
    t = scan_checkpoint(untied=True)
    path = tmp_path / "draft_scan.bin"
    write_scan_table(t, 1, path)
    index = json.loads(path.with_suffix(".json").read_text())["tensors"]["embed_perm_f16"]
    blob = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.float16)
    start = index["offset"] // 2
    rows = blob[start : start + VOCAB * HIDDEN].reshape(VOCAB, HIDDEN)
    expected = t["lm_head.weight"][t["masked_embedding.token_ordering"]].to(torch.float16)
    assert torch.equal(rows, expected)


def test_load_drafter_accepts_bracketed_directory(tmp_path):
    drafter = tmp_path / "model_[v1]"
    drafter.mkdir()
    save_file({"w": torch.ones(2, 2)}, str(drafter / "model.safetensors"))
    assert set(load_drafter(drafter)) == {"w"}


# ---- Checkpoint checks and bundle component ----


def checkpoint(vocab=8, hidden=4, centroids=2, scale=1.0, layers=1):
    g = torch.Generator().manual_seed(0)
    t = {
        "model.embed_tokens.weight": torch.randn(vocab, hidden, generator=g) * scale,
        "masked_embedding.token_ordering": torch.randperm(vocab, generator=g),
        "masked_embedding.centroids.weight": torch.randn(centroids, hidden, generator=g),
        "pre_projection.weight": torch.randn(hidden, 2 * hidden, generator=g),
        "post_projection.weight": torch.randn(hidden, hidden, generator=g),
        "model.norm.weight": torch.randn(hidden, generator=g),
    }
    for i in range(layers):
        p = f"model.layers.{i}."
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
            "self_attn.q_norm",
        ):
            t[p + name + ".weight"] = torch.ones(hidden)
        t[p + "layer_scalar"] = torch.ones(1)
        for name in (
            "self_attn.q_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ):
            t[p + name + ".weight"] = torch.randn(hidden, hidden, generator=g)
    return t


def test_scan_table_rejects_vocab_not_divisible_by_centroids(tmp_path):
    # vocab 10 / 4 centroids -> 2 per centroid: the tokens in slots 8 and 9 would be unreachable.
    with pytest.raises(SystemExit, match="not divisible"):
        write_scan_table(checkpoint(vocab=10, centroids=4), 1, tmp_path / SCAN_NAME)


def test_scan_table_rejects_zero_centroids(tmp_path):
    t = checkpoint()
    t["masked_embedding.centroids.weight"] = torch.zeros(0, 4)
    with pytest.raises(SystemExit, match="not divisible"):
        write_scan_table(t, 1, tmp_path / SCAN_NAME)


def test_rejects_weights_that_overflow_fp16():
    with pytest.raises(SystemExit, match="float16"):
        check_checkpoint(checkpoint(scale=1e6))


def test_rejects_non_finite_weights():
    t = checkpoint()
    t["model.norm.weight"][0] = float("nan")
    with pytest.raises(SystemExit, match="non-finite"):
        check_checkpoint(t)


def test_names_a_missing_tensor():
    t = checkpoint()
    del t["pre_projection.weight"]
    with pytest.raises(SystemExit, match="pre_projection.weight"):
        check_checkpoint(t)


def test_names_a_missing_layer_tensor():
    t = checkpoint(layers=2)
    del t["model.layers.1.mlp.up_proj.weight"]
    with pytest.raises(SystemExit, match="model.layers.1.mlp.up_proj.weight"):
        check_checkpoint(t)


def test_rejects_a_layer_pattern_the_runtime_does_not_implement():
    # The runtime treats only the last layer as full attention.
    with pytest.raises(SystemExit, match="layer_types"):
        check_checkpoint(
            checkpoint(layers=2), {"layer_types": ["full_attention", "sliding_attention"]}
        )


def test_accepts_the_supported_layer_pattern():
    check_checkpoint(checkpoint(layers=2), {"layer_types": ["sliding_attention", "full_attention"]})
    check_checkpoint(checkpoint(layers=2))


def test_sidecars_carry_a_format_version(tmp_path):
    write_scan_table(checkpoint(), 1, tmp_path / SCAN_NAME)
    assert json.loads((tmp_path / "draft_scan.json").read_text())["version"] == 1


def test_scaled_rms_norm_matches_the_textbook_form_without_overflowing_fp16():
    g = torch.Generator().manual_seed(0)
    weight = torch.randn(64, generator=g)
    x = torch.randn(3, 64, generator=g) * 100
    norm = RMSNorm(weight)
    reference = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.eps) * weight
    torch.testing.assert_close(norm(x), reference, rtol=1e-5, atol=1e-5)
    # |x| = 1000 squares past fp16's range (65504); the scaled square stays inside it.
    big = torch.full((1, 64), 1000.0, dtype=torch.float16)
    out = norm.half()(big)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(
        out.float(), weight.half().float().expand(1, 64), rtol=2e-3, atol=2e-3
    )


def write_bundle(tmp_path, emits_hidden_states):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    overrides = {"sliding_window": 512}
    if emits_hidden_states:
        overrides["emits_hidden_states"] = True
    metadata = {
        "assets": {"main": "model.aimodel"},
        "auxiliary_assets": {"per_layer_embeddings": "ple.safetensors"},
        "language": {"overrides": overrides},
    }
    (bundle / "metadata.json").write_text(json.dumps(metadata))
    return bundle


def write_drafter_files(tmp_path):
    out = tmp_path / "drafter"
    (out / "draft_step.aimodel").mkdir(parents=True)
    (out / "draft_step.aimodel" / "model.mlir").write_text("program")
    for name in DRAFTER_AUXILIARY_KEYS.values():
        (out / name).write_text(name)
    return out


def test_add_to_bundle_declares_the_drafter_component(tmp_path):
    bundle = write_bundle(tmp_path, emits_hidden_states=True)
    add_to_bundle(write_drafter_files(tmp_path), bundle)
    metadata = json.loads((bundle / "metadata.json").read_text())
    assert metadata["assets"] == {"main": "model.aimodel", "drafter": "draft_step.aimodel"}
    assert metadata["auxiliary_assets"] == {
        "per_layer_embeddings": "ple.safetensors",
        "drafter_config": "draft_step.json",
        "drafter_scan": "draft_scan.bin",
        "drafter_scan_index": "draft_scan.json",
    }
    assert metadata["speculative"] == {"drafter_kind": DRAFTER_KIND}
    assert (bundle / "draft_step.aimodel" / "model.mlir").read_text() == "program"
    for name in DRAFTER_AUXILIARY_KEYS.values():
        assert (bundle / name).read_text() == name


def test_add_to_bundle_keeps_the_rest_of_the_speculative_block(tmp_path):
    bundle = write_bundle(tmp_path, emits_hidden_states=True)
    path = bundle / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["speculative"] = {"num_draft_tokens": 4}
    path.write_text(json.dumps(metadata))
    add_to_bundle(write_drafter_files(tmp_path), bundle)
    assert json.loads(path.read_text())["speculative"] == {
        "num_draft_tokens": 4,
        "drafter_kind": DRAFTER_KIND,
    }


def test_add_to_bundle_can_be_repeated_and_warns_when_it_replaces_the_program(tmp_path, capsys):
    bundle = write_bundle(tmp_path, emits_hidden_states=True)
    out = write_drafter_files(tmp_path)
    add_to_bundle(out, bundle)
    assert "Warning" not in capsys.readouterr().out
    add_to_bundle(out, bundle)
    assert "Warning: replacing the existing draft_step.aimodel" in capsys.readouterr().out
    assert (
        json.loads((bundle / "metadata.json").read_text())["assets"]["drafter"]
        == "draft_step.aimodel"
    )


@pytest.mark.parametrize("speculative", [{}, {"drafter_kind": "dflash"}])
def test_add_to_bundle_refuses_a_bundle_with_another_drafter(tmp_path, speculative):
    bundle = write_bundle(tmp_path, emits_hidden_states=True)
    path = bundle / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["assets"]["drafter"] = "other.aimodel"
    metadata["speculative"] = speculative
    path.write_text(json.dumps(metadata))
    with pytest.raises(SystemExit, match="already declares a drafter"):
        add_to_bundle(write_drafter_files(tmp_path), bundle)
    assert json.loads(path.read_text())["assets"]["drafter"] == "other.aimodel"


def test_add_to_bundle_refuses_a_target_without_hidden_states(tmp_path):
    bundle = write_bundle(tmp_path, emits_hidden_states=False)
    with pytest.raises(SystemExit, match="--emit-hidden-states"):
        add_to_bundle(write_drafter_files(tmp_path), bundle)


class _TextConfig:
    sliding_window = 512
    head_dim = 256
    global_head_dim = 512
    final_logit_softcapping = 30.0
    rope_parameters = {
        "sliding_attention": {"rope_theta": 10000.0},
        "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.25},
    }


def test_metadata_records_hidden_states_only_when_emitted():
    assert "emits_hidden_states" not in _ios_metadata_extras(_TextConfig())["overrides"]
    assert _ios_metadata_extras(_TextConfig(), True)["overrides"]["emits_hidden_states"] is True


# ---- Finding the assistant ----


def test_the_default_assistant_is_the_targets_own():
    assert default_drafter("google/gemma-4-E2B-it") == "google/gemma-4-E2B-it-assistant"
    assert default_drafter("google/gemma-4-E4B-it") == "google/gemma-4-E4B-it-assistant"


def _record_downloads(monkeypatch, result):
    calls = []

    def fake(repo, **kwargs):
        calls.append((repo, kwargs))
        if isinstance(result, Exception):
            raise result
        return str(result)

    monkeypatch.setattr(export_drafter, "snapshot_download", fake)
    return calls


def test_a_local_directory_is_used_as_is(tmp_path, monkeypatch):
    calls = _record_downloads(monkeypatch, tmp_path / "unused")
    assert resolve_drafter(str(tmp_path), "google/gemma-4-E2B-it") == tmp_path
    assert calls == []


def test_an_omitted_drafter_downloads_the_targets_assistant(tmp_path, monkeypatch):
    calls = _record_downloads(monkeypatch, tmp_path)
    assert resolve_drafter(None, "google/gemma-4-E4B-it") == tmp_path
    # Only the weights and the config are fetched, not a model.
    assert calls == [
        (
            "google/gemma-4-E4B-it-assistant",
            {"allow_patterns": ["*.safetensors", "config.json"]},
        )
    ]


def test_a_repo_id_is_downloaded(tmp_path, monkeypatch):
    calls = _record_downloads(monkeypatch, tmp_path)
    assert resolve_drafter("someone/other-assistant", "google/gemma-4-E2B-it") == tmp_path
    assert calls[0][0] == "someone/other-assistant"


def test_neither_a_directory_nor_a_repo_id_is_refused(tmp_path, monkeypatch):
    calls = _record_downloads(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match="neither a directory nor a hub repo id"):
        resolve_drafter(str(tmp_path / "missing"), "google/gemma-4-E2B-it")
    assert calls == []


def test_a_missing_default_assistant_says_to_pass_one(monkeypatch):
    class NotFound(Exception):
        pass

    monkeypatch.setattr(export_drafter, "RepositoryNotFoundError", NotFound)
    _record_downloads(monkeypatch, NotFound("404"))
    with pytest.raises(SystemExit, match="no default assistant for acme/model.*--drafter"):
        resolve_drafter(None, "acme/model")
    with pytest.raises(SystemExit, match="was not found") as explicit:
        resolve_drafter("acme/typo-assistant", "acme/model")
    assert "pass --drafter" not in str(explicit.value)


def test_a_gated_assistant_says_to_log_in(monkeypatch):
    class Gated(Exception):
        pass

    monkeypatch.setattr(export_drafter, "GatedRepoError", Gated)
    _record_downloads(monkeypatch, Gated("401"))
    with pytest.raises(SystemExit, match="gated.*log in"):
        resolve_drafter("acme/assistant", "acme/model")


def test_an_assistant_for_another_target_is_refused():
    t = checkpoint(hidden=4)  # takes 4-wide hidden states
    check_checkpoint(t, backbone_hidden_size=4)
    with pytest.raises(SystemExit, match="4 wide.*target's hidden size is 6"):
        check_checkpoint(t, backbone_hidden_size=6)
