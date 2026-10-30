# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the EmbeddingGemma 2 text encoder (``models/macos/embedding_gemma2.py``).

Tiers
-----
1. Self-contained (always run): config parsing, mask semantics, padding invariance, pooling
   mask negative test, inputs_embeds path, torch.export at a static bucket.
2. HF reference (tiny random config): needs ``transformers>=5.19`` (``models/embedding_gemma2``).
   The repo pins an older transformers, so run in an overlay::

       uv run --no-sync --with "transformers>=5.19" --with "sentence-transformers>=6.1" \\
           python -m pytest -q -p no:rerunfailures -p no:xdist -o addopts="" <this file>

3. Real weights (CPU fp32): additionally needs the ``google/embeddinggemma-2`` snapshot in the
   local HF cache (the ``embeddinggemma2_snapshot`` fixture skips otherwise). Compared against
   sentence-transformers>=6.1 when present; otherwise against HF ``EmbeddingGemma2Model`` + the
   mean-pool + L2-normalize re-implementation in ``_st_mean_pool_normalize``.

Tiers 2/3 skip cleanly when their dependencies are unavailable.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from coreai_models.models.macos.embedding_gemma2 import (
    EmbeddingGemma2TextConfig,
    EmbeddingGemma2TextEncoder,
    build_attention_masks,
    load_embedding_gemma2_text_encoder,
)
from tests.test_model_units.test_models._embedding_gemma2_common import (
    TINY_TEXT_CONFIG,
    _cos,
    _ragged_batch,
    _st_mean_pool_normalize,
)


def _randomize_(module: torch.nn.Module, seed: int = 0) -> None:
    """Perturb every parameter/buffer (norm weights, layer_scalar included) so bugs are visible."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, tensor in sorted(module.state_dict().items()):
            if not tensor.is_floating_point():
                continue
            if name.endswith("layer_scalar"):
                tensor.copy_(0.5 + torch.rand(tensor.shape, generator=gen))
            elif "norm" in name:
                tensor.copy_(1.0 + 0.3 * torch.randn(tensor.shape, generator=gen))
            else:
                tensor.copy_(0.2 * torch.randn(tensor.shape, generator=gen))


def _tiny_model(seed: int = 0) -> EmbeddingGemma2TextEncoder:
    model = EmbeddingGemma2TextEncoder(EmbeddingGemma2TextConfig.from_dict(TINY_TEXT_CONFIG))
    _randomize_(model, seed)
    return model.eval()


# --- Tier 1: self-contained ---


def test_config_from_dict_resolves_per_layer_overrides():
    cfg = EmbeddingGemma2TextConfig.from_dict(TINY_TEXT_CONFIG)
    assert [cfg.layer_head_dim(i) for i in range(4)] == [16, 32, 16, 32]
    assert [cfg.layer_num_key_value_heads(i) for i in range(4)] == [2, 1, 2, 1]
    assert (cfg.rope_theta_sliding, cfg.rope_theta_full) == (10_000.0, 1_000_000.0)


@pytest.mark.parametrize("key", ["layer_types", "per_layer_config", "rope_parameters"])
def test_config_requires_the_layer_layout(key):
    config = {k: v for k, v in TINY_TEXT_CONFIG.items() if k != key}
    with pytest.raises(ValueError, match=key):
        EmbeddingGemma2TextConfig.from_dict(config)


def test_config_requires_both_rope_thetas():
    rope = {"sliding_attention": {"rope_theta": 10_000.0}, "full_attention": {}}
    with pytest.raises(ValueError, match="full_attention"):
        EmbeddingGemma2TextConfig.from_dict({**TINY_TEXT_CONFIG, "rope_parameters": rope})


def test_config_from_snapshot_matches_published_layout(embeddinggemma2_snapshot):
    cfg = EmbeddingGemma2TextConfig.from_pretrained(embeddinggemma2_snapshot)
    full = [i for i in range(cfg.num_hidden_layers) if not cfg.is_sliding(i)]
    assert full == [5, 11, 17, 23]
    assert all(cfg.layer_head_dim(i) == 512 and cfg.layer_num_key_value_heads(i) == 1 for i in full)
    assert cfg.layer_head_dim(0) == 256 and cfg.layer_num_key_value_heads(0) == 2
    assert (cfg.hidden_size, cfg.embedding_dim, cfg.sliding_window) == (512, 768, 512)


def test_masks_bidirectional_window_and_key_padding():
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0]], dtype=torch.int32)
    full, sliding = build_attention_masks(attention_mask, sliding_window=2)
    assert full.shape == sliding.shape == (1, 1, 7, 7)
    i, j = torch.meshgrid(torch.arange(7), torch.arange(7), indexing="ij")
    key_valid = (j < 5).unsqueeze(0).unsqueeze(0)
    assert torch.equal(full, key_valid.expand_as(full))
    assert torch.equal(sliding, ((i - j).abs() <= 2).unsqueeze(0).unsqueeze(0) & key_valid)
    # Symmetric (bidirectional) on valid positions.
    valid = sliding[0, 0, :5, :5]
    assert torch.equal(valid, valid.T)


def test_forward_shapes_and_unit_norm():
    model = _tiny_model()
    ids, mask = _ragged_batch([9, 5], seq_len=12, vocab=128)
    emb, tokens = model(input_ids=ids, attention_mask=mask, return_token_states=True)
    assert emb.shape == (2, 48) and emb.dtype == torch.float32
    assert tokens.shape == (2, 12, 48)
    torch.testing.assert_close(emb.norm(dim=-1), torch.ones(2), atol=1e-6, rtol=0)


def test_padding_invariance_tiny():
    """A sequence embedded alone (no padding) == the same sequence inside a padded batch."""
    model = _tiny_model()
    ids, mask = _ragged_batch([11, 4, 7], seq_len=16, vocab=128)
    with torch.no_grad():
        batched = model(input_ids=ids, attention_mask=mask)
        for row, n in enumerate([11, 4, 7]):
            alone = model(input_ids=ids[row : row + 1, :n], attention_mask=mask[row : row + 1, :n])
            assert _cos(alone[0], batched[row]).item() > 1 - 1e-6
            torch.testing.assert_close(alone[0], batched[row], atol=2e-5, rtol=0)


def test_pooling_mask_excluding_prompt_changes_result():
    model = _tiny_model()
    ids, mask = _ragged_batch([10], seq_len=10, vocab=128)
    no_prompt = mask.clone()
    no_prompt[:, :4] = 0  # pretend the first 4 tokens are the task prompt
    with torch.no_grad():
        with_prompt = model(input_ids=ids, attention_mask=mask)
        without_prompt = model(input_ids=ids, attention_mask=mask, pooling_mask=no_prompt)
    assert _cos(with_prompt, without_prompt).item() < 0.9999


def test_projection_after_pool_equals_project_then_pool():
    model = _tiny_model()
    ids, mask = _ragged_batch([8, 3], seq_len=10, vocab=128)
    with torch.no_grad():
        emb, tokens = model(input_ids=ids, attention_mask=mask, return_token_states=True)
    m = mask.float().unsqueeze(-1)
    reference = F.normalize((tokens * m).sum(1) / m.sum(1).clamp(min=1e-9), dim=-1)
    torch.testing.assert_close(emb, reference, atol=1e-6, rtol=0)


def test_inputs_embeds_path_matches_input_ids():
    model = _tiny_model()
    ids, mask = _ragged_batch([6, 6], seq_len=6, vocab=128)
    with torch.no_grad():
        a = model(input_ids=ids, attention_mask=mask)
        b = model(inputs_embeds=model.embed_tokens(ids), attention_mask=mask)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    with pytest.raises(ValueError):
        model(input_ids=ids, inputs_embeds=model.embed_tokens(ids))


def test_torch_export_static_bucket():
    """torch.export at a static (B=2, L=128) bucket with int32 ids/mask; outputs match eager."""
    model = _tiny_model()
    ids, mask = _ragged_batch([128, 37], seq_len=128, vocab=128)
    with torch.no_grad():
        exported = torch.export.export(
            model, args=(), kwargs={"input_ids": ids, "attention_mask": mask}
        )
        eager = model(input_ids=ids, attention_mask=mask)
        traced = exported.module()(input_ids=ids, attention_mask=mask)
    assert traced.shape == (2, 48)
    torch.testing.assert_close(traced, eager, atol=1e-6, rtol=0)
    placeholders = [
        n
        for n in exported.graph.nodes
        if n.op == "placeholder" and n.name in ("input_ids", "attention_mask")
    ]
    assert len(placeholders) == 2
    assert all(n.meta["val"].dtype == torch.int32 for n in placeholders)


# --- Tier 2: tiny random config vs HF EmbeddingGemma2TextModel ---


def _hf_modeling():
    return pytest.importorskip(
        "transformers.models.embedding_gemma2.modeling_embedding_gemma2",
        reason="needs transformers>=5.19 (run in a uv --with overlay)",
    )


def _tiny_hf(attn_impl: str):
    """Tiny random HF ``EmbeddingGemma2TextModel`` with perturbed weights."""
    modeling = _hf_modeling()
    from transformers.models.embedding_gemma2.configuration_embedding_gemma2 import (
        EmbeddingGemma2Config,
    )

    torch.manual_seed(0)
    hf_cfg = EmbeddingGemma2Config(text_config=dict(TINY_TEXT_CONFIG)).text_config
    hf_cfg._attn_implementation = attn_impl
    hf = modeling.EmbeddingGemma2TextModel(hf_cfg).eval()
    _randomize_(hf, seed=3)
    return hf


@pytest.mark.parametrize("attn_impl", ["eager", "sdpa"])
def test_tiny_parity_vs_hf(attn_impl):
    hf = _tiny_hf(attn_impl)

    ours = EmbeddingGemma2TextEncoder(EmbeddingGemma2TextConfig.from_dict(TINY_TEXT_CONFIG)).eval()
    ours.load_state_dict(hf.state_dict())

    # Long enough (L=24 > 2*window+1) that the sliding window actually bites, with padding.
    ids, mask = _ragged_batch([24, 13, 5], seq_len=24, vocab=128, seed=7)
    with torch.no_grad():
        hf_tokens = hf(input_ids=ids.long(), attention_mask=mask.long()).last_hidden_state
        hf_emb = _st_mean_pool_normalize(hf_tokens, mask)
        emb, tokens = ours(input_ids=ids, attention_mask=mask, return_token_states=True)

    valid = mask.bool()
    torch.testing.assert_close(tokens[valid], hf_tokens[valid], atol=1e-4, rtol=1e-4)
    assert _cos(emb, hf_emb).min().item() > 1 - 1e-6
    torch.testing.assert_close(emb, hf_emb, atol=1e-5, rtol=0)


def test_tiny_sliding_window_matters_vs_hf():
    """Guard against a silently-ignored window: widening ours must break parity with HF."""
    hf = _tiny_hf("eager")
    wide = dict(TINY_TEXT_CONFIG, sliding_window=1000)
    ours = EmbeddingGemma2TextEncoder(EmbeddingGemma2TextConfig.from_dict(wide)).eval()
    ours.load_state_dict(hf.state_dict())
    ids, mask = _ragged_batch([24], seq_len=24, vocab=128, seed=7)
    with torch.no_grad():
        hf_emb = _st_mean_pool_normalize(
            hf(input_ids=ids.long(), attention_mask=mask.long()).last_hidden_state, mask
        )
        emb = ours(input_ids=ids, attention_mask=mask)
    assert (emb - hf_emb).abs().max().item() > 1e-3


# --- Tier 3: real weights (CPU fp32) ---

REAL_TEXTS = [
    ("query", "What is the capital of France?"),
    ("document", "Paris is the capital and most populous city of France."),
    ("query", "how do neural networks learn"),
    ("document", "Gradient descent iteratively updates the weights of a model to minimize a loss."),
    ("STS", "A man is playing a guitar on stage."),
    ("document", "El aprendizaje automático es una rama de la inteligencia artificial."),
    ("CodeRetrieval", "def add(a, b):\n    return a + b"),
    ("query", "x"),
]


@pytest.fixture(scope="module")
def real_setup(embeddinggemma2_snapshot):
    snapshot = embeddinggemma2_snapshot
    _hf_modeling()
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import EmbeddingGemma2Model

    prompts = None
    st_model = None
    try:
        from sentence_transformers import SentenceTransformer

        st_model = SentenceTransformer(
            str(snapshot), device="cpu", model_kwargs={"dtype": torch.float32}
        )
        prompts = st_model.prompts
    except Exception:  # noqa: BLE001 - optional reference
        import json

        prompts = json.loads((snapshot / "config_sentence_transformers.json").read_text())[
            "prompts"
        ]

    cfg = AutoConfig.from_pretrained(str(snapshot))
    cfg.vision_config = None
    cfg.audio_config = None
    hf = EmbeddingGemma2Model.from_pretrained(str(snapshot), config=cfg, dtype=torch.float32).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot))
    ours = load_embedding_gemma2_text_encoder(snapshot, dtype=torch.float32)
    texts = [prompts[p] + t for p, t in REAL_TEXTS]
    if st_model is not None:
        tokenize = lambda batch: st_model.tokenize(batch)  # noqa: E731
    else:
        tokenize = lambda batch: tokenizer(batch, padding=True, return_tensors="pt")  # noqa: E731
    return {"st": st_model, "hf": hf, "ours": ours, "texts": texts, "tokenize": tokenize}


def test_real_weight_key_coverage(real_setup, embeddinggemma2_snapshot):
    from safetensors import safe_open

    with safe_open(str(embeddinggemma2_snapshot / "model.safetensors"), framework="pt") as handle:
        all_keys = list(handle.keys())  # noqa: SIM118 - safe_open is not a Mapping
        hf_text_keys = {
            k[len("language_model.") :] for k in all_keys if k.startswith("language_model.")
        }
    ours = set(real_setup["ours"].state_dict().keys())
    assert ours == hf_text_keys, (sorted(ours - hf_text_keys)[:5], sorted(hf_text_keys - ours)[:5])


def test_real_weight_parity_and_padding_invariance(real_setup):
    ours, hf, texts = real_setup["ours"], real_setup["hf"], real_setup["texts"]
    features = real_setup["tokenize"](texts)
    ids = features["input_ids"].to(torch.int32)
    mask = features["attention_mask"].to(torch.int32)
    assert int(mask.sum(1).min()) < int(mask.sum(1).max()), "batch must be mixed-length"

    with torch.inference_mode():
        hf.language_model.config._attn_implementation = "sdpa"
        hf_tokens = hf(input_ids=ids.long(), attention_mask=mask.long()).last_hidden_state
        ref = _st_mean_pool_normalize(hf_tokens, mask)
        if real_setup["st"] is not None:
            st_emb = real_setup["st"].encode(texts, batch_size=len(texts), convert_to_tensor=True)
            assert _cos(st_emb, ref).min().item() > 0.99999  # our pooling re-impl == ST
            ref = st_emb
        emb, tokens = ours(input_ids=ids, attention_mask=mask, return_token_states=True)
        alone = torch.cat(
            [
                ours(
                    input_ids=ids[r : r + 1, : int(mask[r].sum())],
                    attention_mask=mask[r : r + 1, : int(mask[r].sum())],
                )
                for r in range(len(texts))
            ]
        )

    cos = _cos(emb, ref)
    emb_max_abs = (emb - ref).abs().max().item()
    pad_cos = _cos(alone, emb)
    print(
        f"\nEG2 real fp32 CPU: n={len(texts)} lens={mask.sum(1).tolist()} "
        f"cos min={cos.min().item():.9f} median={cos.median().item():.9f} "
        f"emb max_abs={emb_max_abs:.3e} "
        f"padding-invariance cos min={pad_cos.min().item():.9f}"
    )
    assert cos.min().item() >= 0.99999
    assert pad_cos.min().item() >= 0.99999


def test_real_weight_excluding_prompt_changes_embedding(real_setup):
    ours, texts = real_setup["ours"], real_setup["texts"]
    features = real_setup["tokenize"](texts[:1])
    ids = features["input_ids"].to(torch.int32)
    mask = features["attention_mask"].to(torch.int32)
    prompt_len = int(
        real_setup["tokenize"]([texts[0][: texts[0].index("query: ") + len("query:")]])[
            "attention_mask"
        ].sum()
    )
    no_prompt = mask.clone()
    no_prompt[:, :prompt_len] = 0
    with torch.inference_mode():
        a = ours(input_ids=ids, attention_mask=mask)
        b = ours(input_ids=ids, attention_mask=mask, pooling_mask=no_prompt)
    cos = _cos(a, b).item()
    print(f"\nEG2 include_prompt vs exclude_prompt cos={cos:.6f} (prompt_len={prompt_len})")
    assert cos < 0.999


def test_real_weight_long_sequence_window_bites(real_setup):
    """> 2 * 512 + 1 tokens so the sliding layers' |i - j| <= 512 window is exercised for real."""
    ours, hf = real_setup["ours"], real_setup["hf"]
    paragraph = (
        "On-device semantic search embeds documents once and compares vectors at query time. "
    )
    features = real_setup["tokenize"](
        [real_setup["texts"][1] + paragraph * 90, real_setup["texts"][0]]
    )
    ids = features["input_ids"].to(torch.int32)
    mask = features["attention_mask"].to(torch.int32)
    assert ids.shape[1] > 1100
    with torch.inference_mode():
        hf.language_model.config._attn_implementation = "sdpa"
        ref = _st_mean_pool_normalize(
            hf(input_ids=ids.long(), attention_mask=mask.long()).last_hidden_state, mask
        )
        emb = ours(input_ids=ids, attention_mask=mask)
    cos = _cos(emb, ref)
    print(
        f"\nEG2 long L={ids.shape[1]} lens={mask.sum(1).tolist()} cos={cos.tolist()} "
        f"max_abs={(emb - ref).abs().max().item():.3e}"
    )
    assert cos.min().item() >= 0.99999
