# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the EmbeddingGemma 2 text encoder (``models/macos/embedding_gemma2.py``).

Tiers
-----
1. Self-contained (always run): config parsing, padding invariance, pooling mask negative test,
   inputs_embeds path.
2. HF reference (tiny random config): needs ``transformers>=5.19`` (``models/embedding_gemma2``).
   The repo pins an older transformers, so run in an overlay::

       uv run --no-sync --with "transformers>=5.19" \\
           python -m pytest -q -p no:rerunfailures -p no:xdist -o addopts="" <this file>

3. Real weights (CPU fp32): additionally needs the ``google/embeddinggemma-2`` snapshot in the
   local HF cache (the ``embeddinggemma2_snapshot`` fixture skips otherwise). Compared against HF
   ``EmbeddingGemma2Model`` + the sentence-transformers mean-pool + L2-normalize
   re-implementation in ``_st_mean_pool_normalize``.

Tiers 2/3 skip cleanly when their dependencies are unavailable.
"""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F

from coreai_models.models.macos.embedding_gemma2 import (
    EmbeddingGemma2TextConfig,
    EmbeddingGemma2TextEncoder,
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


# --- Tier 2: tiny random config vs HF EmbeddingGemma2TextModel ---


def _hf_modeling():
    return pytest.importorskip(
        "transformers.models.embedding_gemma2.modeling_embedding_gemma2",
        reason="needs transformers>=5.19 (run in a uv --with overlay)",
    )


def _tiny_hf():
    """Tiny random HF ``EmbeddingGemma2TextModel`` (eager attention) with perturbed weights."""
    modeling = _hf_modeling()
    from transformers.models.embedding_gemma2.configuration_embedding_gemma2 import (
        EmbeddingGemma2Config,
    )

    torch.manual_seed(0)
    hf_cfg = EmbeddingGemma2Config(text_config=dict(TINY_TEXT_CONFIG)).text_config
    hf_cfg._attn_implementation = "eager"
    hf = modeling.EmbeddingGemma2TextModel(hf_cfg).eval()
    _randomize_(hf, seed=3)
    return hf


def test_tiny_parity_vs_hf():
    hf = _tiny_hf()

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

    prompts = json.loads((snapshot / "config_sentence_transformers.json").read_text())["prompts"]
    cfg = AutoConfig.from_pretrained(str(snapshot))
    cfg.vision_config = None
    cfg.audio_config = None
    hf = EmbeddingGemma2Model.from_pretrained(str(snapshot), config=cfg, dtype=torch.float32).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot))
    ours = load_embedding_gemma2_text_encoder(snapshot, dtype=torch.float32)
    texts = [prompts[p] + t for p, t in REAL_TEXTS]
    tokenize = lambda batch: tokenizer(batch, padding=True, return_tensors="pt")  # noqa: E731
    return {"hf": hf, "ours": ours, "texts": texts, "tokenize": tokenize}


def test_real_weight_parity_and_padding_invariance(real_setup):
    """Includes a row over 2 * 512 + 1 tokens, so the sliding layers' window bites for real."""
    ours, hf = real_setup["ours"], real_setup["hf"]
    paragraph = (
        "On-device semantic search embeds documents once and compares vectors at query time. "
    )
    texts = real_setup["texts"] + [real_setup["texts"][1] + paragraph * 90]
    features = real_setup["tokenize"](texts)
    ids = features["input_ids"].to(torch.int32)
    mask = features["attention_mask"].to(torch.int32)
    assert int(mask.sum(1).min()) < int(mask.sum(1).max()), "batch must be mixed-length"
    assert ids.shape[1] > 1100

    with torch.inference_mode():
        hf.language_model.config._attn_implementation = "sdpa"
        hf_tokens = hf(input_ids=ids.long(), attention_mask=mask.long()).last_hidden_state
        ref = _st_mean_pool_normalize(hf_tokens, mask)
        emb = ours(input_ids=ids, attention_mask=mask)
        alone = torch.cat(
            [
                ours(
                    input_ids=ids[r : r + 1, : int(mask[r].sum())],
                    attention_mask=mask[r : r + 1, : int(mask[r].sum())],
                )
                for r in range(len(texts))
            ]
        )

    assert _cos(emb, ref).min().item() >= 0.99999
    assert _cos(alone, emb).min().item() >= 0.99999


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
    assert _cos(a, b).item() < 0.999
