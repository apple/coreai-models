# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Neural Engine EG2 text encoder (``models/ios/embedding_gemma2.py``) vs the macOS reference."""

import math

import pytest
import torch

from coreai_models.embedding.export_ios import EncoderBandWrapper
from coreai_models.embedding.export_utils import TokenTableExportWrapper, scaled_token_table
from coreai_models.models.ios.embedding_gemma2 import (
    MASK_NEG,
    load_embedding_gemma2_text_encoder,
    text_encoder_from_state_dict,
)
from coreai_models.models.macos.embedding_gemma2 import (
    EmbeddingGemma2TextConfig,
    EmbeddingGemma2TextEncoder,
)
from coreai_models.models.macos.embedding_gemma2 import (
    load_embedding_gemma2_text_encoder as load_macos_text_encoder,
)
from tests.test_model_units.test_models._embedding_gemma2_common import (
    TINY_TEXT_CONFIG,
    _cos,
    _ragged_batch,
)


def _host_inputs(
    ids: torch.Tensor,
    attention_mask: torch.Tensor,
    embed_table: torch.Tensor,
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encoder inputs for ``ids [1, L]``: scaled embeddings, bias ``[1, L, 1, L]``, pool weights."""
    length = ids.shape[1]
    table = scaled_token_table(embed_table, math.sqrt(embed_table.shape[1]), dtype)
    key_ok = attention_mask[0].bool().reshape(length, 1, 1).expand(length, 1, length)
    bias = torch.where(key_ok, 0.0, MASK_NEG).reshape(1, length, 1, length)
    pool = attention_mask.float() / attention_mask.float().sum(-1, keepdim=True)
    return TokenTableExportWrapper(table, dtype)(ids), bias.to(dtype), pool.to(dtype)


def _banded(ios: torch.nn.Module, length: int) -> EncoderBandWrapper:
    return EncoderBandWrapper(ios, length, ios.config.sliding_window)


def _perturb(model: torch.nn.Module, seed: int) -> None:
    """Randomize the norm weights and layer scalars of ``model``."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, tensor in sorted(model.state_dict().items()):
            if name.endswith("layer_scalar"):
                tensor.copy_(0.5 + torch.rand(tensor.shape, generator=gen))
            elif "norm" in name:
                tensor.copy_(1.0 + 0.3 * torch.randn(tensor.shape, generator=gen))


def _reference_and_state_dict():
    torch.manual_seed(0)
    cfg = EmbeddingGemma2TextConfig.from_dict(TINY_TEXT_CONFIG)
    ref = EmbeddingGemma2TextEncoder(cfg)
    _perturb(ref, seed=0)
    ref.eval()
    state = {k: v.detach().clone() for k, v in ref.state_dict().items()}
    return cfg, ref, state


def test_float32_matches_macos_reference_per_row() -> None:
    cfg, ref, state = _reference_and_state_dict()
    ios, table = text_encoder_from_state_dict(cfg, state, dtype=torch.float32)
    ids, mask = _ragged_batch([12, 7, 4], seq_len=12, vocab=cfg.vocab_size)
    with torch.no_grad():
        want = ref(input_ids=ids, attention_mask=mask)
        # Length 12 takes the banded path (> sliding_window + 1); length 4 the unbanded one.
        for row, length in ((0, 12), (1, 12), (2, 12), (2, 4)):
            r = slice(row, row + 1)
            inputs = _host_inputs(ids[r, :length], mask[r, :length], table, torch.float32)
            got = _banded(ios, length)(*inputs)
            assert got.shape == (1, cfg.embedding_dim)
            cos = _cos(got, want[r])
            assert float(cos) >= 0.9999, (row, length, float(cos))


def test_float16_stays_finite_with_large_activations() -> None:
    cfg, _, state = _reference_and_state_dict()
    _, table = text_encoder_from_state_dict(cfg, state, dtype=torch.float32)
    ids, mask = _ragged_batch([12], seq_len=12, vocab=cfg.vocab_size)
    # Scale the token table to a peak scaled embedding of 3000, past sqrt(float16 max).
    peak = table[ids.long()].abs().max().item() * math.sqrt(cfg.hidden_size)
    table = table * (3000.0 / peak)
    ios32, _ = text_encoder_from_state_dict(cfg, state, dtype=torch.float32)
    ios16, _ = text_encoder_from_state_dict(cfg, state, dtype=torch.float16)
    with torch.no_grad():
        want = _banded(ios32, 12)(*_host_inputs(ids, mask, table, torch.float32))
        got = _banded(ios16, 12)(*_host_inputs(ids, mask, table, torch.float16))
    assert torch.isfinite(got.float()).all()
    assert float(_cos(got.float(), want)) >= 0.99


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_all_zero_pool_weights_give_a_zero_vector(dtype: torch.dtype) -> None:
    cfg, _, state = _reference_and_state_dict()
    ios, table = text_encoder_from_state_dict(cfg, state, dtype=dtype)
    ids, mask = _ragged_batch([6], seq_len=6, vocab=cfg.vocab_size)
    embeds, bias, pool = _host_inputs(ids, mask, table, dtype)
    with torch.no_grad():
        got = _banded(ios, 6)(embeds, bias, torch.zeros_like(pool))
    assert torch.equal(got, torch.zeros_like(got))


@pytest.mark.parametrize(("length", "valid"), [(64, 40), (1024, 700)])
def test_real_weight_float16_matches_float32(embeddinggemma2_snapshot, length, valid) -> None:
    reference = load_macos_text_encoder(embeddinggemma2_snapshot, dtype=torch.float32).eval()
    ios16, table = load_embedding_gemma2_text_encoder(embeddinggemma2_snapshot, torch.float16)
    gen = torch.Generator().manual_seed(length)
    ids = torch.randint(3, table.shape[0], (1, length), generator=gen, dtype=torch.int32)
    ids[0, 0], ids[0, valid - 1] = 2, 1  # BOS ... EOS
    mask = (torch.arange(length) < valid).to(torch.int32)[None]
    with torch.no_grad():
        want = reference(input_ids=ids, attention_mask=mask)
        got = _banded(ios16, length)(*_host_inputs(ids, mask, table, torch.float16))
    assert torch.isfinite(got).all()
    assert float(_cos(got.float(), want)) >= 0.9999
