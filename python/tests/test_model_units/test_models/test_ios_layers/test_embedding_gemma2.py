# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Neural Engine EG2 text encoder (``models/ios/embedding_gemma2.py``) vs the macOS reference."""

import math

import pytest
import torch

from coreai_models.models.ios.embedding_gemma2 import (
    MASK_NEG,
    ScaledRMSNorm,
    text_encoder_from_state_dict,
)
from coreai_models.models.macos.embedding_gemma2 import (
    EmbeddingGemma2TextConfig,
    EmbeddingGemma2TextEncoder,
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
    sliding_window: int,
    dtype: torch.dtype = torch.float16,
) -> dict[str, torch.Tensor]:
    """Graph inputs for one sequence (``ids [1, L]``): scaled embeddings, bias, pooling weights."""
    b, length = ids.shape
    emb = embed_table[ids.long()].float() * math.sqrt(embed_table.shape[1])
    valid = attention_mask.bool()
    key_ok = valid[0].reshape(length, 1, 1).expand(length, 1, length)  # (Lk, 1, Lq)
    bias = torch.where(key_ok, 0.0, MASK_NEG).reshape(1, length, 1, length)
    out = {
        "inputs_embeds": emb.to(dtype),
        "bias": bias.to(dtype),
        "pool_weights": (attention_mask.float() / attention_mask.float().sum(-1, keepdim=True)).to(
            dtype
        ),
    }
    if length > sliding_window + 1:
        pos = torch.arange(length)
        band = (pos[:, None] - pos[None, :]).abs() <= sliding_window  # (Lk, Lq) symmetric
        sb = torch.where(key_ok & band.reshape(length, 1, length), 0.0, MASK_NEG)
        out["sliding_bias"] = sb.reshape(1, length, 1, length).to(dtype)
    return out


def _perturb(model: torch.nn.Module, seed: int) -> None:
    """Randomize norm weights and layer scalars so the comparison exercises them."""
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
        for row in range(ids.shape[0]):
            n = int(mask[row].sum())
            inputs = _host_inputs(
                ids[row : row + 1], mask[row : row + 1], table, cfg.sliding_window, torch.float32
            )
            got = ios(**inputs)
            assert got.shape == (1, cfg.embedding_dim)
            cos = _cos(got, want[row : row + 1])
            assert float(cos) >= 0.9999, (row, n, float(cos))


def test_float16_stays_finite_with_large_activations() -> None:
    cfg, _, state = _reference_and_state_dict()
    _, table = text_encoder_from_state_dict(cfg, state, dtype=torch.float32)
    ids, mask = _ragged_batch([12], seq_len=12, vocab=cfg.vocab_size)
    # Scale the token embeddings so the hidden stream reaches about 3e3, past sqrt(fp16 max).
    peak = table[ids.long()].abs().max().item() * math.sqrt(cfg.hidden_size)
    table = table * (3000.0 / peak)
    ios32, _ = text_encoder_from_state_dict(cfg, state, dtype=torch.float32)
    ios16, _ = text_encoder_from_state_dict(cfg, state, dtype=torch.float16)
    with torch.no_grad():
        want = ios32(**_host_inputs(ids, mask, table, cfg.sliding_window, torch.float32))
        got = ios16(**_host_inputs(ids, mask, table, cfg.sliding_window, torch.float16))
    assert torch.isfinite(got.float()).all()
    assert float(_cos(got.float(), want)) >= 0.99


@pytest.mark.parametrize("peak", [0.0, 1e-7, 1e-5, 1e-3, 3.9e-3, 4e-3, 1.0, 1e2, 1e4, 6e4])
def test_float16_scaled_rms_norm_matches_float32_across_magnitudes(peak: float) -> None:
    eps = 1e-6
    x = torch.randn(8, 256, generator=torch.Generator().manual_seed(0))
    x = x / x.abs().amax(1, keepdim=True) * peak
    x16 = x.half()
    want = x16.float() * torch.rsqrt((x16.float() ** 2).mean(1, keepdim=True) + eps)
    with torch.no_grad():
        got = ScaledRMSNorm(256, eps, with_scale=False).half()(x16).float()
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, want, rtol=2e-3, atol=2e-3)
