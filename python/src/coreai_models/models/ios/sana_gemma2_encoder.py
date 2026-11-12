# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Neural Engine float16 Gemma-2 text encoder for Sana Sprint, channels-first (BC1S) conv layout.

Same inputs and outputs as ``SanaTextEncoderWrapper``: ``(input_ids, attention_mask) [1, L]`` ->
``hidden_states [1, 300, D]`` (BOS plus the last 299 positions). Matches transformers' default
SDPA attention, which ignores ``attn_logit_softcapping``, as the GPU bundle does.

Weight folds, computed in float32: Gemma's ``(1 + w)`` norm scale, ``query_pre_attn_scalar^-0.5``
into ``q_proj`` (a power of two) and ``sqrt(hidden_size)`` into the token table. Norms are
max-abs; positions are static, so RoPE and the causal mask are constant buffers.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from coreai_models._constants import MASK_NEG
from coreai_models.diffusion.sana import TEXT_SEQUENCE_LENGTH
from coreai_models.primitives.ios.gelu import gelu_ane
from coreai_models.primitives.ios.rms_norm import MaxAbsRMSNorm


def conv1x1(cin: int, cout: int) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, kernel_size=1, bias=False)


def _load(conv: nn.Conv2d, linear: Any, scale: float = 1.0) -> None:
    conv.weight.data.copy_(linear.weight.detach().float()[:, :, None, None] * scale)


def _norm(dim: int, eps: float, src: Any) -> MaxAbsRMSNorm:
    norm = MaxAbsRMSNorm(dim, eps, axis=-3, floor="add")
    norm.weight.data.copy_((1.0 + src.weight.detach().float()).reshape(-1, 1, 1))
    return norm


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=2)
    return torch.cat([-x2, x1], dim=2)


class Attention(nn.Module):
    """Causal GQA over ``(1, D, 1, L)``.

    Heads are on dim 1 and head_dim on dim 2; the scores put keys on dim 2.
    """

    def __init__(self, cfg: Any, src: Any) -> None:
        super().__init__()
        self.heads = cfg.num_attention_heads
        self.kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        d = cfg.hidden_size
        self.q_proj = conv1x1(d, self.heads * self.head_dim)
        self.k_proj = conv1x1(d, self.kv_heads * self.head_dim)
        self.v_proj = conv1x1(d, self.kv_heads * self.head_dim)
        self.o_proj = conv1x1(self.heads * self.head_dim, d)
        _load(self.q_proj, src.q_proj, cfg.query_pre_attn_scalar**-0.5)
        _load(self.k_proj, src.k_proj)
        _load(self.v_proj, src.v_proj)
        _load(self.o_proj, src.o_proj)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        b, _, _, n = x.shape
        q = self.q_proj(x).reshape(b, self.heads, self.head_dim, n)
        k = self.k_proj(x).reshape(b, self.kv_heads, self.head_dim, n)
        v = self.v_proj(x).reshape(b, self.kv_heads, self.head_dim, n)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        rep = self.heads // self.kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        scores = torch.einsum("bhdk,bhdq->bhkq", k, q) + bias  # bias: (1, 1, L_k, L_q)
        attn = torch.softmax(scores, dim=2)
        out = torch.einsum("bhkq,bhdk->bhdq", attn, v)
        return self.o_proj(out.reshape(b, -1, 1, n))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: Any, src: Any) -> None:
        super().__init__()
        d, eps = cfg.hidden_size, cfg.rms_norm_eps
        self.self_attn = Attention(cfg, src.self_attn)
        self.input_layernorm = _norm(d, eps, src.input_layernorm)
        self.post_attention_layernorm = _norm(d, eps, src.post_attention_layernorm)
        self.pre_feedforward_layernorm = _norm(d, eps, src.pre_feedforward_layernorm)
        self.post_feedforward_layernorm = _norm(d, eps, src.post_feedforward_layernorm)
        self.gate_proj = conv1x1(d, cfg.intermediate_size)
        self.up_proj = conv1x1(d, cfg.intermediate_size)
        self.down_proj = conv1x1(cfg.intermediate_size, d)
        _load(self.gate_proj, src.mlp.gate_proj)
        _load(self.up_proj, src.mlp.up_proj)
        _load(self.down_proj, src.mlp.down_proj)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        h = self.self_attn(self.input_layernorm(x), cos, sin, bias)
        x = x + self.post_attention_layernorm(h)
        h = self.pre_feedforward_layernorm(x)
        h = self.down_proj(gelu_ane(self.gate_proj(h)) * self.up_proj(h))
        return x + self.post_feedforward_layernorm(h)


class SanaGemma2Encoder(nn.Module):
    """``(input_ids [1, L], attention_mask [1, L]) -> hidden_states [1, 300, D]``, static ``L``."""

    cos: torch.Tensor
    sin: torch.Tensor
    causal: torch.Tensor
    mask_neg: torch.Tensor
    one: torch.Tensor

    def __init__(self, src: Any, length: int) -> None:
        super().__init__()
        cfg = src.config
        assert cfg.hidden_activation == "gelu_pytorch_tanh"
        assert cfg.sliding_window >= length, "the sliding-window mask would differ from causal"
        d = cfg.hidden_size
        embed = src.embed_tokens
        table = embed.weight.detach().float() * float(embed.scalar_embed_scale)
        self.embed_tokens = nn.Embedding.from_pretrained(table)
        self.layers = nn.ModuleList([DecoderLayer(cfg, layer) for layer in src.layers])
        self.norm = _norm(d, cfg.rms_norm_eps, src.norm)

        pos = torch.arange(length, dtype=torch.float64)
        inv_freq = src.rotary_emb.inv_freq.detach().double()
        freqs = torch.outer(inv_freq, pos)  # (head_dim / 2, L)
        emb = torch.cat([freqs, freqs], dim=0)  # (head_dim, L)
        self.register_buffer("cos", emb.cos().float().reshape(1, 1, -1, length))
        self.register_buffer("sin", emb.sin().float().reshape(1, 1, -1, length))
        key = torch.arange(length)[:, None]
        query = torch.arange(length)[None, :]
        causal = torch.where(key > query, MASK_NEG, 0.0)
        self.register_buffer("causal", causal.reshape(1, 1, length, length))
        self.register_buffer("mask_neg", torch.tensor(MASK_NEG))
        self.register_buffer("one", torch.tensor(1.0))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        n = input_ids.shape[1]
        x = self.embed_tokens(input_ids).transpose(1, 2).unsqueeze(2)  # (1, D, 1, L)
        dtype = x.dtype
        pad = (self.one - attention_mask.to(dtype)) * self.mask_neg  # 0 for valid keys
        # Minimum, not sum: a padded key past the diagonal stays at MASK_NEG instead of 2x.
        bias = torch.minimum(self.causal, pad.reshape(1, 1, n, 1))
        for layer in self.layers:
            x = layer(x, self.cos, self.sin, bias)
        x = self.norm(x)
        x = torch.cat([x[..., :1], x[..., -(TEXT_SEQUENCE_LENGTH - 1) :]], dim=-1)
        return x.squeeze(2).transpose(1, 2)

    @classmethod
    def from_source_model(cls, src: Any, length: int) -> SanaGemma2Encoder:
        """Build from a transformers ``Gemma2Model`` (any dtype); folds run in float32."""
        return cls(src, length).float().eval()
