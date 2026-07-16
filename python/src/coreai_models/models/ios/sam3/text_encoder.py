# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Re-authored SAM3 text encoder in BC1S layout.

24 transformer layers from the HF ``CLIPTextModelWithProjection``,
re-implemented for iOS: ``nn.Linear`` becomes
``nn.Conv2d(1x1)`` and the causal mask uses ``-40000.0`` (not ``-inf``)
on masked positions.
"""

import torch
import torch.nn as nn

from coreai_models.primitives.ios.bidirectional_sdpa import BidirectionalSDPA
from coreai_models.primitives.ios.gelu import GELUReauthored
from coreai_models.primitives.ios.layer_norm import LayerNormReauthored


def _make_causal_mask(seq_len: int) -> torch.Tensor:
    """Causal mask in GPU SDPA layout (1, 1, seq_len, seq_len), f16."""
    mask = torch.full((seq_len, seq_len), -40000.0, dtype=torch.float16)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


class TextEncoderAttention(nn.Module):
    """Multi-head causal attention with Conv2d(1x1) projections."""

    def __init__(self, embed_dim: int, num_heads: int, max_seq_len: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True)
        self.k_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True)
        self.v_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True)
        self.out_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True)

        self.sdpa = BidirectionalSDPA(num_heads=num_heads, head_dim=self.head_dim)

        self.register_buffer("causal_mask", _make_causal_mask(max_seq_len))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        seq_len = hidden_states.shape[3]
        mask = self.causal_mask[:, :, :seq_len, :seq_len].expand(-1, self.num_heads, -1, -1)

        out = self.sdpa(q, k, v, attention_mask=mask)
        return self.out_proj(out)

    @classmethod
    def from_hf_attention(cls, hf_attn: nn.Module, max_seq_len: int = 32) -> "TextEncoderAttention":
        ane_attn = cls(
            embed_dim=hf_attn.embed_dim,
            num_heads=hf_attn.num_heads,
            max_seq_len=max_seq_len,
        )
        for name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            hf_linear = getattr(hf_attn, name)
            ane_conv = getattr(ane_attn, name)
            ane_conv.weight.data = hf_linear.weight.data.reshape(
                hf_linear.out_features, hf_linear.in_features, 1, 1
            )
            if hf_linear.bias is not None:
                ane_conv.bias.data = hf_linear.bias.data.clone()
        return ane_attn


class TextEncoderLayer(nn.Module):
    """Pre-norm CLIP encoder layer in BC1S layout."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        intermediate_size: int,
        max_seq_len: int = 32,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.self_attn = TextEncoderAttention(embed_dim, num_heads, max_seq_len)
        self.layer_norm1 = LayerNormReauthored(embed_dim, eps=layer_norm_eps)
        self.layer_norm2 = LayerNormReauthored(embed_dim, eps=layer_norm_eps)

        self.mlp_fc1 = nn.Conv2d(embed_dim, intermediate_size, kernel_size=1, bias=True)
        self.mlp_act = GELUReauthored()
        self.mlp_fc2 = nn.Conv2d(intermediate_size, embed_dim, kernel_size=1, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp_fc1(hidden_states)
        hidden_states = self.mlp_act(hidden_states)
        hidden_states = self.mlp_fc2(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    @classmethod
    def from_hf_layer(cls, hf_layer: nn.Module, max_seq_len: int = 32) -> "TextEncoderLayer":
        embed_dim = hf_layer.embed_dim
        num_heads = hf_layer.self_attn.num_heads
        intermediate_size = hf_layer.mlp.fc1.out_features
        layer_norm_eps = hf_layer.layer_norm1.eps

        ane_layer = cls(
            embed_dim=embed_dim,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            max_seq_len=max_seq_len,
            layer_norm_eps=layer_norm_eps,
        )

        ane_layer.self_attn = TextEncoderAttention.from_hf_attention(
            hf_layer.self_attn, max_seq_len=max_seq_len
        )

        ane_layer.layer_norm1 = LayerNormReauthored.from_torch_layer_norm(hf_layer.layer_norm1)
        ane_layer.layer_norm2 = LayerNormReauthored.from_torch_layer_norm(hf_layer.layer_norm2)

        ane_layer.mlp_fc1.weight.data = hf_layer.mlp.fc1.weight.data.reshape(
            intermediate_size, embed_dim, 1, 1
        )
        ane_layer.mlp_fc1.bias.data = hf_layer.mlp.fc1.bias.data.clone()
        ane_layer.mlp_fc2.weight.data = hf_layer.mlp.fc2.weight.data.reshape(
            embed_dim, intermediate_size, 1, 1
        )
        ane_layer.mlp_fc2.bias.data = hf_layer.mlp.fc2.bias.data.clone()
        return ane_layer


class TextEncoderReauthored(nn.Module):
    """CLIP text encoder in BC1S, output is ``last_hidden_state``.

    SAM3 takes ``last_hidden_state`` (not the pooled output) and projects
    it externally via ``Sam3Model.text_projection``. The internal
    ``text_projection`` (1024 -> 512) here is included so HF state-dict
    loading is complete, but is not used by SAM3's main forward path.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        max_seq_len: int = 32,
        projection_dim: int = 512,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(max_seq_len).unsqueeze(0),
            persistent=False,
        )

        self.layers = nn.ModuleList(
            [
                TextEncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    intermediate_size=intermediate_size,
                    max_seq_len=max_seq_len,
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_layer_norm = LayerNormReauthored(embed_dim, eps=layer_norm_eps)
        self.text_projection = nn.Conv2d(embed_dim, projection_dim, kernel_size=1, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        position_ids = self.position_ids[:, :seq_len]

        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        hidden_states = token_embeds + position_embeds

        # (B, seq_len, embed_dim) -> BC1S (B, embed_dim, 1, seq_len).
        hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2)

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return self.final_layer_norm(hidden_states)

    @classmethod
    def from_hf_text_encoder(cls, hf_text_encoder: nn.Module) -> "TextEncoderReauthored":
        text_model = hf_text_encoder.text_model
        config = text_model.config

        ane_encoder = cls(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            num_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            max_seq_len=config.max_position_embeddings,
            projection_dim=config.projection_dim,
            layer_norm_eps=config.layer_norm_eps,
        )

        ane_encoder.token_embedding.weight.data = (
            text_model.embeddings.token_embedding.weight.data.clone()
        )
        ane_encoder.position_embedding.weight.data = (
            text_model.embeddings.position_embedding.weight.data.clone()
        )

        for i, hf_layer in enumerate(text_model.encoder.layers):
            ane_encoder.layers[i] = TextEncoderLayer.from_hf_layer(
                hf_layer, max_seq_len=config.max_position_embeddings
            )

        ane_encoder.final_layer_norm = LayerNormReauthored.from_torch_layer_norm(
            text_model.final_layer_norm
        )

        hf_proj = hf_text_encoder.text_projection
        ane_encoder.text_projection.weight.data = hf_proj.weight.data.reshape(
            hf_proj.out_features, hf_proj.in_features, 1, 1
        )
        return ane_encoder
