# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for iOS Qwen2 model parity with HuggingFace."""

import pytest
import torch
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2Config,
    Qwen2DecoderLayer,
    Qwen2RotaryEmbedding,
)
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2ForCausalLM as HFQwen2ForCausalLM,
)

from coreai_models._hf import resolve_rope_theta
from coreai_models.models.ios.qwen2 import (
    Attention,
    Qwen2ForCausalLMForiOS,
    TransformerBlock,
)
from coreai_models.primitives.ios.cache import KVCacheHandler
from coreai_models.primitives.ios.rope import RoPECache as RoPE
from tests._runner_infra.testing_utils import (
    ForCausalLMTestBase,
    assert_close,
    run_compare_coreai,
)


def _make_ne_qwen2_config(
    hidden_size: int = 64,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    num_hidden_layers: int = 1,
    intermediate_size: int = 128,
    vocab_size: int = 100,
    max_position_embeddings: int = 32,
) -> Qwen2Config:
    config = Qwen2Config(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        rope_theta=10000.0,
        rope_parameters={"rope_type": "default"},
    )
    return config


def _make_hf_qwen2_config(
    hidden_size: int = 64,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    num_hidden_layers: int = 1,
    intermediate_size: int = 128,
    vocab_size: int = 100,
    max_position_embeddings: int = 32,
) -> Qwen2Config:
    return Qwen2Config(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
    )


def _make_ne_causal_mask(
    seq_len: int, max_seq_len: int, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Create a causal mask for the iOS SDPA.

    The iOS SDPA expects shape (1, max_seq_len, 1, seq_len) where
    mask[0, j, 0, i] = -inf if position j should not attend to position i.
    """
    mask = torch.zeros(1, max_seq_len, 1, seq_len, dtype=dtype)
    for i in range(seq_len):
        mask[0, i + 1 :, 0, i] = float("-inf")
    return mask


class TestNEQwen2ForCausalLM:
    """Test iOS Qwen2ForCausalLM against HuggingFace reference."""

    def test_forward_parity_multi_token(self):
        """Multi-token prefill: iOS model matches HF logits."""
        seq_len = 4
        max_seq = 32
        hf_config = _make_hf_qwen2_config(max_position_embeddings=max_seq)
        ne_config = _make_ne_qwen2_config(max_position_embeddings=max_seq)

        hf_model = HFQwen2ForCausalLM(hf_config).to(torch.float32).eval()
        sd = dict(hf_model.state_dict())

        ne_model = Qwen2ForCausalLMForiOS(
            ne_config, model_device="cpu", disable_embedding_quantization=True
        )
        ne_model.to(torch.float32).eval()
        ne_model._mutate_state_dict(sd)
        ne_model.load_state_dict(sd, assign=True, strict=True)

        input_ids = torch.randint(0, 100, (1, seq_len))
        position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
        in_step = torch.tensor([0], dtype=torch.int32)
        causal_mask = _make_ne_causal_mask(seq_len, max_seq)
        k_cache, v_cache = KVCacheHandler.get_kv_cache_from_hf(ne_config, dtype=torch.float32)

        with torch.no_grad():
            ne_out = ne_model(input_ids, position_ids, in_step, causal_mask, k_cache, v_cache)
            hf_out = hf_model(input_ids=input_ids, position_ids=position_ids.long())

        ne_logits = ne_out.squeeze(1)
        torch.testing.assert_close(ne_logits, hf_out.logits, atol=1e-5, rtol=1e-5)

    def test_forward_parity_single_token(self):
        """Single-token decode: iOS model matches HF logits."""
        max_seq = 32
        hf_config = _make_hf_qwen2_config(max_position_embeddings=max_seq)
        ne_config = _make_ne_qwen2_config(max_position_embeddings=max_seq)

        hf_model = HFQwen2ForCausalLM(hf_config).to(torch.float32).eval()
        sd = dict(hf_model.state_dict())

        ne_model = Qwen2ForCausalLMForiOS(
            ne_config, model_device="cpu", disable_embedding_quantization=True
        )
        ne_model.to(torch.float32).eval()
        ne_model._mutate_state_dict(sd)
        ne_model.load_state_dict(sd, assign=True, strict=True)

        input_ids = torch.randint(0, 100, (1, 1))
        position_ids = torch.tensor([[0]], dtype=torch.int32)
        in_step = torch.tensor([0], dtype=torch.int32)
        causal_mask = _make_ne_causal_mask(1, max_seq)
        k_cache, v_cache = KVCacheHandler.get_kv_cache_from_hf(ne_config, dtype=torch.float32)

        with torch.no_grad():
            ne_out = ne_model(input_ids, position_ids, in_step, causal_mask, k_cache, v_cache)
            hf_out = hf_model(input_ids=input_ids, position_ids=position_ids.long())

        ne_logits = ne_out.squeeze(1)
        torch.testing.assert_close(ne_logits, hf_out.logits, atol=1e-5, rtol=1e-5)

    def test_forward_parity_two_layers(self):
        """Two-layer model: verify parity scales with depth."""
        seq_len = 4
        max_seq = 32
        hf_config = _make_hf_qwen2_config(num_hidden_layers=2, max_position_embeddings=max_seq)
        ne_config = _make_ne_qwen2_config(num_hidden_layers=2, max_position_embeddings=max_seq)

        hf_model = HFQwen2ForCausalLM(hf_config).to(torch.float32).eval()
        sd = dict(hf_model.state_dict())

        ne_model = Qwen2ForCausalLMForiOS(
            ne_config, model_device="cpu", disable_embedding_quantization=True
        )
        ne_model.to(torch.float32).eval()
        ne_model._mutate_state_dict(sd)
        ne_model.load_state_dict(sd, assign=True, strict=True)

        input_ids = torch.randint(0, 100, (1, seq_len))
        position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
        in_step = torch.tensor([0], dtype=torch.int32)
        causal_mask = _make_ne_causal_mask(seq_len, max_seq)
        k_cache, v_cache = KVCacheHandler.get_kv_cache_from_hf(ne_config, dtype=torch.float32)

        with torch.no_grad():
            ne_out = ne_model(input_ids, position_ids, in_step, causal_mask, k_cache, v_cache)
            hf_out = hf_model(input_ids=input_ids, position_ids=position_ids.long())

        ne_logits = ne_out.squeeze(1)
        torch.testing.assert_close(ne_logits, hf_out.logits, atol=1e-5, rtol=1e-5)

    def test_output_shape(self):
        """Output shape is (batch, 1, seq_len, vocab_size) for iOS layout."""
        max_seq = 32
        ne_config = _make_ne_qwen2_config(max_position_embeddings=max_seq)
        ne_model = Qwen2ForCausalLMForiOS(
            ne_config, model_device="cpu", disable_embedding_quantization=True
        )
        ne_model.to(torch.float32).eval()

        batch, seq_len, vocab = 1, 4, 100
        input_ids = torch.randint(0, vocab, (batch, seq_len))
        position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
        in_step = torch.tensor([0], dtype=torch.int32)
        causal_mask = _make_ne_causal_mask(seq_len, max_seq)
        k_cache, v_cache = KVCacheHandler.get_kv_cache_from_hf(ne_config, dtype=torch.float32)

        with torch.no_grad():
            out = ne_model(input_ids, position_ids, in_step, causal_mask, k_cache, v_cache)

        assert out.shape == (batch, 1, seq_len, vocab)

    def test_mutate_state_dict_adds_conv2d_dims(self):
        """_mutate_state_dict reshapes linear weights to Conv2d (unsqueeze x2)."""
        ne_config = _make_ne_qwen2_config()
        ne_model = Qwen2ForCausalLMForiOS(
            ne_config, model_device="cpu", disable_embedding_quantization=True
        )

        hidden = 64
        n_heads = 4
        n_kv_heads = 2
        head_dim = hidden // n_heads

        sd = {}
        sd["model.embed_tokens.weight"] = torch.randn(100, hidden)
        sd["model.norm.weight"] = torch.randn(hidden)
        sd["lm_head.weight"] = torch.randn(100, hidden)
        sd["model.layers.0.self_attn.q_proj.weight"] = torch.randn(n_heads * head_dim, hidden)
        sd["model.layers.0.self_attn.q_proj.bias"] = torch.randn(n_heads * head_dim)
        sd["model.layers.0.self_attn.k_proj.weight"] = torch.randn(n_kv_heads * head_dim, hidden)
        sd["model.layers.0.self_attn.k_proj.bias"] = torch.randn(n_kv_heads * head_dim)
        sd["model.layers.0.self_attn.v_proj.weight"] = torch.randn(n_kv_heads * head_dim, hidden)
        sd["model.layers.0.self_attn.v_proj.bias"] = torch.randn(n_kv_heads * head_dim)
        sd["model.layers.0.self_attn.o_proj.weight"] = torch.randn(hidden, hidden)
        sd["model.layers.0.mlp.gate_proj.weight"] = torch.randn(128, hidden)
        sd["model.layers.0.mlp.up_proj.weight"] = torch.randn(128, hidden)
        sd["model.layers.0.mlp.down_proj.weight"] = torch.randn(hidden, 128)
        sd["model.layers.0.input_layernorm.weight"] = torch.randn(hidden)
        sd["model.layers.0.post_attention_layernorm.weight"] = torch.randn(hidden)

        ne_model._mutate_state_dict(sd)

        # Attention weights should be unsqueezed to 4D (Conv2d format)
        q_key = "extend.model.layers.0.self_attn.q_proj.weight"
        assert q_key in sd
        assert sd[q_key].dim() == 4

        # MLP weights should also be 4D
        gate_key = "extend.model.layers.0.mlp.gate_proj.weight"
        assert gate_key in sd
        assert sd[gate_key].dim() == 4

        # Embedding table should exist under load_embeddings
        assert "load_embeddings.embedding_table" in sd

    def test_attention_bias_false_parity(self):
        """SmolLM2-style: attention_bias=False + tie_word_embeddings=True on iOS."""
        seq_len = 4
        max_seq = 32
        hf_config = Qwen2Config(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_hidden_layers=2,
            intermediate_size=128,
            vocab_size=100,
            max_position_embeddings=max_seq,
            attention_bias=False,
            tie_word_embeddings=True,
            rope_theta=130000.0,
            rope_parameters={"rope_type": "default"},
        )

        ne_config = Qwen2Config(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_hidden_layers=2,
            intermediate_size=128,
            vocab_size=100,
            max_position_embeddings=max_seq,
            attention_bias=False,
            tie_word_embeddings=True,
            rope_theta=130000.0,
            rope_parameters={"rope_type": "default"},
        )

        hf_model = HFQwen2ForCausalLM(hf_config).to(torch.float32).eval()
        sd = dict(hf_model.state_dict())

        ne_model = Qwen2ForCausalLMForiOS(
            ne_config, model_device="cpu", disable_embedding_quantization=True
        )
        ne_model.to(torch.float32).eval()
        ne_model._mutate_state_dict(sd)
        ne_model.load_state_dict(sd, assign=True, strict=True)

        input_ids = torch.randint(0, 100, (1, seq_len))
        position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
        in_step = torch.tensor([0], dtype=torch.int32)
        causal_mask = _make_ne_causal_mask(seq_len, max_seq)
        k_cache, v_cache = KVCacheHandler.get_kv_cache_from_hf(ne_config, dtype=torch.float32)

        with torch.no_grad():
            ne_out = ne_model(input_ids, position_ids, in_step, causal_mask, k_cache, v_cache)
            hf_out = hf_model(input_ids=input_ids, position_ids=position_ids.long())

        torch.testing.assert_close(ne_out[:, 0, :, :], hf_out.logits, atol=1e-5, rtol=1e-5)


@pytest.fixture
def base_qwen2_config():
    """Base Qwen2Config with common test parameters."""
    config = Qwen2Config(
        rms_norm_eps=9.87,
        rope_theta=1e5,
        hidden_size=4,
        head_dim=16,
        intermediate_size=6,
        num_attention_heads=8,
        num_key_value_heads=4,
        num_hidden_layers=2,
        max_position_embeddings=512,
    )
    config._attn_implementation = "sdpa"
    return config


@pytest.mark.parametrize(
    "heads,layer_idx",
    [
        ((1, 1), 0),
        ((8, 8), 0),
        ((8, 4), 1),
    ],
)
class Qwen2iOSComponentTestBase:
    """Base class with common utilities for testing iOS Qwen2 components."""

    @staticmethod
    def setup_attention_weights(our_attention, hf_attention, config):
        """Setup attention weights. iOS uses Conv2d, so weights need reshaping."""
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        head_dim = config.head_dim
        hidden_size = config.hidden_size

        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        q_weight = torch.randn(q_size, hidden_size)
        q_bias = torch.randn(q_size)
        k_weight = torch.randn(k_size, hidden_size)
        k_bias = torch.randn(k_size)
        v_weight = torch.randn(v_size, hidden_size)
        v_bias = torch.randn(v_size)

        our_attention.q_proj.weight = torch.nn.Parameter(q_weight.unsqueeze(-1).unsqueeze(-1))
        our_attention.q_proj.bias = torch.nn.Parameter(q_bias.clone())

        our_attention.k_proj.weight = torch.nn.Parameter(k_weight.unsqueeze(-1).unsqueeze(-1))
        our_attention.k_proj.bias = torch.nn.Parameter(k_bias.clone())

        our_attention.v_proj.weight = torch.nn.Parameter(v_weight.unsqueeze(-1).unsqueeze(-1))
        our_attention.v_proj.bias = torch.nn.Parameter(v_bias.clone())

        hf_attention.q_proj.weight = torch.nn.Parameter(q_weight.clone())
        hf_attention.q_proj.bias = torch.nn.Parameter(q_bias.clone())

        hf_attention.k_proj.weight = torch.nn.Parameter(k_weight.clone())
        hf_attention.k_proj.bias = torch.nn.Parameter(k_bias.clone())

        hf_attention.v_proj.weight = torch.nn.Parameter(v_weight.clone())
        hf_attention.v_proj.bias = torch.nn.Parameter(v_bias.clone())

        o_proj_weight = torch.randn(hidden_size, num_attention_heads * head_dim)
        our_attention.o_proj.weight = torch.nn.Parameter(o_proj_weight.unsqueeze(-1).unsqueeze(-1))
        hf_attention.o_proj.weight = torch.nn.Parameter(o_proj_weight.clone())

    @staticmethod
    def setup_mlp_weights(our_mlp, hf_mlp, config):
        """Setup MLP weights. iOS uses Conv2d, so weights need reshaping."""
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        gate_weight = torch.randn(intermediate_size, hidden_size)
        our_mlp.gate_proj.weight = torch.nn.Parameter(gate_weight.unsqueeze(-1).unsqueeze(-1))
        hf_mlp.gate_proj.weight = torch.nn.Parameter(gate_weight.clone())

        up_weight = torch.randn(intermediate_size, hidden_size)
        our_mlp.up_proj.weight = torch.nn.Parameter(up_weight.unsqueeze(-1).unsqueeze(-1))
        hf_mlp.up_proj.weight = torch.nn.Parameter(up_weight.clone())

        down_weight = torch.randn(hidden_size, intermediate_size)
        our_mlp.down_proj.weight = torch.nn.Parameter(down_weight.unsqueeze(-1).unsqueeze(-1))
        hf_mlp.down_proj.weight = torch.nn.Parameter(down_weight.clone())

    @staticmethod
    def setup_layernorm_weights(our_block, hf_block, config):
        """Setup layernorm weights for transformer blocks."""
        hidden_size = config.hidden_size

        input_ln_weight = torch.randn(hidden_size)
        our_block.input_layernorm.weight = torch.nn.Parameter(input_ln_weight.clone())
        hf_block.input_layernorm.weight = torch.nn.Parameter(input_ln_weight.clone())

        post_attn_ln_weight = torch.randn(hidden_size)
        our_block.post_attention_layernorm.weight = torch.nn.Parameter(post_attn_ln_weight.clone())
        hf_block.post_attention_layernorm.weight = torch.nn.Parameter(post_attn_ln_weight.clone())

    @staticmethod
    def create_test_inputs(
        config,
    ):
        """Create standard test inputs for iOS tests (4D tensors)."""
        hidden_size = config.hidden_size
        batch_size = 2
        seq_len = 10

        # we set the offset to 0
        offset = 0
        x = torch.randn(batch_size, seq_len, 1, hidden_size)
        position_ids = offset + torch.arange(seq_len, dtype=torch.int32).unsqueeze(0).expand(
            batch_size, -1
        )
        in_step = torch.tensor([0], dtype=torch.int32)

        # we not passing the cache, we need to set the shape of casual mask in such
        causal_mask = torch.zeros(
            (1, seq_len, 1, seq_len),
            dtype=torch.float32,
        )
        causal_mask[:, :offset, :, :] = float("-inf")
        for i in range(seq_len):
            causal_mask[:, offset + i + 1 :, :, i] = float("-inf")

        return x, position_ids, in_step, causal_mask


class TestQwen2iOSAttention(Qwen2iOSComponentTestBase):
    def get_model_asset(
        self,
        base_qwen2_config,
        heads: tuple[int, int],
        layer_idx: int,
        precision: torch.dtype = torch.float32,
    ) -> tuple[torch.nn.Module, tuple, torch.Tensor]:
        config = base_qwen2_config
        num_attention_heads, num_key_value_heads = heads
        config.num_attention_heads = num_attention_heads
        config.num_key_value_heads = num_key_value_heads

        our_attention = Attention(config=config, layer_idx=layer_idx)
        hf_attention = Qwen2Attention(config=config, layer_idx=layer_idx)
        self.setup_attention_weights(our_attention, hf_attention, config)

        x, position_ids, in_step, causal_mask = self.create_test_inputs(config)
        hf_rotary = Qwen2RotaryEmbedding(config)
        hf_attention_mask = causal_mask.permute(0, 2, 3, 1)

        x = x.to(precision)
        causal_mask = causal_mask.to(precision)
        our_attention = our_attention.to(precision)
        hf_attention = hf_attention.to(precision)
        hf_rotary = hf_rotary.to(precision)
        hf_attention_mask = hf_attention_mask.to(precision)

        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        our_rope = RoPE(
            head_dim,
            config.max_position_embeddings,
            resolve_rope_theta(config),
        ).to(precision)
        rope_cos, rope_sin = our_rope.gather_cos_sin(position_ids)

        x_hf = x.squeeze(2)
        cos_hf, sin_hf = hf_rotary(x_hf, position_ids)
        hf_output = hf_attention(
            hidden_states=x_hf,
            attention_mask=hf_attention_mask,
            position_embeddings=(cos_hf, sin_hf),
        )[0]
        hf_output = hf_output.unsqueeze(2)

        return (
            our_attention,
            (x, rope_cos, rope_sin, in_step, causal_mask),
            hf_output,
        )

    @pytest.mark.parametrize(
        "precision",
        [
            torch.float32,
            torch.float16,
        ],
    )
    def test_hf(
        self,
        base_qwen2_config,
        heads: tuple[int, int],
        layer_idx: int,
        precision: torch.dtype,
    ) -> None:
        model, inputs, expected_output = self.get_model_asset(base_qwen2_config, heads, layer_idx)
        assert_close(model(*inputs), expected_output, rtol=1e-1)

    def test_coreai(
        self,
        base_qwen2_config,
        heads: tuple[int, int],
        layer_idx: int,
    ) -> None:
        """Test Core AI compilation and execution."""
        model, inputs, _ = self.get_model_asset(base_qwen2_config, heads, layer_idx)
        run_compare_coreai(
            model=model,
            inputs=inputs,
        )


class TestQwen2iOSTransformerBlock(Qwen2iOSComponentTestBase):
    def get_model_asset(
        self,
        base_qwen2_config,
        heads: tuple[int, int],
        layer_idx: int,
        precision: torch.dtype = torch.float32,
    ) -> tuple[torch.nn.Module, tuple, torch.Tensor]:
        config = base_qwen2_config
        num_attention_heads, num_key_value_heads = heads
        config.num_attention_heads = num_attention_heads
        config.num_key_value_heads = num_key_value_heads

        our_block = TransformerBlock(config=config, layer_idx=layer_idx)
        hf_block = Qwen2DecoderLayer(config=config, layer_idx=layer_idx)
        self.setup_attention_weights(
            our_block.self_attn,
            hf_block.self_attn,
            config,
        )
        self.setup_mlp_weights(our_block.mlp, hf_block.mlp, config)
        self.setup_layernorm_weights(our_block, hf_block, config)

        x, position_ids, in_step, causal_mask = self.create_test_inputs(config)

        hf_rotary = Qwen2RotaryEmbedding(config)
        hf_attention_mask = causal_mask.permute(0, 2, 3, 1)

        x = x.to(precision)
        causal_mask = causal_mask.to(precision)
        our_block = our_block.to(precision)
        hf_block = hf_block.to(precision)
        hf_rotary = hf_rotary.to(precision)
        hf_attention_mask = hf_attention_mask.to(precision)

        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        our_rope = RoPE(
            head_dim,
            config.max_position_embeddings,
            resolve_rope_theta(config),
        ).to(precision)
        rope_cos, rope_sin = our_rope.gather_cos_sin(position_ids)

        x_hf = x.squeeze(2)
        cos_hf, sin_hf = hf_rotary(x_hf, position_ids)
        hf_output = hf_block(
            hidden_states=x_hf,
            attention_mask=hf_attention_mask,
            position_embeddings=(cos_hf, sin_hf),
            cache_position=position_ids,
        )
        hf_output = hf_output.unsqueeze(2)

        return our_block, (x, rope_cos, rope_sin, in_step, causal_mask), hf_output

    @pytest.mark.parametrize("precision", [torch.float32, torch.float16])
    def test_hf(
        self,
        base_qwen2_config,
        heads: tuple[int, int],
        layer_idx: int,
        precision: torch.dtype,
    ) -> None:
        model, inputs, expected_output = self.get_model_asset(
            base_qwen2_config, heads, layer_idx, precision
        )
        atol = 5e-2 if precision == torch.float16 else 1e-4
        assert_close(model(*inputs), expected_output, rtol=3e-2, atol=atol)

    def test_coreai(
        self,
        base_qwen2_config,
        heads: tuple[int, int],
        layer_idx: int,
    ) -> None:
        """Test Core AI compilation and execution."""
        model, inputs, _ = self.get_model_asset(base_qwen2_config, heads, layer_idx)
        run_compare_coreai(
            model=model,
            inputs=inputs,
        )


@pytest.mark.slow
class TestQwen2iOSForCausalLM(ForCausalLMTestBase):
    _toy_model_id = "yujiepan/qwen2.5-tiny-random"
    _model_class = Qwen2ForCausalLMForiOS
    _test_kv_cache = False
