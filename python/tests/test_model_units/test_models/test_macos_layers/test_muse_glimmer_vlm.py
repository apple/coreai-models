# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for Muse Glimmer VLM components.

Covers the VLM-specific classes added for vision-language support:
- MuseGlimmerForCausalLMEmbeddings (text decoder, inputs_embeds variant)
- MuseGlimmerVisionModel (ViT-G/14 vision encoder + projector)
- EmbedTokensNormed (normalized token embedding lookup)
- build_reference_inputs (inputs_embeds, not input_ids)

All tests use small configs with random weights — no HF downloads.
"""

from types import SimpleNamespace

import torch

from coreai_models.models.macos.muse_glimmer import MuseGlimmerForCausalLMEmbeddings
from coreai_models.models.macos.muse_glimmer_vision import MuseGlimmerVisionModel
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.vlm.export import EmbedTokensNormed


def _make_glimmer_config(**overrides) -> SimpleNamespace:
    """Minimal Muse Glimmer config for fast unit tests (tiny dims)."""
    defaults = dict(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        intermediate_size=128,
        vocab_size=200,
        max_position_embeddings=32,
        head_dim=16,
        attention_bias=False,
        hidden_activation="silu",
        rms_norm_eps=1e-5,
        sliding_window=8,
        final_logit_softcapping=20.0,
        tie_word_embeddings=False,
        output_multiplier=0.196,
        post_norm_eps=1e-8,
        qk_scale_factor=3.87,
        layer_types=["sliding_attention", "full_attention"],
        layer_rope_theta=[500000, 0],
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestMuseGlimmerForCausalLMEmbeddings:
    """Test the inputs_embeds variant of the Muse Glimmer text decoder."""

    def test_forward_produces_finite_output(self):
        """Forward with random inputs_embeds should produce finite logits."""
        config = _make_glimmer_config()
        model = MuseGlimmerForCausalLMEmbeddings(config, model_device="cpu")
        model.to(torch.float32).eval()

        seq_len = 6
        inputs_embeds = torch.randn(1, seq_len, config.hidden_size)
        position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=torch.float32)

        with torch.no_grad():
            logits = model(inputs_embeds, position_ids, k_cache, v_cache)

        assert logits.shape == (1, seq_len, config.vocab_size)
        assert torch.isfinite(logits).all()

    def test_no_embed_tokens(self):
        """The embeddings variant should not have embed_tokens (that lives in embed.aimodel)."""
        config = _make_glimmer_config()
        model = MuseGlimmerForCausalLMEmbeddings(config, model_device="cpu")
        assert not hasattr(model.model, "embed_tokens")

    def test_logit_softcapping(self):
        """Logits should be bounded by the softcap value."""
        config = _make_glimmer_config(final_logit_softcapping=20.0)
        model = MuseGlimmerForCausalLMEmbeddings(config, model_device="cpu")
        model.to(torch.float32).eval()

        inputs_embeds = torch.randn(1, 4, config.hidden_size)
        position_ids = torch.arange(4, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=torch.float32)

        with torch.no_grad():
            logits = model(inputs_embeds, position_ids, k_cache, v_cache)

        assert logits.abs().max() <= 20.0

    def test_mutate_state_dict_drops_embed_tokens(self):
        """_mutate_state_dict should remove embed_tokens keys (they belong in embed.aimodel)."""
        config = _make_glimmer_config(num_hidden_layers=1, tie_word_embeddings=False)
        model = MuseGlimmerForCausalLMEmbeddings(config, model_device="cpu")

        sd = {
            "model.language_model.embed_tokens.weight": torch.randn(200, 64),
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
            "lm_head.weight": torch.randn(200, 64),
        }
        model._mutate_state_dict(sd)

        # embed_tokens should be deleted (not tied)
        assert all("embed_tokens" not in k for k in sd)
        assert "lm_head.weight" in sd

    def test_mutate_state_dict_promotes_embed_when_tied(self):
        """With tie_word_embeddings, embed_tokens should become lm_head.weight."""
        config = _make_glimmer_config(num_hidden_layers=1, tie_word_embeddings=True)
        model = MuseGlimmerForCausalLMEmbeddings(config, model_device="cpu")

        sd = {
            "model.language_model.embed_tokens.weight": torch.randn(200, 64),
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
        }
        model._mutate_state_dict(sd)

        assert "lm_head.weight" in sd
        assert all("embed_tokens" not in k for k in sd)


class TestMuseGlimmerVisionModel:
    """Test the ViT-G/14 vision encoder + multi-modal projector."""

    def _make_tiny_vision_model(self) -> MuseGlimmerVisionModel:
        """Create a small vision model with random weights for testing."""
        return MuseGlimmerVisionModel(
            # Tiny encoder config
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            patch_size=14,
            patch_temporal=2,
            merge_size=2,
            pos_emb_height=4,
            pos_emb_width=4,
            layer_norm_eps=1e-5,
            rope_theta=10000.0,
            # Tiny projector config
            out_hidden_size=32 * 4,  # hidden_size * merge_size^2 = 32 * 4
            projector_hidden_size=32,
            text_hidden_size=64,
            rms_norm_eps=1e-5,
        ).eval()

    def test_forward_produces_finite_output(self):
        """Forward with random pixel_values should produce finite image features."""
        model = self._make_tiny_vision_model()
        # Image size must match the grid: pos_emb_height * patch_size = 4 * 14 = 56
        image_size = 4 * 14  # 56x56
        pixel_values = torch.randn(1, 3, image_size, image_size)

        with torch.no_grad():
            out = model(pixel_values)

        # After merge: (4/2) * (4/2) = 4 merged tokens
        expected_tokens = (4 // 2) * (4 // 2)
        assert out.shape == (1, expected_tokens, 64)  # [1, N, text_hidden_size]
        assert torch.isfinite(out).all()

    def test_patchify_shape(self):
        """Patchify should produce the right number of patches."""
        model = self._make_tiny_vision_model()
        image_size = 4 * 14  # 56
        pixel_values = torch.randn(1, 3, image_size, image_size)

        patches, grid_h, grid_w = model.patchify(pixel_values)

        assert grid_h == 4
        assert grid_w == 4
        assert patches.shape[0] == 4 * 4  # 16 patches
        # patch_dim = patch_temporal * 3 * patch_size^2 = 2 * 3 * 14 * 14 = 1176
        assert patches.shape[1] == 2 * 3 * 14 * 14

    def test_output_is_normalized(self):
        """Vision output should be RMSNorm'd (perception norm) with controlled magnitude."""
        model = self._make_tiny_vision_model()
        image_size = 4 * 14
        pixel_values = torch.randn(1, 3, image_size, image_size)

        with torch.no_grad():
            out = model(pixel_values)

        # After weight-less RMSNorm, the RMS of each feature vector should be ~1.0
        rms = out.float().pow(2).mean(-1).sqrt()
        # Allow some tolerance since the norm is approximate with eps
        assert (rms - 1.0).abs().max() < 0.1


class TestEmbedTokensNormed:
    """Test the normalized token embedding lookup module."""

    def test_output_shape(self):
        """Output should match [batch, seq_len, hidden_size]."""
        vocab_size, hidden_size = 100, 64
        weight = torch.randn(vocab_size, hidden_size, dtype=torch.float16)
        module = EmbedTokensNormed(weight, eps=1e-5).eval()

        input_ids = torch.randint(0, vocab_size, (1, 8), dtype=torch.int32)
        with torch.no_grad():
            out = module(input_ids)

        assert out.shape == (1, 8, hidden_size)

    def test_output_is_normalized(self):
        """After weight-less RMSNorm, the RMS of each embedding should be ~1.0."""
        vocab_size, hidden_size = 100, 64
        weight = torch.randn(vocab_size, hidden_size, dtype=torch.float32)
        module = EmbedTokensNormed(weight, eps=1e-5).eval()

        input_ids = torch.randint(0, vocab_size, (1, 16), dtype=torch.int32)
        with torch.no_grad():
            out = module(input_ids)

        # RMS should be close to 1.0 for each token
        rms = out.pow(2).mean(-1).sqrt()
        assert (rms - 1.0).abs().max() < 0.01, f"RMS range: [{rms.min():.4f}, {rms.max():.4f}]"

    def test_output_is_finite(self):
        """Output should contain no NaN or Inf values."""
        vocab_size, hidden_size = 100, 64
        weight = torch.randn(vocab_size, hidden_size, dtype=torch.float16)
        module = EmbedTokensNormed(weight, eps=1e-5).eval()

        input_ids = torch.randint(0, vocab_size, (1, 8), dtype=torch.int32)
        with torch.no_grad():
            out = module(input_ids)

        assert torch.isfinite(out).all()


class TestBuildReferenceInputs:
    """Test that build_reference_inputs returns the correct structure."""

    def test_returns_inputs_embeds_not_input_ids(self):
        """VLM decoder should use inputs_embeds (not input_ids) in reference inputs."""
        config = _make_glimmer_config()
        model = MuseGlimmerForCausalLMEmbeddings(config, model_device="cpu")

        spec = SimpleNamespace(cache_seq_len=32, query_len=4, offset=4)
        ref = model.build_reference_inputs(config, torch.float16, spec)

        # Should be keyed by graph name "main"
        assert "main" in ref
        inputs = ref["main"]

        assert "inputs_embeds" in inputs
        assert "input_ids" not in inputs
        assert "position_ids" in inputs
        assert "k_cache" in inputs
        assert "v_cache" in inputs

    def test_inputs_embeds_shape(self):
        """inputs_embeds should be [1, query_len, hidden_size]."""
        config = _make_glimmer_config()
        model = MuseGlimmerForCausalLMEmbeddings(config, model_device="cpu")

        spec = SimpleNamespace(cache_seq_len=32, query_len=8, offset=4)
        ref = model.build_reference_inputs(config, torch.float16, spec)

        embeds = ref["main"]["inputs_embeds"]
        assert embeds.shape == (1, 8, config.hidden_size)
        assert embeds.dtype == torch.float16

    def test_cache_shapes(self):
        """KV caches should have the right shape [n_layers, 1, n_kv, max_ctx, head_dim]."""
        config = _make_glimmer_config()
        model = MuseGlimmerForCausalLMEmbeddings(config, model_device="cpu")

        spec = SimpleNamespace(cache_seq_len=32, query_len=4, offset=4)
        ref = model.build_reference_inputs(config, torch.float16, spec)

        k = ref["main"]["k_cache"]
        v = ref["main"]["v_cache"]
        expected = (
            config.num_hidden_layers,
            1,
            config.num_key_value_heads,
            32,  # cache_seq_len
            config.head_dim,
        )
        assert k.shape == expected
        assert v.shape == expected
