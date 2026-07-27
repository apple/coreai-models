# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
FLUX.2 component specifications and torch wrappers for Core AI export.

FLUX.2 Klein 4B is a DiT (Diffusion Transformer) that uses:
- Qwen3 text encoder (intermediate hidden states from layers 9, 18, 27)
- 25-block double-stream + single-stream transformer with 4D RoPE
- AutoencoderKLFlux2 VAE with batch normalization

Key difference from SD: the transformer uses pre-computed RoPE embeddings
passed as model inputs (not computed in-graph) to work around a Core AI graph
optimizer bug that corrupts monolithic 25-block transformers when RoPE
frequency ops (arange, outer, pow, repeat_interleave) are in the compiled
graph. Pre-computing RoPE outside the graph avoids this issue.
"""

from typing import Any, cast

import torch

# ---------------------------------------------------------------------------
# RoPE pre-computation (outside the exported graph)
# Core AI graph optimizer corrupts RoPE frequency ops (arange, outer, pow,
# repeat_interleave) in monolithic 25-block transformers.
# Workaround: compute embeddings in Python/Swift and pass as model inputs.
# ---------------------------------------------------------------------------


def _compute_rope_embeddings(
    img_ids: torch.Tensor,
    txt_ids: torch.Tensor,
    axes_dim: list[int],
    theta: float = 2000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute concatenated (cos, sin) RoPE embeddings from position IDs.

    Replicates Flux2PosEmbed.forward() + get_1d_rotary_pos_embed() logic:
      - For each axis: outer(pos, inv_freq) -> cos/sin -> repeat_interleave(2)
      - Concatenate across axes -> [S, sum(axes_dim)]
      - Concatenate text + image -> [txt_S + img_S, D]

    Returns (rotary_emb_cos, rotary_emb_sin) each of shape [txt_S + img_S, D].
    """
    if img_ids.ndim == 3:
        img_ids = img_ids[0]
    if txt_ids.ndim == 3:
        txt_ids = txt_ids[0]

    def _embed_ids(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos_parts = []
        sin_parts = []
        for i, dim in enumerate(axes_dim):
            pos = ids[:, i].float()
            inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
            freqs = torch.outer(pos.double(), inv_freq)
            cos = freqs.cos().repeat_interleave(2, dim=1).float()
            sin = freqs.sin().repeat_interleave(2, dim=1).float()
            cos_parts.append(cos)
            sin_parts.append(sin)
        return torch.cat(cos_parts, dim=-1), torch.cat(sin_parts, dim=-1)

    img_cos, img_sin = _embed_ids(img_ids)
    txt_cos, txt_sin = _embed_ids(txt_ids)

    # HF concatenates text FIRST, then image
    rotary_cos = torch.cat([txt_cos, img_cos], dim=0)
    rotary_sin = torch.cat([txt_sin, img_sin], dim=0)
    return rotary_cos, rotary_sin


# ---------------------------------------------------------------------------
# Torch wrappers
# ---------------------------------------------------------------------------


class Flux2TransformerPrecomputedRoPEWrapper(torch.nn.Module):
    """Wraps Flux2Transformer for export with pre-computed RoPE embeddings.

    Instead of accepting (img_ids, txt_ids) and computing RoPE internally via
    self.pos_embed(), this wrapper accepts (rotary_emb_cos, rotary_emb_sin)
    directly.  This removes all RoPE frequency computation from the traced graph,
    leaving only the simple elementwise rotation in each attention block.
    """

    def __init__(self, transformer: torch.nn.Module) -> None:
        super().__init__()
        self.model = transformer

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance: torch.Tensor,
        rotary_emb_cos: torch.Tensor,
        rotary_emb_sin: torch.Tensor,
    ) -> torch.Tensor:
        model = self.model
        num_txt_tokens = encoder_hidden_states.shape[1]

        # 1. Timestep + guidance embedding
        t = timestep.to(hidden_states.dtype) * 1000
        g = guidance.to(hidden_states.dtype) * 1000
        temb = model.time_guidance_embed(t, g)

        # 2. Modulation parameters
        double_stream_mod_img = model.double_stream_modulation_img(temb)
        double_stream_mod_txt = model.double_stream_modulation_txt(temb)
        single_stream_mod = model.single_stream_modulation(temb)

        # 3. Input projections
        hidden_states = model.x_embedder(hidden_states)
        encoder_hidden_states = model.context_embedder(encoder_hidden_states)

        # 4. RoPE -- PRE-COMPUTED, passed as model inputs (not computed in-graph)
        concat_rotary_emb = (rotary_emb_cos, rotary_emb_sin)

        # 5. Double stream blocks
        for block in model.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_img=double_stream_mod_img,
                temb_mod_txt=double_stream_mod_txt,
                image_rotary_emb=concat_rotary_emb,
            )

        # 6. Concatenate text + image for single stream
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # 7. Single stream blocks
        for block in model.single_transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=None,
                temb_mod=single_stream_mod,
                image_rotary_emb=concat_rotary_emb,
            )

        # 8. Remove text tokens
        hidden_states = hidden_states[:, num_txt_tokens:, ...]

        # 9. Output norm + projection
        hidden_states = model.norm_out(hidden_states, temb)
        return model.proj_out(hidden_states)


class Flux2TextEncoderWrapper(torch.nn.Module):
    """Wraps Qwen3ForCausalLM to extract and concatenate intermediate hidden states.

    FLUX.2 uses hidden states from 3 intermediate layers (default: 9, 18, 27),
    stacked and reshaped from [1, 3, seq_len, 2560] -> [1, seq_len, 7680].
    """

    def __init__(
        self, text_encoder: torch.nn.Module, hidden_states_layers: tuple[int, ...] = (9, 18, 27)
    ) -> None:
        super().__init__()
        self.model = text_encoder
        self.hidden_states_layers = hidden_states_layers

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        stacked = torch.stack([outputs.hidden_states[k] for k in self.hidden_states_layers], dim=1)
        batch_size, num_layers, seq_len, hidden_dim = stacked.shape
        return stacked.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_layers * hidden_dim)


class Flux2VAEDecoderWrapper(torch.nn.Module):
    """Wraps AutoencoderKLFlux2.decode: (latent) -> (image)."""

    def __init__(self, vae: torch.nn.Module) -> None:
        super().__init__()
        self.vae: Any = vae
        # Ensure all parameters + buffers (including BN running stats) share the same dtype
        self.vae = self.vae.to(next(vae.parameters()).dtype)
        from coreai_models.diffusion.components import _patch_nearest_upsample

        _patch_nearest_upsample(self.vae.decoder)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.vae.decode(z).sample)


class Flux2VAEEncoderWrapper(torch.nn.Module):
    """Wraps AutoencoderKLFlux2.encode: (image) -> (latent)."""

    def __init__(self, vae: torch.nn.Module) -> None:
        super().__init__()
        self.vae: Any = vae
        self.vae = self.vae.to(next(vae.parameters()).dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # diffusers encodes img2img reference images with
        # `retrieve_latents(..., sample_mode="argmax")` -> `latent_dist.mode()`,
        # i.e. the distribution MEAN (first `latent_channels` channels), not the
        # raw `parameters` tensor (which is mean concat logvar = 2x channels).
        # Returning `.parameters` would emit 64 channels where the pipeline
        # expects 32, corrupting the img2img latents. `.mode()` is deterministic,
        # so it is also the correct choice for a traced/exported graph.
        return cast(torch.Tensor, self.vae.encode(x).latent_dist.mode())


# ---------------------------------------------------------------------------
# Dummy-input factories
# ---------------------------------------------------------------------------


def _dummy_flux2_transformer_impl(pipe: Any, grid_size: int) -> tuple[torch.Tensor, ...]:
    cfg = pipe.transformer.config
    dtype = next(pipe.transformer.parameters()).dtype
    image_seq_len = grid_size * grid_size
    text_seq_len = 512
    axes_dim = list(cfg.axes_dims_rope)
    theta = cfg.rope_theta if hasattr(cfg, "rope_theta") else 2000.0

    num_rope_axes = len(axes_dim)
    img_ids = torch.zeros(1, image_seq_len, num_rope_axes)
    for h in range(grid_size):
        for w in range(grid_size):
            idx = h * grid_size + w
            img_ids[0, idx, 1] = float(h)
            img_ids[0, idx, 2] = float(w)

    txt_ids = torch.zeros(1, text_seq_len, num_rope_axes)
    for i in range(text_seq_len):
        txt_ids[0, i, 3] = float(i)

    rotary_cos, rotary_sin = _compute_rope_embeddings(img_ids, txt_ids, axes_dim, theta=theta)

    return (
        torch.randn(1, image_seq_len, cfg.in_channels, dtype=dtype),
        torch.randn(1, text_seq_len, cfg.joint_attention_dim, dtype=dtype),
        torch.tensor([0.5], dtype=dtype),
        torch.tensor([1.0], dtype=dtype),
        rotary_cos,
        rotary_sin,
    )


def dummy_flux2_transformer(pipe: Any) -> tuple[torch.Tensor, ...]:
    """1024×1024 (grid=64, seqLen=4096)."""
    return _dummy_flux2_transformer_impl(pipe, grid_size=64)


def dummy_flux2_text_encoder(pipe: Any) -> tuple[torch.Tensor, ...]:
    text_seq_len = 512
    return (
        torch.zeros(1, text_seq_len, dtype=torch.long),  # input_ids
        torch.ones(1, text_seq_len, dtype=torch.long),  # attention_mask
    )


def dummy_flux2_vae_decoder(pipe: Any) -> tuple[torch.Tensor, ...]:
    latent_channels = pipe.vae.config.latent_channels
    sample_size = 128  # 1024 / 8
    dtype = next(pipe.vae.parameters()).dtype
    return (torch.randn(1, latent_channels, sample_size, sample_size, dtype=dtype),)


def dummy_flux2_vae_decoder_half(pipe: Any) -> tuple[torch.Tensor, ...]:
    latent_channels = pipe.vae.config.latent_channels
    sample_size = 64  # 512 / 8
    dtype = next(pipe.vae.parameters()).dtype
    return (torch.randn(1, latent_channels, sample_size, sample_size, dtype=dtype),)


def dummy_flux2_vae_encoder(pipe: Any) -> tuple[torch.Tensor, ...]:
    dtype = next(pipe.vae.parameters()).dtype
    return (torch.randn(1, 3, 1024, 1024, dtype=dtype),)


def dummy_flux2_vae_encoder_half(pipe: Any) -> tuple[torch.Tensor, ...]:
    dtype = next(pipe.vae.parameters()).dtype
    return (torch.randn(1, 3, 512, 512, dtype=dtype),)


def dummy_flux2_transformer_512(pipe: Any) -> tuple[torch.Tensor, ...]:
    """512×512 (grid=32, seqLen=1024)."""
    return _dummy_flux2_transformer_impl(pipe, grid_size=32)
