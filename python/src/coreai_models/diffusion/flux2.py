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

The transformer computes RoPE in-graph from position IDs (img_ids, txt_ids),
matching upstream diffusers. Position IDs are cheap to build and depend only on
grid geometry, so the exported graph owns the frequency computation.
"""

from functools import lru_cache
from typing import Any, cast

import torch

# ---------------------------------------------------------------------------
# Torch wrappers
# ---------------------------------------------------------------------------


def _patch_attn_split_for_dynamic(transformer: torch.nn.Module) -> int:
    """Make the joint-attention split convertible under a symbolic image sequence.

    Upstream splits the attention output back into text/image with
    ``split_with_sizes([txt_len, total - txt_len])``. With a dynamic image
    sequence the second extent is a SymInt, and the Core AI converter rejects it
    (``TypeError: int() argument ... not 'Node'``) after ``split_with_sizes``
    survives decomposition. The text length is static, so two constant-offset
    slices are bit-identical and convert cleanly.

    Swaps the processor per module via ``set_processor`` rather than patching the
    class, so this is idempotent and leaves the single-stream blocks
    (``Flux2ParallelSelfAttnProcessor``, which splits on the static feature dim)
    untouched. Returns the number of modules patched.
    """
    from diffusers.models.transformers.transformer_flux2 import Flux2AttnProcessor

    replacement_cls = _slice_split_processor_cls()
    patched = 0
    for module in transformer.modules():
        processor = getattr(module, "processor", None)
        if processor is None or not isinstance(processor, Flux2AttnProcessor):
            continue
        if isinstance(processor, replacement_cls):
            continue
        cast(Any, module).set_processor(replacement_cls())
        patched += 1
    return patched


@lru_cache(maxsize=1)
def _slice_split_processor_cls() -> type:
    """Build the patched processor class once.

    Memoized so every call returns the *same* class object — otherwise the
    ``isinstance`` guard in :func:`_patch_attn_split_for_dynamic` could never
    match a previously-patched module and the patch would not be idempotent.
    Deferred so importing this module does not require diffusers.
    """
    from diffusers.models.attention_dispatch import dispatch_attention_fn
    from diffusers.models.embeddings import apply_rotary_emb
    from diffusers.models.transformers.transformer_flux2 import (
        Flux2AttnProcessor,
        _get_qkv_projections,
    )

    class SliceSplitAttnProcessor(Flux2AttnProcessor):  # type: ignore[misc, valid-type]
        """``Flux2AttnProcessor`` with the text/image split at a constant offset."""

        def __call__(  # type: ignore[no-untyped-def]
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            image_rotary_emb=None,
        ):
            query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
                attn, hidden_states, encoder_hidden_states
            )
            query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
            key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
            value = value.unflatten(-1, (attn.heads, -1))

            txt_len = None
            if attn.added_kv_proj_dim is not None:
                encoder_query = attn.norm_added_q(encoder_query.unflatten(-1, (attn.heads, -1)))
                encoder_key = attn.norm_added_k(encoder_key.unflatten(-1, (attn.heads, -1)))
                encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))
                txt_len = encoder_query.shape[1]
                query = torch.cat([encoder_query, query], dim=1)
                key = torch.cat([encoder_key, key], dim=1)
                value = torch.cat([encoder_value, value], dim=1)

            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
            hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

            if encoder_hidden_states is not None:
                # Constant offsets instead of split_with_sizes — see the caller's docstring.
                encoder_hidden_states = attn.to_add_out(hidden_states[:, :txt_len])
                hidden_states = hidden_states[:, txt_len:]

            hidden_states = attn.to_out[1](attn.to_out[0](hidden_states))
            if encoder_hidden_states is not None:
                return hidden_states, encoder_hidden_states
            return hidden_states

    return SliceSplitAttnProcessor


class Flux2TransformerWrapper(torch.nn.Module):
    """Wraps Flux2Transformer2DModel for export with in-graph RoPE.

    Takes position IDs and lets the model compute rotary embeddings internally via
    self.pos_embed(), so the exported graph matches upstream diffusers.

    ``dynamic=True`` additionally patches the joint-attention split so the graph
    converts with a symbolic image sequence length. It is bit-identical, but is
    opt-in so static exports keep their existing graph.
    """

    def __init__(self, transformer: torch.nn.Module, dynamic: bool = False) -> None:
        super().__init__()
        self.model = transformer
        if dynamic:
            _patch_attn_split_for_dynamic(transformer)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
    ) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self.model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                guidance=guidance,
                img_ids=img_ids,
                txt_ids=txt_ids,
            ).sample,
        )


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
    """Wraps AutoencoderKLFlux2.decode: (latent) -> (image).

    ``dynamic=True`` skips ``_patch_nearest_upsample``. That patch replaces
    ``F.interpolate(nearest)`` with ``repeat_interleave(2, -1).repeat_interleave(2, -2)``
    to keep the op off BNNS/CPU, but under a symbolic latent size the
    ``repeat_interleave`` lowers to an expand with a zero stride, and region
    formation rejects it outright:

        'mpsx.ane' op Smallest stride was: 0 expected 4 for
        type 'memref<?x512x128x?x?xf16>'  ->  Pass failed: ANERegionFormationPass

    Stock ``F.interpolate`` exports and runs correctly at every latent size, so the
    dynamic path takes it. Loaded with ``expectFrequentReshapes`` (as the Swift pipeline
    does for dynamic bundles) it decodes as fast as the patched static decoder.
    """

    def __init__(self, vae: torch.nn.Module, dynamic: bool = False) -> None:
        super().__init__()
        self.vae: Any = vae
        # Ensure all parameters + buffers (including BN running stats) share the same dtype
        self.vae = self.vae.to(next(vae.parameters()).dtype)
        if not dynamic:
            from coreai_models.diffusion.components import _patch_nearest_upsample

            # Mutates the shared pipe.vae.decoder in place and is never undone, so a
            # dynamic decoder wrapped later in the same process would inherit it.
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

# Reference i carries T=10*(i+1) on RoPE axis 0 (diffusers `_prepare_image_ids`), so
# in-graph RoPE separates each reference from the noise grid and from each other where
# H/W coincide. Mirrors FlowTransformerPipeline.referenceTokenTimeOffset.
REFERENCE_TOKEN_TIME_OFFSET = 10.0


def _grid_position_ids(side: int, num_axes: int, time_offset: float = 0.0) -> torch.Tensor:
    """`[side*side, num_axes]` position IDs as [T, H, W, L], row-major over H then W."""
    coords = torch.arange(side, dtype=torch.float32)
    grid_h, grid_w = torch.meshgrid(coords, coords, indexing="ij")
    ids = torch.zeros(side * side, num_axes)
    ids[:, 0] = time_offset
    ids[:, 1] = grid_h.reshape(-1)
    ids[:, 2] = grid_w.reshape(-1)
    return ids


def _text_position_ids(text_seq_len: int, num_axes: int) -> torch.Tensor:
    """`[1, text_seq_len, num_axes]` — sequence index on the last axis, spatial unused."""
    ids = torch.zeros(1, text_seq_len, num_axes)
    ids[0, :, num_axes - 1] = torch.arange(text_seq_len, dtype=torch.float32)
    return ids


def _dummy_flux2_transformer_impl(pipe: Any, grid_size: int) -> tuple[torch.Tensor, ...]:
    cfg = pipe.transformer.config
    dtype = next(pipe.transformer.parameters()).dtype
    image_seq_len = grid_size * grid_size
    text_seq_len = 512
    axes_dim = list(cfg.axes_dims_rope)

    # Position IDs per token: [T, H, W, L]. Image tokens carry the spatial grid on
    # axes 1/2; text tokens carry the sequence index on the last axis.
    num_rope_axes = len(axes_dim)
    img_ids = _grid_position_ids(grid_size, num_rope_axes).unsqueeze(0)
    txt_ids = _text_position_ids(text_seq_len, num_rope_axes)

    return (
        torch.randn(1, image_seq_len, cfg.in_channels, dtype=dtype),
        torch.randn(1, text_seq_len, cfg.joint_attention_dim, dtype=dtype),
        torch.tensor([0.5], dtype=dtype),
        torch.tensor([1.0], dtype=dtype),
        img_ids,
        txt_ids,
    )


def dummy_flux2_transformer(pipe: Any) -> tuple[torch.Tensor, ...]:
    """1024×1024 (grid=64, seqLen=4096)."""
    return _dummy_flux2_transformer_impl(pipe, grid_size=64)


def dummy_flux2_transformer_quant_trace(pipe: Any) -> tuple[torch.Tensor, ...]:
    """Small trace for the weight quantizer's shape-discovery forward.

    Identical to dummy_flux2_transformer but with smaller grid size for faster
    export. We don't need the full grid size for quantization trace.
    """
    return _dummy_flux2_transformer_impl(pipe, grid_size=8)


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


# ---------------------------------------------------------------------------
# img2img dummy factories — concatenated noise + reference tokens
# ---------------------------------------------------------------------------


def _dummy_flux2_transformer_img2img(
    pipe: Any, noise_grid: int, ref_grids: tuple[int, ...]
) -> tuple[torch.Tensor, ...]:
    """Build dummy inputs for img2img transformer with concatenated reference tokens.

    The img2img approach concatenates noise tokens + reference tokens along the
    sequence dimension. The transformer processes the full sequence and we slice
    off the noise predictions afterward.

    Args:
        pipe: HF pipeline (for config access).
        noise_grid: Grid size for noise tokens (64 = 1024×1024).
        ref_grids: Grid size of each reference image, in order (16/32/64 =
            quarter/half/full at 1024).
    """
    cfg = pipe.transformer.config
    dtype = next(pipe.transformer.parameters()).dtype
    total_img_seq = noise_grid * noise_grid + sum(g * g for g in ref_grids)
    text_seq = 512
    num_rope_axes = len(cfg.axes_dims_rope)

    # Position IDs: noise (T=0), then reference i at T=10*(i+1)
    img_ids = torch.cat(
        [_grid_position_ids(noise_grid, num_rope_axes)]
        + [
            _grid_position_ids(g, num_rope_axes, time_offset=REFERENCE_TOKEN_TIME_OFFSET * (i + 1))
            for i, g in enumerate(ref_grids)
        ]
    ).unsqueeze(0)
    txt_ids = _text_position_ids(text_seq, num_rope_axes)

    return (
        torch.randn(1, total_img_seq, cfg.in_channels, dtype=dtype),
        torch.randn(1, text_seq, cfg.joint_attention_dim, dtype=dtype),
        torch.tensor([0.5], dtype=dtype),
        torch.tensor([1.0], dtype=dtype),
        img_ids,
        txt_ids,
    )


def dummy_flux2_transformer_img2img_quarter(pipe: Any) -> tuple[torch.Tensor, ...]:
    """img2img quarter (1024×1024): 4096 noise + 256 reference = 4352 img tokens."""
    return _dummy_flux2_transformer_img2img(pipe, noise_grid=64, ref_grids=(16,))


def dummy_flux2_transformer_img2img_half(pipe: Any) -> tuple[torch.Tensor, ...]:
    """img2img half (1024×1024): 4096 noise + 1024 reference = 5120 img tokens."""
    return _dummy_flux2_transformer_img2img(pipe, noise_grid=64, ref_grids=(32,))


def dummy_flux2_transformer_img2img_full(pipe: Any) -> tuple[torch.Tensor, ...]:
    """img2img full (1024×1024): 4096 noise + 4096 reference = 8192 img tokens."""
    return _dummy_flux2_transformer_img2img(pipe, noise_grid=64, ref_grids=(64,))


def dummy_flux2_transformer_img2img_512_quarter(pipe: Any) -> tuple[torch.Tensor, ...]:
    """img2img quarter (512×512): 1024 noise + 64 reference = 1088 img tokens."""
    return _dummy_flux2_transformer_img2img(pipe, noise_grid=32, ref_grids=(8,))


def dummy_flux2_transformer_img2img_512_half(pipe: Any) -> tuple[torch.Tensor, ...]:
    """img2img half (512×512): 1024 noise + 256 reference = 1280 img tokens."""
    return _dummy_flux2_transformer_img2img(pipe, noise_grid=32, ref_grids=(16,))


def dummy_flux2_transformer_img2img_512_full(pipe: Any) -> tuple[torch.Tensor, ...]:
    """img2img full (512×512): 1024 noise + 1024 reference = 2048 img tokens."""
    return _dummy_flux2_transformer_img2img(pipe, noise_grid=32, ref_grids=(32,))


# ---------------------------------------------------------------------------
# Dynamic-shape specs
#
# `export_stateless` traces with `args=`, so each spec is a *positional* tuple
# aligned with its dummy-input tuple -- unlike the LLM paths, which key by
# parameter name. A None entry pins that input to its traced shape.
# ---------------------------------------------------------------------------

# Covers 256x256 txt2img (256 tokens) through 1024x1024 img2img with three
# full-resolution references (4096 noise + 12288 reference). Bounds must bracket
# the traced shape, and torch.export rejects a degenerate `Dim(min=n, max=n)`.
FLUX2_IMG_SEQ_MIN = 256
FLUX2_IMG_SEQ_MAX = 16384

# Output resolution bounds for a dynamic export. The multiple-of-16 requirement is
# real, not stylistic: the VAE downsamples by 8 and patchify folds a further 2x2, so
# `image_size / 16` must be the integral token-grid side.
#
# These are exported into metadata.json so the Swift side can reject an out-of-range
# resolution up front. Keep them here, beside the Dims they describe, rather than
# duplicating the numbers in the descriptor builder.
FLUX2_IMAGE_SIZE_MULTIPLE = 16
FLUX2_MIN_IMAGE_SIZE = 256
FLUX2_MAX_IMAGE_SIZE = 1024
_VAE_SPATIAL_SCALE = 8

# The encoder reaches lower than the output minimum on purpose. An img2img reference
# is encoded at its own size -- `reference_grid_side * 16` pixels -- so a small
# reference grid needs a correspondingly small encode. 128px is a poor *output* but a
# perfectly good reference thumbnail (8x8 tokens), and without it the smallest
# reachable reference would be 16x16 = 256 tokens, a regression on the fixed-shape
# path's quarter grid at 512.
#
# Only the encoder's Dim uses this; the decoder stays bounded by the output minimum
# because it only ever produces final images.
FLUX2_MIN_REFERENCE_SIZE = 128


def flux2_transformer_dynamic_shapes() -> tuple[dict[int, "torch.export.Dim"] | None, ...]:
    """Image sequence length is flexible; the text side stays fixed at 512.

    `hidden_states` and `img_ids` share one `Dim` object, which is how an
    equality constraint between two axes is expressed -- they are the same token
    sequence viewed as content and as position. The length is a sum of squares
    (noise grid plus each reference grid), so it is not factorable.
    """
    img_seq = torch.export.Dim("img_seq", min=FLUX2_IMG_SEQ_MIN, max=FLUX2_IMG_SEQ_MAX)
    return (
        {1: img_seq},  # hidden_states         [1, img_seq, in_channels]
        None,  # encoder_hidden_states [1, 512, joint_attention_dim]
        None,  # timestep              [1]
        None,  # guidance              [1]
        {1: img_seq},  # img_ids               [1, img_seq, num_axes]
        None,  # txt_ids               [1, 512, num_axes]
    )


def flux2_vae_decoder_dynamic_shapes() -> tuple[dict[int, "torch.export.Dim"] | None, ...]:
    """Latent height and width are flexible, and tied to each other (square only).

    Sharing one `Dim` across dims 2 and 3 is what enforces H == W. Latent side is
    `image_size / 8`; `image_size % 16 == 0` keeps it even.
    """
    latent_side = torch.export.Dim(
        "latent_side",
        min=FLUX2_MIN_IMAGE_SIZE // _VAE_SPATIAL_SCALE,
        max=FLUX2_MAX_IMAGE_SIZE // _VAE_SPATIAL_SCALE,
    )
    return ({2: latent_side, 3: latent_side},)  # z [1, latent_channels, S, S]


def flux2_vae_encoder_dynamic_shapes() -> tuple[dict[int, "torch.export.Dim"] | None, ...]:
    """Image height and width are flexible, and tied to each other (square only).

    Declared as a *multiple of 16* rather than a free range. The encoder's downsample
    stack produces guards over ``x // 2``, ``x // 4`` and ``x // 8``, which
    torch.export cannot discharge for a free dim -- it reports "Constraints violated"
    even though every value in range does satisfy them. Factoring hands it the
    divisibility fact directly.

    Lower bound is ``FLUX2_MIN_REFERENCE_SIZE``, not the output minimum, so a small
    img2img reference grid can be encoded at its own size rather than encoded full
    and then reduced.
    """
    side_16 = (
        torch.export.Dim(
            "image_side_16",
            min=FLUX2_MIN_REFERENCE_SIZE // FLUX2_IMAGE_SIZE_MULTIPLE,
            max=FLUX2_MAX_IMAGE_SIZE // FLUX2_IMAGE_SIZE_MULTIPLE,
        )
        * FLUX2_IMAGE_SIZE_MULTIPLE
    )
    return ({2: side_16, 3: side_16},)  # image [1, 3, S, S]
