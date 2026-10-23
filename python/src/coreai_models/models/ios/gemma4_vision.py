# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Gemma4 vision encoder iOS implementation.

Extracts the bespoke RoPE ViT vision tower + multimodal projector from a
multimodal Gemma4 checkpoint (``Gemma4ForConditionalGeneration``) and re-authors
it for the Neural Engine, producing image *soft tokens* in text-decoder space.

Bucketed geometry:
  - A ladder of square canvas sizes (patch-count buckets), not one fixed size: for
    a given bucket, ``image_size`` px -> ``image_size/patch_size`` grid -> a static
    ``num_patches`` count -> 3x3 spatial pool -> ``num_soft_tokens`` tokens. The
    runner picks the smallest bucket that fits an input image without upscaling it,
    letterboxes (aspect-preserving resize + pad) into that bucket's canvas, and
    passes an ``attn_mask`` marking which patches are real vs. padding.
  - The learned 2D position embeddings and 2D RoPE tables are compile-time constants
    *for the current bucket*, baked as buffers by ``resize_to`` -- one program is
    traced per bucket (see ``export/gemma4_vision.py``), each reusing the same
    loaded weights, mirroring how the text ladder retraces one decoder module per
    context bucket. ``attn_mask`` is the one true runtime input, since which patches
    are padding depends on the request's image, not just the chosen bucket.
  - One fused ``encode_image_<size>`` graph per bucket: patch-embed -> N ViT blocks
    (masked bidirectional attention) -> pool -> projector, output
    ``[1, num_soft_tokens, text_hidden]`` fp16. Padding is always aligned to whole
    pooled blocks (the letterbox only pads in multiples of ``patch_size *
    pooling_kernel_size``), so a pooled cell is never a mix of real and padding
    patches -- the runner drops the padding-only cells from the flat output itself.

Key architecture facts (verified against transformers >=5.5 ``gemma4``, E2B config):
  - hidden 768, 16 layers, 12 heads full MHA (head_dim 64), patch 16, RMSNorm eps 1e-6.
  - QK-norm (scaled) + scale-less V-norm; GeGLU ``gelu_pytorch_tanh``; 4 RMSNorms/layer.
  - Attention is **bidirectional**, scale 1.0 (Q/K are RMS-normed), no causal mask.
  - ``use_clipped_linears``: every projection is a bias-free Linear wrapped in a
    scalar ``clamp(input)`` / ``clamp(output)`` (ranges are per-linear buffers).
  - Patch embed = bias-free ``Linear(3*patch^2->hidden)`` (no conv stem); pixels
    centered in-model as ``2*(x-0.5)``.
  - 2D RoPE: theta 100, applied per spatial dim to head_dim/2 channels each
    (``apply_multidimensional_rope``, ndim=2).
  - Projector (``Gemma4MultimodalEmbedder``): scale-less RMSNorm -> bias-free
    ``Linear(hidden->text_hidden)``. NOTE the HF pooler's ``x sqrt(hidden)`` scaling is
    a no-op here -- it is immediately followed by the scale-less RMSNorm, which is
    scale-invariant -- so it is dropped (avoids fp16 overflow on the ANE).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from coreai_models.primitives.ios.rope import apply_rope
from coreai_models.primitives.ios.sdpa import SDPA

# fp16-overflow-safe RMSNorm. RMSNorm is scale-invariant, so we normalize each row
# by its own max magnitude *before* squaring: ``m = amax(|x|)`` (a data-dependent,
# per-row reduction), ``xn = x/m``. Since ``|xn| <= 1``, ``xn^2`` can never overflow
# fp16 (whereas raw ``x^2`` overflows once ``|x|>~256``: ``256^2~=65504``, the fp16 max
# -- the tower's residual stream peaks at ``|x|~=500`` on real images, so raw ``x^2``
# overflows to ``inf`` -> ``rsqrt(inf)=0`` -> zeroed features -> garbage soft tokens).
#
# The result is the *exact* RMSNorm because ``eps`` is folded back at the right scale::
#
#     xn*rsqrt(mean(xn^2) + (sqrt(eps)/m)^2) * w
#       = (x/m)*rsqrt(mean(x^2)/m^2 + eps/m^2) * w
#       = (x/m)*m*rsqrt(mean(x^2) + eps) * w
#       = x*rsqrt(mean(x^2) + eps) * w
#
# ``(sqrt(eps)/m)^2`` computes ``eps/m^2`` without ever forming ``m^2`` (which would
# itself overflow fp16 for ``m~=500``). Crucially this survives the ANE graph
# optimizer: a constant prescale (``x*0.125``) gets reassociated back into ``x^2*c^2``
# (recomputing the overflowing ``x^2``) or stripped as scale-invariant, but
# ``x/amax(|x|)`` is data-dependent and cannot be folded away. All ops
# (``abs``/``amax``/``div``/``mean``/``rsqrt``) are fp16 and ANE-resident -- no fp32
# upcast.


class RMSNormNoScale(nn.Module):
    """RMSNorm without a learnable scale (v_norm and the projector pre-norm).

    fp16-overflow-safe via per-row max-normalization before squaring (see the comment
    directly above, and :class:`RMSNorm`, which adds the scale). Exact and ANE-resident (no fp32
    upcast).
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self._eps = eps
        # Registered as a buffer (not a plain attribute) so it follows the module's
        # device on `.to(device)` -- a plain `torch.tensor(...)` created in forward()
        # defaults to CPU and raises on a GPU/MPS input (e.g. during parity tests).
        self.register_buffer("_eps_sqrt", torch.tensor(eps**0.5), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Per-row max magnitude; clamp floor avoids 0/0 on an all-zero row (real rows
        # have m >> floor, so this is a no-op for them).
        m = torch.clamp(x.abs().amax(-1, keepdim=True), min=1e-4)
        xn = x / m
        eps_sqrt = self._eps_sqrt.to(dtype=x.dtype)
        eps_term = (eps_sqrt / m) * (eps_sqrt / m)  # eps/m^2, no m^2 overflow
        mean_square = (xn * xn).mean(-1, keepdim=True)
        return xn * torch.rsqrt(mean_square + eps_term)


class RMSNorm(RMSNormNoScale):
    """:class:`RMSNormNoScale` followed by a learnable per-channel scale (last-dim reduction)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__(eps)
        with torch.device("cpu"):
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) * self.weight


class ClippedConvLinear(nn.Module):
    """Bias-free Linear (as a 1x1 Conv2d) with scalar input/output clamping.

    Mirrors HF ``Gemma4ClippableLinear``: ``clamp(x, in_min, in_max)`` -> linear ->
    ``clamp(out, out_min, out_max)``. Operates in BC1S-friendly ``(B, seq, 1, C)``
    layout: clamp is elementwise, the matmul runs as a 1x1 conv over the seq axis.
    Clamp bounds default to +-inf (no-op) and are overwritten by the checkpoint's
    per-linear range buffers at load.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Conv2d(in_features, out_features, kernel_size=1, bias=False)
        with torch.device("cpu"):
            self.input_min = nn.Buffer(torch.tensor(-float("inf")))
            self.input_max = nn.Buffer(torch.tensor(float("inf")))
            self.output_min = nn.Buffer(torch.tensor(-float("inf")))
            self.output_max = nn.Buffer(torch.tensor(float("inf")))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq, 1, in_features)
        x = torch.clamp(x, self.input_min, self.input_max)
        x = self.linear(x.transpose(-3, -1))  # (B, out, 1, seq)
        x = x.transpose(-3, -1)  # (B, seq, 1, out)
        x = torch.clamp(x, self.output_min, self.output_max)
        return x


class VisionAttention(nn.Module):
    """Bidirectional full-MHA vision attention with QK-norm + scale-less V-norm."""

    def __init__(self, config) -> None:
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        eps = config.rms_norm_eps

        self.q_proj = ClippedConvLinear(dim, self.n_heads * self.head_dim)
        self.k_proj = ClippedConvLinear(dim, self.n_heads * self.head_dim)
        self.v_proj = ClippedConvLinear(dim, self.n_heads * self.head_dim)
        self.o_proj = ClippedConvLinear(self.n_heads * self.head_dim, dim)

        self.q_norm = RMSNorm(self.head_dim, eps=eps)
        self.k_norm = RMSNorm(self.head_dim, eps=eps)
        self.v_norm = RMSNormNoScale(eps=eps)

        # scale 1.0: Q/K are RMS-normed, matching HF Gemma4VisionAttention.scaling.
        self.sdpa = SDPA(head_dim=self.head_dim, scale=1.0)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _, _ = x.shape
        n_heads, head_dim = self.n_heads, self.head_dim

        query = self.q_proj(x).reshape(batch_size, seq_len, n_heads, head_dim)
        query = self.q_norm(query)
        query = apply_rope(query, rope_cos, rope_sin, head_axis=2, ndim=2)

        key = self.k_proj(x).reshape(batch_size, seq_len, n_heads, head_dim)
        key = self.k_norm(key)
        key = apply_rope(key, rope_cos, rope_sin, head_axis=2, ndim=2)

        value = self.v_proj(x).reshape(batch_size, seq_len, n_heads, head_dim)
        value = self.v_norm(value)

        # To SDPA's BC1S layout: (B, n_heads*head_dim, 1, seq).
        query = query.reshape(batch_size, seq_len, 1, n_heads * head_dim).transpose(-3, -1)
        key = key.reshape(batch_size, seq_len, 1, n_heads * head_dim).transpose(-3, -1)
        value = value.reshape(batch_size, seq_len, 1, n_heads * head_dim).transpose(-3, -1)

        output = self.sdpa(query, key, value, attn_mask)  # (B, n_heads*head_dim, 1, seq)
        output = output.transpose(-3, -1)  # (B, seq, 1, n_heads*head_dim)
        return self.o_proj(output)


class VisionMLP(nn.Module):
    """GeGLU MLP with clipped linears (``gelu_pytorch_tanh`` gate)."""

    def __init__(self, config) -> None:
        super().__init__()
        dim = config.hidden_size
        hidden = config.intermediate_size
        self.gate_proj = ClippedConvLinear(dim, hidden)
        self.up_proj = ClippedConvLinear(dim, hidden)
        self.down_proj = ClippedConvLinear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = nn.functional.gelu(self.gate_proj(x), approximate="tanh")
        return self.down_proj(gate * self.up_proj(x))


class VisionEncoderLayer(nn.Module):
    """One ViT block: post-norm attention + post-norm GeGLU MLP (Gemma layout)."""

    def __init__(self, config) -> None:
        super().__init__()
        eps = config.rms_norm_eps
        dim = config.hidden_size
        self.self_attn = VisionAttention(config)
        self.mlp = VisionMLP(config)
        self.input_layernorm = RMSNorm(dim, eps=eps)
        self.post_attention_layernorm = RMSNorm(dim, eps=eps)
        self.pre_feedforward_layernorm = RMSNorm(dim, eps=eps)
        self.post_feedforward_layernorm = RMSNorm(dim, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        r = self.self_attn(self.input_layernorm(x), rope_cos, rope_sin, attn_mask)
        x = x + self.post_attention_layernorm(r)
        r = self.mlp(self.pre_feedforward_layernorm(x))
        x = x + self.post_feedforward_layernorm(r)
        return x


class Gemma4VisionEncoderForiOS(nn.Module):
    """Fused Gemma4 vision encoder: pixels -> image soft tokens (text-decoder space).

    Input  ``pixel_values`` ``(1, 3, image_size, image_size)`` NCHW, rescaled to [0,1].
    Input  ``attn_mask``    ``(1, num_patches, 1, num_patches)`` -- 0 for a real-patch
           key, a large negative sentinel for a padding-patch key (see ``resize_to``).
    Output ``image_features`` ``(1, num_soft_tokens, text_hidden)`` fp16.

    Ladder of buckets, not one fixed size: an input image is letterboxed (aspect
    ratio preserved) into the smallest bucket's square canvas that fits it without
    upscaling, so most images use only part of the canvas. ``attn_mask`` excludes
    the padding patches from attention; the caller (Swift) also knows which of the
    output's pooled cells are padding-only and drops them before treating the rest
    as soft tokens (padding is always pool-block-aligned by construction here -- see
    ``resize_to`` -- so a pooled cell is never a mix of real and padding patches).

    The 2D learned position embeddings (``pos_embeds``) and 2D RoPE tables
    (``rope_cos``/``rope_sin``) are baked as buffers for the *current* grid by
    ``resize_to``, so the graph has position/RoPE inputs for none of its buckets --
    only ``attn_mask`` varies per request, since which patches are padding depends
    on the actual image, not just the chosen bucket.
    """

    def __init__(self, config, text_hidden_size: int, image_size: int) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.text_hidden_size = text_hidden_size
        self.pool_k = config.pooling_kernel_size
        # Soft-token dtype: fp16 for the runner contract; overridden to the load
        # dtype by from_state_dict (fp32 in parity tests).
        self._output_dtype = torch.float16

        patch_dim = 3 * config.patch_size * config.patch_size
        self.input_proj = nn.Conv2d(patch_dim, self.hidden_size, kernel_size=1, bias=False)

        self.layers = nn.ModuleList(
            [VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        # Projector (Gemma4MultimodalEmbedder): scale-less RMSNorm -> bias-free Linear.
        self.embedding_pre_projection_norm = RMSNormNoScale(eps=config.rms_norm_eps)
        self.embedding_projection = nn.Conv2d(
            self.hidden_size, text_hidden_size, kernel_size=1, bias=False
        )

        # Raw learned position-embedding table (HF `patch_embedder.position_embedding_table`),
        # filled by from_state_dict. Resolution-independent -- `resize_to` re-queries it
        # by (x, y) coordinate for whatever grid the current bucket needs, so it's kept
        # once here rather than re-derived per bucket.
        with torch.device("cpu"):
            self._position_embedding_table_raw = nn.Buffer(
                torch.zeros(2, config.position_embedding_size, config.hidden_size),
                persistent=False,
            )

        self.resize_to(image_size)

    def resize_to(self, image_size: int) -> None:
        """(Re-)derive grid geometry and the position/RoPE buffers for `image_size`,
        reusing this instance's already-loaded weights unchanged (the position
        embedding table, and every conv/attention layer, are resolution-independent).

        Used to trace one program per bucket in the size ladder from a single loaded
        model -- mirrors the text ladder retracing one `Gemma4Extend` instance per
        context bucket, just for grid size instead of context length.
        """
        from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionRotaryEmbedding

        self.image_size = image_size
        self.grid = image_size // self.patch_size  # patches per side (e.g. 48)
        self.num_patches = self.grid * self.grid  # e.g. 2304
        k = self.pool_k
        self.pooled_grid = self.grid // k  # e.g. 16
        self.num_soft_tokens = self.pooled_grid * self.pooled_grid  # e.g. 256
        assert self.pooled_grid * k == self.grid, (
            f"grid {self.grid} not divisible by pooling kernel {k}"
        )

        with torch.device("cpu"):
            self.pos_embeds = nn.Buffer(
                torch.zeros(1, self.num_patches, 1, self.hidden_size, dtype=self._output_dtype),
                persistent=False,
            )
            self.rope_cos = nn.Buffer(
                torch.zeros(1, self.num_patches, self.config.head_dim, dtype=self._output_dtype),
                persistent=False,
            )
            self.rope_sin = nn.Buffer(
                torch.zeros(1, self.num_patches, self.config.head_dim, dtype=self._output_dtype),
                persistent=False,
            )

        # position_ids: (1, num_patches, 2) with (x=col, y=row), row-major patches.
        grid = self.grid
        rows = torch.arange(grid).repeat_interleave(grid)  # y per patch
        cols = torch.arange(grid).repeat(grid)  # x per patch
        pos_ids = torch.stack([cols, rows], dim=-1).unsqueeze(0)  # (1, N, 2)

        with torch.no_grad():
            # Learned 2D position embeddings: the x row of the table indexed by column plus the y
            # row indexed by row (HF's patch embedder computes exactly this; no patch here is
            # padding). `.float()` guards against `self._position_embedding_table_raw` having
            # been downcast by an earlier whole-model `.to(dtype=...)` call (this method re-runs
            # once per bucket in the size ladder, potentially long after that cast): computing in
            # fp32 avoids compounding rounding across buckets, even though the *stored* table may
            # already carry one round of precision loss from that cast.
            x_table, y_table = self._position_embedding_table_raw.float()  # (size, hidden) each
            pos_embeds = F.embedding(pos_ids[..., 0], x_table) + F.embedding(
                pos_ids[..., 1], y_table
            )
            self.pos_embeds.copy_(pos_embeds.reshape(1, self.num_patches, 1, -1))

            # 2D RoPE cos/sin via the HF rotary module (theta 100, default rope) --
            # config-only (no learned weights), so no stored raw table is needed.
            hf_rope = Gemma4VisionRotaryEmbedding(self.config)
            cos, sin = hf_rope(torch.zeros(1, dtype=torch.float32), pos_ids)  # (1, N, head_dim)
            self.rope_cos.copy_(cos.reshape(1, self.num_patches, -1))
            self.rope_sin.copy_(sin.reshape(1, self.num_patches, -1))

    def _patchify(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """NCHW pixels -> row-major patch vectors, matching HF ``convert_image_to_patches``.

        ``pixel_values``: ``(1, 3, H, W)`` (the layout Swift ``ImagePreprocessor.preprocessCHW``
        produces, reshaped to NCHW). Returns ``(1, num_patches, 3*patch^2)`` where patch
        n = row*grid + col and the per-patch vector is ordered ``(patch_h, patch_w, channel)``
        (channel innermost) -- exactly what ``patch_embedder.input_proj`` was trained on.

        Reproduces the processor's ``reshape(C, nph, p, npw, p) -> permute(1,3,2,4,0) ->
        reshape(N, ·)`` at rank 5 (batch 1 is squeezed; a rank-6 batched reshape would
        exceed the ANE's max tensor rank).
        """
        ps, g = self.patch_size, self.grid
        # (1,3,H,W) -> (3, nph, p, npw, p): batch 1 folds away in the reshape.
        x = pixel_values.reshape(3, g, ps, g, ps)
        x = x.permute(1, 3, 2, 4, 0)  # (nph, npw, p, p, C)
        return x.reshape(1, self.num_patches, ps * ps * 3)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        """3x3 spatial average pool of the (row-major) patch grid.

        ``x``: ``(B, num_patches, hidden)`` in row-major order (patch n = row*grid + col).
        Returns ``(B, num_soft_tokens, hidden)`` where soft token m = row_block*pooled_grid
        + col_block -- matching HF ``Gemma4VisionPooler`` for a full (no-padding) grid.

        Uses ``avg_pool2d`` (rank-4 NCHW) rather than a grid reshape: a
        ``(B, pg, k, pg, k, hidden)`` reshape is rank 6 and exceeds the ANE's max
        tensor rank (5), dropping the op off the Neural Engine.
        """
        b, g = x.shape[0], self.grid
        # (B, N, hidden) row-major -> (B, hidden, grid, grid) NCHW (row=H, col=W).
        x = x.reshape(b, g, g, self.hidden_size).permute(0, 3, 1, 2)
        x = nn.functional.avg_pool2d(x, kernel_size=self.pool_k, stride=self.pool_k)
        # (B, hidden, pg, pg) -> (B, pg*pg, hidden), row-major over pooled blocks.
        pg = self.pooled_grid
        return x.reshape(b, self.hidden_size, pg * pg).transpose(1, 2)

    def forward(self, pixel_values: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """``pixel_values``: ``(1, 3, H, W)`` NCHW, rescaled to [0,1] (runner side).
        ``attn_mask``: ``(1, num_patches, 1, num_patches)`` -- 0 for a real-patch key,
        a large negative sentinel for a padding-patch key (see ``resize_to``'s
        docstring). Fully-visible (all-zero) for a request that fills the whole
        bucket, i.e. the original fixed-size behavior.

        fp16 in the shipped graph so ANE I/O stays fp16-native (the Swift runner fills
        it as Float16); the ``to(...)`` keeps the module usable at fp32 in parity tests.
        """
        b = pixel_values.shape[0]
        pixel_values = pixel_values.to(self.input_proj.weight.dtype)

        # In-graph patchify (Swift feeds NCHW pixels, like the Qwen VLM path).
        patches = self._patchify(pixel_values)  # (1, num_patches, 3*patch^2)

        # Patch embed: center pixels (2*(x-0.5)) then bias-free projection + pos embeds.
        x = 2.0 * (patches - 0.5)
        x = x.reshape(b, self.num_patches, 1, -1)
        x = self.input_proj(x.transpose(-3, -1)).transpose(-3, -1)  # (B, N, 1, hidden)
        x = x + self.pos_embeds

        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin, attn_mask)

        # (B, N, 1, hidden) -> (B, N, hidden) for pooling.
        x = x.reshape(b, self.num_patches, self.hidden_size)
        x = self._pool(x)  # (B, num_soft_tokens, hidden)

        # Projector. The HF x sqrt(hidden) pooler scaling is dropped: it precedes the
        # scale-less RMSNorm, which is scale-invariant, so it cancels exactly.
        x = x.reshape(b, self.num_soft_tokens, 1, self.hidden_size)
        x = self.embedding_pre_projection_norm(x)
        x = self.embedding_projection(x.transpose(-3, -1)).transpose(-3, -1)
        # NB: the tail projection runs in channel-major BC1S (conv over a transposed
        # view), so the logical (B, tokens, hidden) result sits on a transposed stride.
        # The runtime reads the output NDArray as a FLAT contiguous [tokens, hidden]
        # buffer (StaticShapeEngine.scatterImageEmbeddings); if the physical layout is
        # channel-major the flat read transposes each token's vector -> scrambled soft
        # tokens -> the decoder hallucinates. Force a contiguous token-major copy so
        # the flat read is correct.
        x = x.reshape(b, self.num_soft_tokens, self.text_hidden_size).contiguous()
        return x.to(self._output_dtype)

    # ------------------------------------------------------------------
    # Loader
    # ------------------------------------------------------------------

    @classmethod
    def from_hf(
        cls,
        huggingface_model_id: str,
        image_size: int = 768,
        target_dtype: torch.dtype = torch.float16,
    ) -> "Gemma4VisionEncoderForiOS":
        """Load the vision tower + projector from a multimodal Gemma4 checkpoint.

        Bakes the fixed-grid position embeddings and 2D RoPE tables (computed with
        the HF reference modules for exact parity), remaps HF weights into the
        BC1S/Conv2d layout, and loads the per-linear clip ranges.

        Loads weights directly from the checkpoint's safetensors shards rather than
        instantiating ``Gemma4ForConditionalGeneration`` (which would build and
        weight the full multimodal model, including the multi-billion-parameter text
        decoder, just to read off the ~200M-parameter vision tower) -- config comes
        from a separate, weight-free ``AutoConfig`` load.
        """
        from transformers import AutoConfig

        from coreai_models._hf import load_named_tensors_from_weight_files

        hf_config = AutoConfig.from_pretrained(huggingface_model_id)
        vision_cfg = hf_config.vision_config
        text_hidden = hf_config.text_config.hidden_size

        vision_prefix = "model.vision_tower."
        embed_prefix = "model.embed_vision."
        full_sd = {
            k: v
            for k, v in load_named_tensors_from_weight_files(huggingface_model_id).items()
            if k.startswith(vision_prefix) or k.startswith(embed_prefix)
        }
        return cls.from_state_dict(
            full_sd,
            vision_cfg,
            text_hidden_size=text_hidden,
            image_size=image_size,
            target_dtype=target_dtype,
        )

    @classmethod
    def from_state_dict(
        cls,
        full_sd: dict[str, torch.Tensor],
        vision_cfg,
        *,
        text_hidden_size: int,
        image_size: int = 768,
        target_dtype: torch.dtype = torch.float16,
        vision_prefix: str = "model.vision_tower.",
        embed_prefix: str = "model.embed_vision.",
    ) -> "Gemma4VisionEncoderForiOS":
        """Build + load from a state dict (shared by :meth:`from_hf` and tests).

        ``full_sd`` must carry the vision tower under ``vision_prefix`` and the
        projector under ``embed_prefix`` (the HF ``Gemma4ForConditionalGeneration``
        layout). Weights are read in float32 for the position/RoPE baking.
        """
        model = cls(vision_cfg, text_hidden_size=text_hidden_size, image_size=image_size)
        model._output_dtype = target_dtype
        vt = vision_prefix
        ev = embed_prefix

        # Load the raw position-embedding table, then re-bake pos_embeds/rope_cos/
        # rope_sin for `image_size` against the *real* table (resize_to's own call
        # from __init__ baked against a still-zero table).
        with torch.no_grad():
            model._position_embedding_table_raw.copy_(
                full_sd[vt + "patch_embedder.position_embedding_table"].float()
            )
        model.resize_to(image_size)

        # ---- Remap encoder + projector weights ----
        new_sd: dict[str, torch.Tensor] = {}

        def conv(w: torch.Tensor) -> torch.Tensor:
            return w.unsqueeze(-1).unsqueeze(-1)

        new_sd["input_proj.weight"] = conv(full_sd[vt + "patch_embedder.input_proj.weight"])
        new_sd["embedding_projection.weight"] = conv(full_sd[ev + "embedding_projection.weight"])

        n_layers = vision_cfg.num_hidden_layers
        for i in range(n_layers):
            src = f"{vt}encoder.layers.{i}."
            dst = f"layers.{i}."
            for norm in [
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ]:
                new_sd[dst + norm + ".weight"] = full_sd[src + norm + ".weight"]
            new_sd[dst + "self_attn.q_norm.weight"] = full_sd[src + "self_attn.q_norm.weight"]
            new_sd[dst + "self_attn.k_norm.weight"] = full_sd[src + "self_attn.k_norm.weight"]
            # Clipped linears: linear weight (->conv) + 4 scalar clip ranges. Clip
            # ranges are kept fp32 regardless of the checkpoint's stored dtype (see
            # the cast loop below, which explicitly skips these keys) -- force it here
            # too, since `full_sd` is no longer guaranteed fp32 end-to-end (loaded
            # straight from safetensors rather than via a `torch_dtype=torch.float32`
            # model load).
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                new_sd[dst + f"self_attn.{proj}.linear.weight"] = conv(
                    full_sd[src + f"self_attn.{proj}.linear.weight"]
                )
                for b_ in ["input_min", "input_max", "output_min", "output_max"]:
                    new_sd[dst + f"self_attn.{proj}.{b_}"] = full_sd[
                        src + f"self_attn.{proj}.{b_}"
                    ].float()
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                new_sd[dst + f"mlp.{proj}.linear.weight"] = conv(
                    full_sd[src + f"mlp.{proj}.linear.weight"]
                )
                for b_ in ["input_min", "input_max", "output_min", "output_max"]:
                    new_sd[dst + f"mlp.{proj}.{b_}"] = full_sd[src + f"mlp.{proj}.{b_}"].float()

        del full_sd

        model.to(dtype=target_dtype)
        # Cast float weights to target dtype; keep clip-range buffers in fp32 for stable clamp.
        for k_, v in new_sd.items():
            if v.is_floating_point() and not any(
                s in k_ for s in ["input_min", "input_max", "output_min", "output_max"]
            ):
                new_sd[k_] = v.to(target_dtype)

        missing, unexpected = model.load_state_dict(new_sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"state_dict mismatch: missing={missing}, unexpected={unexpected}")

        return model.eval()
