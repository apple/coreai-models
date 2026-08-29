# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Muse Glimmer ViT-G/14 vision encoder for CoreAI export.

Standalone re-implementation of the Muse Glimmer perception encoder that
loads directly from the HF safetensors checkpoint.  The text decoder lives
in ``muse_glimmer.py``; this file covers the vision pipeline:

  pixel_values [1, 3, H, W]
    -> patchify (14x14, temporal duplication for single image)
    -> learned position embeddings (bilinear-interpolated 32x32 table)
    -> 2D RoPE (interleaved [w,h,w,h], positions offset by +1)
    -> LayerNorm pre
    -> 50 transformer layers (pre-norm attention + GELU MLP)
    -> LayerNorm post
    -> 2x2 spatial merge (pixel shuffle)
    -> adapter MLP  fc1(6144->4096)->GELU->fc2(4096->4096)->GELU
    -> projection Linear(4096->6656)
    -> perception norm (weight-less RMSNorm)
  -> image_features [1, N, 6656]

Weight key mapping from HF checkpoint::

  model.vision_tower.patch_embedder.*   -> encoder.patch_embedder.*
  model.vision_tower.ln_pre/ln_post.*   -> encoder.ln_pre/ln_post.*
  model.vision_tower.layers.N.*         -> encoder.layers.N.*
  model.vision_adapter.fc1/fc2.*        -> projector.fc1/fc2.*
  model.vision_projection.*             -> projector.projection.*

Architecture notes (from ``meta-models/Muse-Glimmer-30B``):

  hidden_size       = 1536      num_attention_heads = 16  (head_dim=96)
  intermediate_size = 8960      num_hidden_layers   = 50
  patch_size        = 14        patch_temporal      = 2
  merge_size        = 2         pos_emb 32x32       = 1024 entries
  layer_types       = [W,W,W,F]x12 + [W,F]  (37 window + 13 full)
  rope_theta        = 10000.0   (2D, per spatial dim)

Window vs full attention:  The window size equals ``pos_emb_height * patch_size``
(32*14 = 448).  At the native 448x448 resolution the entire patch grid (32x32)
fits in one window, so window attention = full attention for all layers.  Higher
resolutions would need proper window partitioning.
"""

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open

from coreai_models.models.base import (
    _load_tensors_for_keys,
    _resolve_safetensors_files,
)

# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate second half of the last dimension: ``cat(-x2, x1)``."""
    mid = x.shape[-1] // 2
    return torch.cat((-x[..., mid:], x[..., :mid]), dim=-1)


def _apply_rotary_pos_emb_vision(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply 2D RoPE to query and key tensors (vision encoder variant).

    Args:
        q, k: ``[1, seq_len, num_heads, head_dim]``
        cos, sin: ``[1, seq_len, head_dim]``  (broadcast over heads)
    """
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos = cos.unsqueeze(-2).float()  # [1, seq_len, 1, head_dim]
    sin = sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------


class VisionAttention(nn.Module):
    """Multi-head self-attention for the Muse Glimmer ViT-G/14.

    - 16 heads, head_dim 96, all projections have bias.
    - Non-causal (bidirectional) attention.
    - Pre-computed RoPE cos/sin passed from outside.
    """

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: ``[seq_len, hidden_size]``
            cos, sin: ``[1, seq_len, head_dim]``
        Returns:
            ``[seq_len, hidden_size]``
        """
        seq_len = hidden_states.shape[0]

        # Project -> [1, seq_len, heads, head_dim]
        q = self.q_proj(hidden_states).reshape(1, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(1, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(1, seq_len, self.num_heads, self.head_dim)

        # 2D RoPE
        q, k = _apply_rotary_pos_emb_vision(q, k, cos, sin)

        # -> [1, heads, seq_len, head_dim] for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        # -> [seq_len, hidden_size]
        attn_out = attn_out.transpose(1, 2).reshape(seq_len, self.hidden_size)
        return self.proj(attn_out)


class VisionMLP(nn.Module):
    """``fc1(hidden -> intermediate) -> GELU -> fc2(intermediate -> hidden)``."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class VisionEncoderLayer(nn.Module):
    """Pre-norm transformer layer: ``norm1 -> attn + residual, norm2 -> mlp + residual``."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attn = VisionAttention(hidden_size, num_heads)
        self.mlp = VisionMLP(hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cos, sin)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


# ---------------------------------------------------------------------------
# Patch embedder (matches HF sub-module structure for easy key mapping)
# ---------------------------------------------------------------------------


class VisionPatchEmbedder(nn.Module):
    """Patch embedding + learned position embedding table.

    The ``patch_embedding`` is a bias-free Linear that maps flattened
    per-patch pixels ``[num_patches, patch_temporal * 3 * patch_size^2]``
    to ``[num_patches, hidden_size]``.

    Position embeddings are bilinear-interpolated from a ``(pos_emb_height *
    pos_emb_width, hidden_size)`` learned table.  At the native resolution
    (grid matches the table) this reduces to a simple lookup.
    """

    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        patch_temporal: int,
        pos_emb_height: int,
        pos_emb_width: int,
    ) -> None:
        super().__init__()
        patch_dim = patch_temporal * 3 * patch_size * patch_size
        self.patch_embedding = nn.Linear(patch_dim, hidden_size, bias=False)
        self.position_embedding_table = nn.Embedding(pos_emb_height * pos_emb_width, hidden_size)
        self.pos_emb_height = pos_emb_height
        self.pos_emb_width = pos_emb_width

    def _interpolate_pos(self, grid_h: int, grid_w: int) -> torch.Tensor:
        """Bilinear-interpolated position embeddings for an arbitrary grid.

        At the native resolution (``grid_h == pos_emb_height``) this is a
        direct lookup (no interpolation).  For other resolutions it mimics
        ``F.grid_sample(..., align_corners=False, padding_mode="zeros")``.

        Returns:
            ``[grid_h * grid_w, hidden_size]``
        """
        device = self.position_embedding_table.weight.device
        side_h = self.pos_emb_height
        side_w = self.pos_emb_width

        if grid_h == side_h and grid_w == side_w:
            return self.position_embedding_table(torch.arange(side_h * side_w, device=device))

        # Bilinear interpolation
        h_grid = (torch.arange(grid_h, device=device).float() + 0.5) * (side_h / grid_h) - 0.5
        w_grid = (torch.arange(grid_w, device=device).float() + 0.5) * (side_w / grid_w) - 0.5

        h_floor = torch.floor(h_grid).long()
        w_floor = torch.floor(w_grid).long()
        h_ceil = h_floor + 1
        w_ceil = w_floor + 1
        h_frac = h_grid - h_floor.float()
        w_frac = w_grid - w_floor.float()

        # Validity masks (zero-padding for out-of-bounds)
        h_floor_v = (h_floor >= 0) & (h_floor < side_h)
        h_ceil_v = (h_ceil >= 0) & (h_ceil < side_h)
        w_floor_v = (w_floor >= 0) & (w_floor < side_w)
        w_ceil_v = (w_ceil >= 0) & (w_ceil < side_w)

        h_floor_c = h_floor.clamp(0, side_h - 1)
        h_ceil_c = h_ceil.clamp(0, side_h - 1)
        w_floor_c = w_floor.clamp(0, side_w - 1)
        w_ceil_c = w_ceil.clamp(0, side_w - 1)

        # Four corners: indices [4, N] and weights [4, N]
        indices = torch.stack(
            [
                (h_floor_c[:, None] * side_w + w_floor_c[None, :]).flatten(),
                (h_floor_c[:, None] * side_w + w_ceil_c[None, :]).flatten(),
                (h_ceil_c[:, None] * side_w + w_floor_c[None, :]).flatten(),
                (h_ceil_c[:, None] * side_w + w_ceil_c[None, :]).flatten(),
            ]
        )
        weights = torch.stack(
            [
                (
                    (1 - h_frac)[:, None]
                    * (1 - w_frac)[None, :]
                    * (h_floor_v[:, None] & w_floor_v[None, :])
                ).flatten(),
                (
                    (1 - h_frac)[:, None]
                    * w_frac[None, :]
                    * (h_floor_v[:, None] & w_ceil_v[None, :])
                ).flatten(),
                (
                    h_frac[:, None]
                    * (1 - w_frac)[None, :]
                    * (h_ceil_v[:, None] & w_floor_v[None, :])
                ).flatten(),
                (
                    h_frac[:, None] * w_frac[None, :] * (h_ceil_v[:, None] & w_ceil_v[None, :])
                ).flatten(),
            ]
        )

        pos = self.position_embedding_table(indices)  # [4, N, hidden]
        return (pos * weights[:, :, None]).sum(0)  # [N, hidden]

    def forward(self, patches: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        """Embed patches and add position embeddings.

        Args:
            patches: ``[num_patches, patch_dim]``  pre-patchified pixel values.
            grid_h, grid_w: spatial grid dimensions (before merge).

        Returns:
            ``[num_patches, hidden_size]``
        """
        embeddings = self.patch_embedding(patches)
        pos = self._interpolate_pos(grid_h, grid_w)
        return embeddings + pos.to(embeddings.dtype)


# ---------------------------------------------------------------------------
# Vision encoder (ViT-G/14)
# ---------------------------------------------------------------------------


class MuseGlimmerVisionEncoder(nn.Module):
    """Muse Glimmer ViT-G/14 vision encoder.

    Processes pre-patchified tokens through the full transformer stack and
    produces merged hidden states.

    Architecture::

        patch_embedder  ->  ln_pre  ->  50 x VisionEncoderLayer  ->  ln_post
                                                                    |
                                                             pixel_shuffle (2x2)

    Module names mirror the HF checkpoint keys (after stripping
    ``model.vision_tower.``) so weight loading needs only prefix removal.
    """

    def __init__(
        self,
        hidden_size: int = 1536,
        intermediate_size: int = 8960,
        num_hidden_layers: int = 50,
        num_attention_heads: int = 16,
        patch_size: int = 14,
        patch_temporal: int = 2,
        merge_size: int = 2,
        pos_emb_height: int = 32,
        pos_emb_width: int = 32,
        layer_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.patch_size = patch_size
        self.patch_temporal = patch_temporal
        self.merge_size = merge_size

        # --- Patch embedder (sub-module matches HF key hierarchy) ---
        self.patch_embedder = VisionPatchEmbedder(
            hidden_size,
            patch_size,
            patch_temporal,
            pos_emb_height,
            pos_emb_width,
        )

        # --- Layer norms ---
        self.ln_pre = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ln_post = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        # --- Transformer layers ---
        self.layers = nn.ModuleList(
            [
                VisionEncoderLayer(
                    hidden_size, intermediate_size, num_attention_heads, layer_norm_eps
                )
                for _ in range(num_hidden_layers)
            ]
        )

        # --- RoPE inverse frequencies (pre-computed, persistent=False) ---
        # 2D RoPE: each spatial dim gets ``spatial_dim // 2`` frequencies.
        spatial_dim = self.head_dim // 2  # 48
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, spatial_dim, 2, dtype=torch.float32) / spatial_dim)
        )
        self.register_buffer("_rope_inv_freq", inv_freq, persistent=False)

    # ---- helpers --------------------------------------------------------

    def _compute_rope(self, grid_h: int, grid_w: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Pre-compute 2D RoPE cos/sin for a spatial grid.

        Frequency pattern is ``[freq_w, freq_h, freq_w, freq_h]`` (interleaved,
        matching the HF reference).  Position indices are 1-based (offset +1).

        Returns:
            ``(cos, sin)`` each of shape ``[1, grid_h * grid_w, head_dim]``.
        """
        device = self._rope_inv_freq.device

        hpos, wpos = torch.meshgrid(
            torch.arange(grid_h, device=device),
            torch.arange(grid_w, device=device),
            indexing="ij",
        )
        # 1-based positions
        h_ids = hpos.flatten().float() + 1.0  # [seq_len]
        w_ids = wpos.flatten().float() + 1.0

        # inv_freq: [num_freqs] -> [1, num_freqs, 1]
        inv_freq = self._rope_inv_freq[None, :, None]

        # [1, num_freqs, 1] @ [1, 1, seq_len] -> [1, num_freqs, seq_len] -> transpose
        freq_h = (inv_freq @ h_ids[None, None, :]).transpose(1, 2)  # [1, seq_len, num_freqs]
        freq_w = (inv_freq @ w_ids[None, None, :]).transpose(1, 2)

        # Interleave: [freq_w, freq_h, freq_w, freq_h] -> [1, seq_len, head_dim]
        freq = torch.cat([freq_w, freq_h, freq_w, freq_h], dim=-1)
        return freq.cos(), freq.sin()

    def _pixel_shuffle(self, hidden_states: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        """2x2 spatial merge (pixel shuffle).

        Groups every ``merge_size x merge_size`` block of adjacent patches and
        concatenates their hidden states along the feature dimension.

        ``[num_patches, hidden]`` -> ``[num_patches / merge^2, hidden * merge^2]``
        """
        factor = self.merge_size
        dim = self.hidden_size

        # Permutation that groups 2x2 spatial blocks together
        perm = (
            torch.arange(grid_h * grid_w, device=hidden_states.device)
            .view(grid_h // factor, factor, grid_w // factor, factor)
            .permute(0, 2, 1, 3)
            .reshape(-1)
        )

        n_merged = (grid_h // factor) * (grid_w // factor)
        hs = hidden_states[perm]
        hs = hs.view(n_merged, factor * factor, dim)
        hs = hs.permute(0, 2, 1).contiguous()
        return hs.view(n_merged, dim * factor * factor)

    # ---- forward --------------------------------------------------------

    def forward(
        self,
        patches: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        """Run the vision encoder.

        Args:
            patches: ``[num_patches, patch_dim]``  pre-patchified pixel values.
            grid_h, grid_w: spatial grid dimensions (before merge).

        Returns:
            ``[num_merged, hidden_size * merge_size^2]``  where
            ``num_merged = (grid_h / merge) * (grid_w / merge)``.
        """
        # Patch embedding + position embeddings
        hidden_states = self.patch_embedder(patches, grid_h, grid_w)

        # Pre-norm
        hidden_states = self.ln_pre(hidden_states)

        # 2D RoPE (pre-computed for the static grid)
        cos, sin = self._compute_rope(grid_h, grid_w)
        cos = cos.to(hidden_states.dtype)
        sin = sin.to(hidden_states.dtype)

        # Transformer layers
        # At 448x448 (native resolution) window_size covers the whole grid,
        # so all layers use identical full attention.  For higher resolutions,
        # window-attention layers should restrict to sub-windows.
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin)

        # Post-norm
        hidden_states = self.ln_post(hidden_states)

        # Spatial merge (2x2 pixel shuffle)
        hidden_states = self._pixel_shuffle(hidden_states, grid_h, grid_w)

        return hidden_states


# ---------------------------------------------------------------------------
# Multi-modal projector (vision adapter + linear projection + norm)
# ---------------------------------------------------------------------------


class MuseGlimmerMultiModalProjector(nn.Module):
    """Vision adapter + projection + perception norm for Muse Glimmer.

    Pipeline::

        fc1(out_hidden -> proj_hidden, no bias) -> GELU
        fc2(proj_hidden -> proj_hidden, no bias) -> GELU
        projection(proj_hidden -> text_hidden, no bias)
        perception_norm (weight-less RMSNorm)

    Weight key mapping from HF checkpoint::

        model.vision_adapter.fc1  ->  fc1
        model.vision_adapter.fc2  ->  fc2
        model.vision_projection   ->  projection
    """

    def __init__(
        self,
        out_hidden_size: int = 6144,
        projector_hidden_size: int = 4096,
        text_hidden_size: int = 6656,
        rms_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(out_hidden_size, projector_hidden_size, bias=False)
        self.fc2 = nn.Linear(projector_hidden_size, projector_hidden_size, bias=False)
        self.projection = nn.Linear(projector_hidden_size, text_hidden_size, bias=False)
        self.rms_norm_eps = rms_norm_eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``[N, out_hidden]`` -> ``[N, text_hidden]``."""
        hidden_states = F.gelu(self.fc1(hidden_states))
        hidden_states = F.gelu(self.fc2(hidden_states))

        hidden_states = self.projection(hidden_states)

        x_float = hidden_states.float()
        normed = x_float * torch.pow(
            x_float.pow(2).mean(-1, keepdim=True) + self.rms_norm_eps, -0.5
        )
        return normed.type_as(hidden_states)


# ---------------------------------------------------------------------------
# Combined vision model (encoder + projector, from_pretrained loader)
# ---------------------------------------------------------------------------


class MuseGlimmerVisionModel(nn.Module):
    """End-to-end Muse Glimmer vision pipeline.

    Takes raw pixel values and produces features aligned to the text hidden
    space, ready to be scattered into the token-embedding sequence.

    Input:  ``pixel_values [1, 3, H, W]``  (single image, RGB, pre-normalized)
    Output: ``image_features [1, N, text_hidden_size]``

    where ``N = (H / patch_size / merge_size) * (W / patch_size / merge_size)``,
    e.g. 256 for a 448x448 image (32/2 * 32/2).
    """

    def __init__(
        self,
        # Vision encoder config
        hidden_size: int = 1536,
        intermediate_size: int = 8960,
        num_hidden_layers: int = 50,
        num_attention_heads: int = 16,
        patch_size: int = 14,
        patch_temporal: int = 2,
        merge_size: int = 2,
        pos_emb_height: int = 32,
        pos_emb_width: int = 32,
        layer_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        # Projector config (from top-level HF config)
        out_hidden_size: int = 6144,
        projector_hidden_size: int = 4096,
        text_hidden_size: int = 6656,
        rms_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_temporal = patch_temporal
        self.merge_size = merge_size

        self.encoder = MuseGlimmerVisionEncoder(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            patch_size=patch_size,
            patch_temporal=patch_temporal,
            merge_size=merge_size,
            pos_emb_height=pos_emb_height,
            pos_emb_width=pos_emb_width,
            layer_norm_eps=layer_norm_eps,
            rope_theta=rope_theta,
        )

        self.projector = MuseGlimmerMultiModalProjector(
            out_hidden_size=out_hidden_size,
            projector_hidden_size=projector_hidden_size,
            text_hidden_size=text_hidden_size,
            rms_norm_eps=rms_norm_eps,
        )

    # ---- patchification -------------------------------------------------

    def patchify(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Convert raw pixels to flattened patch tokens.

        Single image: ``[1, 3, H, W]`` -> duplicate along temporal dimension
        (``patch_temporal=2``), then reshape into per-patch vectors of size
        ``patch_temporal * 3 * patch_size^2 = 1176``.

        Returns:
            ``(patches, grid_h, grid_w)`` where patches is
            ``[num_patches, 1176]`` and grid_h/grid_w are the spatial dims.
        """
        ps = self.patch_size
        pt = self.patch_temporal
        c = 3

        # Remove batch dim: [1, 3, H, W] -> [3, H, W]
        x = pixel_values.squeeze(0)
        _, h, w = x.shape
        grid_h = h // ps
        grid_w = w // ps

        # Duplicate for temporal: [3, H, W] -> [temporal, 3, H, W]
        x = x.unsqueeze(0).expand(pt, -1, -1, -1)

        # Reshape into patch grid:
        #   [temporal, C, grid_h, patch_h, grid_w, patch_w]
        x = x.reshape(pt, c, grid_h, ps, grid_w, ps)

        # Permute to group per-patch dims:
        #   [grid_h, grid_w, temporal, C, patch_h, patch_w]
        x = x.permute(2, 4, 0, 1, 3, 5)

        # Flatten: [num_patches, patch_dim]
        return x.reshape(grid_h * grid_w, pt * c * ps * ps), grid_h, grid_w

    # ---- forward --------------------------------------------------------

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Full vision pipeline.

        Args:
            pixel_values: ``[1, 3, H, W]``  pre-normalized image (RGB,
                rescaled by 1/255, normalized with mean=0.5, std=0.5).

        Returns:
            ``[1, N, text_hidden_size]``  image features ready for the
            text decoder, where ``N = (H/14/2) * (W/14/2)``.
        """
        patches, grid_h, grid_w = self.patchify(pixel_values)
        hidden_states = self.encoder(patches, grid_h, grid_w)
        hidden_states = self.projector(hidden_states)
        return hidden_states.unsqueeze(0)  # add batch dim

    # ---- pretrained loader ----------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str,
        dtype: torch.dtype = torch.float32,
    ) -> "MuseGlimmerVisionModel":
        """Load the vision model from a Muse Glimmer HF checkpoint.

        Downloads the checkpoint (or uses cached), reads the vision and
        top-level configs, creates the model on meta device, then loads
        weights from safetensors.

        Args:
            model_id_or_path: HuggingFace model ID
                (e.g. ``"meta-models/Muse-Glimmer-30B"``) or local path.
            dtype: target dtype for parameters (default: float32, suitable
                for vision encoder accuracy).
        """
        # --- 1. Download / resolve local path ---
        if os.path.isdir(model_id_or_path):
            model_dir = model_id_or_path
        else:
            model_dir = snapshot_download(
                model_id_or_path,
                allow_patterns=[
                    "*.safetensors",
                    "*.safetensors.index.json",
                    "config.json",
                ],
            )

        # --- 2. Read configs ---
        with open(os.path.join(model_dir, "config.json")) as f:
            raw_config = json.load(f)

        vision_cfg = raw_config["vision_config"]
        rope_theta = vision_cfg.get("rope_parameters", {}).get("rope_theta", 10000.0)

        # Text config for rms_norm_eps (used in perception norm)
        text_cfg = raw_config.get("text_config", {})
        rms_norm_eps = text_cfg.get("rms_norm_eps", 1e-5)

        # --- 3. Create model on meta device ---
        model = cls(
            hidden_size=vision_cfg["hidden_size"],
            intermediate_size=vision_cfg["intermediate_size"],
            num_hidden_layers=vision_cfg["num_hidden_layers"],
            num_attention_heads=vision_cfg["num_attention_heads"],
            patch_size=vision_cfg["patch_size"],
            patch_temporal=vision_cfg["patch_temporal"],
            merge_size=vision_cfg["merge_size"],
            pos_emb_height=vision_cfg["pos_emb_height"],
            pos_emb_width=vision_cfg["pos_emb_width"],
            layer_norm_eps=vision_cfg.get("layer_norm_eps", 1e-5),
            rope_theta=rope_theta,
            out_hidden_size=raw_config.get("out_hidden_size", 6144),
            projector_hidden_size=raw_config.get("projector_hidden_size", 4096),
            text_hidden_size=text_cfg.get("hidden_size", 6656),
            rms_norm_eps=rms_norm_eps,
        )
        # Meta device init then override with real weights
        model = model.to(dtype=dtype)

        # --- 4. Load vision weights from safetensors ---
        safetensors_files = _resolve_safetensors_files(model_dir)

        # Build key->file index for vision-only keys
        vision_key_to_file: dict[str, str] = {}
        for path in safetensors_files:
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():  # noqa: SIM118
                    if (
                        key.startswith("model.vision_tower.")
                        or key.startswith("model.vision_adapter.")
                        or key.startswith("model.vision_projection.")
                    ):
                        vision_key_to_file[key] = path

        # Load all vision tensors (~ 1.8B params, fits in RAM)
        raw_sd = _load_tensors_for_keys(vision_key_to_file, dtype)

        # --- 5. Remap keys to our module structure ---
        mapped_sd: dict[str, torch.Tensor] = {}
        for hf_key, tensor in raw_sd.items():
            if hf_key.startswith("model.vision_tower."):
                # model.vision_tower.X -> encoder.X
                local_key = "encoder." + hf_key.removeprefix("model.vision_tower.")
                mapped_sd[local_key] = tensor
            elif hf_key.startswith("model.vision_adapter."):
                # model.vision_adapter.fc1.weight -> projector.fc1.weight
                local_key = "projector." + hf_key.removeprefix("model.vision_adapter.")
                mapped_sd[local_key] = tensor
            elif hf_key.startswith("model.vision_projection."):
                # model.vision_projection.weight -> projector.projection.weight
                local_key = "projector.projection." + hf_key.removeprefix(
                    "model.vision_projection."
                )
                mapped_sd[local_key] = tensor

        # _rope_inv_freq is non-persistent, not in the checkpoint
        model.load_state_dict(mapped_sd, strict=False, assign=True)

        # _load_tensors_for_keys skips casting for "embedding_table" keys,
        # so force everything to the target dtype now.
        model = model.to(dtype=dtype)

        # Verify no meta-device params remain
        meta_params = [n for n, p in model.named_parameters() if p.is_meta]
        if meta_params:
            raise RuntimeError(
                f"Vision model has unloaded parameters on meta device: {meta_params}"
            )

        return model.eval()
