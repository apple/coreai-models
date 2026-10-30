# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""EmbeddingGemma 2 image encoder for Core AI export (vision tower + ``embed_vision``).

Host-prepared patches ``[B, P, 768]`` and ``position_ids [B, P, 2]`` (-1 for padding) become soft
tokens ``[B, L, 512]`` with P and L static. Padding and the 3x3 pooling slots are derived in-graph
from ``position_ids``, with no data-dependent control flow. The GPU path runs in float32 or
bfloat16; the float16 variant lives in the iOS modules.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from coreai_models.models.macos.diffusion_gemma import DiffusionGemmaDenseMLP
from coreai_models.models.macos.diffusion_gemma import DiffusionGemmaRMSNorm as RMSNorm
from coreai_models.models.macos.embedding_gemma2 import read_checkpoint_tensors, rope_inv_freq
from coreai_models.primitives.ios.rope import apply_rope
from coreai_models.primitives.macos.sdpa import SDPA

# Special token ids from the EG2 tokenizer / config.json.
EG2_BOS_TOKEN_ID = 2
EG2_EOS_TOKEN_ID = 1
EG2_PAD_TOKEN_ID = 0
EG2_BOI_TOKEN_ID = 255999
EG2_EOI_TOKEN_ID = 258882
EG2_IMAGE_TOKEN_ID = 258880


@dataclass
class EmbeddingGemma2VisionConfig:
    """Vision tower + ``embed_vision`` hyper-parameters (defaults = google/embeddinggemma-2)."""

    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 16
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
    head_dim: int = 64
    patch_size: int = 16
    pooling_kernel_size: int = 3
    position_embedding_size: int = 10240
    rms_norm_eps: float = 1e-6
    rope_theta: float = 100.0
    text_hidden_size: int = 512
    max_soft_tokens: int = 280

    @property
    def max_patches(self) -> int:
        return self.max_soft_tokens * self.pooling_kernel_size**2

    @property
    def patch_dim(self) -> int:
        return 3 * self.patch_size**2

    @classmethod
    def from_hf_config(
        cls, config: dict, max_soft_tokens: int | None = None
    ) -> EmbeddingGemma2VisionConfig:
        """Build from an EG2 ``config.json`` dict (top level, with ``vision_config``)."""
        vc = config["vision_config"]
        if vc.get("use_clipped_linears", False):
            raise NotImplementedError("clipped vision linears are not supported")
        if vc.get("standardize", False):
            raise NotImplementedError(
                "vision standardization (std_bias/std_scale) is not supported"
            )
        rope = vc.get("rope_parameters") or {}
        if rope.get("rope_type", "axial") != "axial":
            raise NotImplementedError(f"unsupported vision rope type {rope.get('rope_type')}")
        return cls(
            hidden_size=vc["hidden_size"],
            intermediate_size=vc["intermediate_size"],
            num_hidden_layers=vc["num_hidden_layers"],
            num_attention_heads=vc["num_attention_heads"],
            num_key_value_heads=vc["num_key_value_heads"],
            head_dim=vc["head_dim"],
            patch_size=vc["patch_size"],
            pooling_kernel_size=vc["pooling_kernel_size"],
            position_embedding_size=vc["position_embedding_size"],
            rms_norm_eps=vc["rms_norm_eps"],
            rope_theta=rope.get("rope_theta", 100.0),
            text_hidden_size=config["text_config"]["hidden_size"],
            max_soft_tokens=max_soft_tokens or config.get("vision_soft_tokens_per_image", 280),
        )


class VisionAttention(nn.Module):
    """Bidirectional MHA with q/k RMSNorm, scale-free v RMSNorm and axial RoPE; scale 1.0."""

    def __init__(self, config: EmbeddingGemma2VisionConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        h = config.hidden_size
        self.q_proj = nn.Linear(h, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(h, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(h, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, h, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.v_norm = RMSNorm(self.head_dim, config.rms_norm_eps, with_scale=False)
        self.sdpa = SDPA(scale=1.0, is_causal=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, n, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(b, n, self.num_heads, self.head_dim))
        k = self.k_norm(self.k_proj(x).view(b, n, self.num_kv_heads, self.head_dim))
        v = self.v_norm(self.v_proj(x).view(b, n, self.num_kv_heads, self.head_dim))
        # Axial 2-D RoPE: half of head_dim rotates by x, half by y (cos/sin from
        # axial_rope_cos_sin, laid out [fx, fx, fy, fy]).
        q = apply_rope(q, cos, sin, head_axis=2, ndim=2).transpose(1, 2)
        k = apply_rope(k, cos, sin, head_axis=2, ndim=2).transpose(1, 2)
        v = v.transpose(1, 2)
        if self.num_kv_heads != self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = self.sdpa(q, k, v, attn_mask=attn_mask)
        return self.o_proj(out.transpose(1, 2).reshape(b, n, self.num_heads * self.head_dim))


class VisionEncoderLayer(nn.Module):
    """Sandwich-norm block: ``x + post_norm(attn(pre_norm(x)))``, same for the MLP."""

    def __init__(self, config: EmbeddingGemma2VisionConfig) -> None:
        super().__init__()
        h, eps = config.hidden_size, config.rms_norm_eps
        self.self_attn = VisionAttention(config)
        self.mlp = DiffusionGemmaDenseMLP(config)
        self.input_layernorm = RMSNorm(h, eps)
        self.post_attention_layernorm = RMSNorm(h, eps)
        self.pre_feedforward_layernorm = RMSNorm(h, eps)
        self.post_feedforward_layernorm = RMSNorm(h, eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.self_attn(self.input_layernorm(x), cos, sin, attn_mask)
        x = x + self.post_attention_layernorm(h)
        h = self.mlp(self.pre_feedforward_layernorm(x))
        return x + self.post_feedforward_layernorm(h)


class VisionPatchEmbedder(nn.Module):
    """``Linear(2*(p-0.5))`` plus separable learned x/y position embeddings."""

    def __init__(self, config: EmbeddingGemma2VisionConfig) -> None:
        super().__init__()
        self.input_proj = nn.Linear(config.patch_dim, config.hidden_size, bias=False)
        self.position_embedding_table = nn.Parameter(
            torch.ones(2, config.position_embedding_size, config.hidden_size)
        )

    def forward(
        self, pixel_values: torch.Tensor, position_ids: torch.Tensor, padding: torch.Tensor
    ) -> torch.Tensor:
        x = 2.0 * (pixel_values - 0.5)
        h = self.input_proj(x.to(self.input_proj.weight.dtype))
        clamped = position_ids.clamp(min=0)
        pos = F.embedding(clamped[..., 0], self.position_embedding_table[0]) + F.embedding(
            clamped[..., 1], self.position_embedding_table[1]
        )
        pos = torch.where(padding.unsqueeze(-1), torch.zeros_like(pos), pos)
        return h + pos


class VisionEncoder(nn.Module):
    """Holds the layer stack (module path ``encoder.layers.N`` mirrors the checkpoint)."""

    def __init__(self, config: EmbeddingGemma2VisionConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )


class MultimodalEmbedder(nn.Module):
    """``embed_vision``: scale-free RMSNorm (fp32) then ``Linear(hidden -> text_hidden)``."""

    def __init__(self, config: EmbeddingGemma2VisionConfig) -> None:
        super().__init__()
        self.embedding_pre_projection_norm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, with_scale=False
        )
        self.embedding_projection = nn.Linear(
            config.hidden_size, config.text_hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.embedding_pre_projection_norm(x.float())
        w = self.embedding_projection.weight
        return F.linear(normed.to(w.dtype), w)


class EmbeddingGemma2ImageEncoder(nn.Module):
    """gemma4_vision tower + ``embed_vision``: patches -> soft tokens ``[B, L, text_hidden]``."""

    def __init__(self, config: EmbeddingGemma2VisionConfig | None = None) -> None:
        super().__init__()
        self.config = config or EmbeddingGemma2VisionConfig()
        cfg = self.config
        if cfg.head_dim % 4 != 0:
            raise ValueError("axial RoPE needs head_dim divisible by 4")
        self.patch_embedder = VisionPatchEmbedder(cfg)
        self.encoder = VisionEncoder(cfg)
        self.embed_vision = MultimodalEmbedder(cfg)
        # Axial RoPE: head_dim // 2 rotary dims per spatial axis.
        self.register_buffer(
            "rope_inv_freq", rope_inv_freq(cfg.head_dim // 2, cfg.rope_theta), persistent=False
        )
        self.register_buffer(
            "pool_slots", torch.arange(cfg.max_soft_tokens, dtype=torch.int64), persistent=False
        )

    # -- pieces (public so tests / activation probes can hook them) ---------------------

    def axial_rope_cos_sin(
        self, position_ids: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``[B, P, 2]`` positions -> cos/sin ``[B, P, head_dim]`` laid out ``[fx, fx, fy, fy]``.

        Padding rows keep their raw ``-1`` coordinates, as in HF; they are masked as keys
        and dropped by the pooler, so their values never reach a real soft token.
        """
        freqs = position_ids.unsqueeze(-1).float() * self.rope_inv_freq  # [B, P, 2, D/4]
        fx, fy = freqs[..., 0, :], freqs[..., 1, :]
        angles = torch.cat([fx, fx, fy, fy], dim=-1)
        return angles.cos().to(dtype), angles.sin().to(dtype)

    def pool(self, hidden: torch.Tensor, position_ids: torch.Tensor, padding: torch.Tensor):
        """Mean over each ``k x k`` patch block, then ``x sqrt(hidden)``; fp32 output.

        Block index = ``x//k + (max_x+1)//k * (y//k)``. With the host contract (a full
        ``(W/k) x (H/k)`` block grid, row-major, padded at the end) the real soft tokens are
        exactly the first ``num_soft_tokens`` slots.
        """
        k = self.config.pooling_kernel_size
        h = hidden.float().masked_fill(padding.unsqueeze(-1), 0.0)
        clamped = position_ids.clamp(min=0)
        max_x = clamped[..., 0].amax(dim=-1, keepdim=True) + 1
        block = torch.div(clamped, k, rounding_mode="floor")
        slot = block[..., 0] + torch.div(max_x, k, rounding_mode="floor") * block[..., 1]
        weights = (slot.unsqueeze(-1) == self.pool_slots).float() / float(k * k)  # [B, P, L]
        pooled = torch.matmul(weights.transpose(1, 2), h)  # [B, L, hidden]
        return pooled * math.sqrt(self.config.hidden_size)

    def encode_pooled(self, pixel_values: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Vision tower output before ``embed_vision``: ``[B, L, hidden]`` (fp32)."""
        cfg = self.config
        if pixel_values.shape[-2] != cfg.max_patches:
            raise ValueError(
                f"expected {cfg.max_patches} patches (budget {cfg.max_soft_tokens}), "
                f"got {pixel_values.shape[-2]}"
            )
        padding = position_ids[..., 0] < 0
        x = self.patch_embedder(pixel_values, position_ids, padding)
        cos, sin = self.axial_rope_cos_sin(position_ids, x.dtype)
        # Boolean key mask (True = attend), the convention of the repo SDPA primitive: its
        # Core AI lowering turns ``~mask`` into a -1e4 additive bias. Do not pass an additive
        # float mask here; that path would read it as a bool and invert it.
        attn_mask = torch.logical_not(padding)[:, None, None, :]
        for layer in self.encoder.layers:
            x = layer(x, cos, sin, attn_mask)
        return self.pool(x, position_ids, padding)

    def forward(self, pixel_values: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """``[B, P, patch_dim]`` + ``[B, P, 2]`` -> soft tokens ``[B, L, text_hidden]``."""
        return self.embed_vision(self.encode_pooled(pixel_values, position_ids))

    # -- loading ---------------------------------------------------------------------------

    @staticmethod
    def map_hf_key(key: str) -> str | None:
        """Checkpoint key -> module key (``None`` for keys outside the image path)."""
        if key.startswith("vision_tower."):
            local = key.removeprefix("vision_tower.")
            return local.replace(".linear.weight", ".weight")
        if key.startswith("embed_vision."):
            return key
        return None

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        dtype: torch.dtype = torch.float32,
        max_soft_tokens: int | None = None,
    ) -> EmbeddingGemma2ImageEncoder:
        """Load from a local EG2 snapshot directory (``config.json`` + ``*.safetensors``)."""
        config = json.loads((Path(model_dir) / "config.json").read_text())
        model = cls(EmbeddingGemma2VisionConfig.from_hf_config(config, max_soft_tokens))
        tensors = read_checkpoint_tensors(model_dir, ("vision_tower.", "embed_vision."), dtype)
        model.load_state_dict(
            {local: v for k, v in tensors.items() if (local := cls.map_hf_key(k)) is not None}
        )
        return model.to(dtype).eval()


def aspect_ratio_preserving_size(
    height: int,
    width: int,
    patch_size: int = 16,
    max_soft_tokens: int = 280,
    pooling_kernel_size: int = 3,
) -> tuple[int, int]:
    """Target ``(height, width)``: the largest size with sides divisible by ``k*patch`` and at
    most ``max_soft_tokens * k^2`` patches, preserving aspect ratio (Gemma4ImageProcessor)."""
    max_patches = max_soft_tokens * pooling_kernel_size**2
    target_px = max_patches * patch_size**2
    factor = math.sqrt(target_px / (height * width))
    side = pooling_kernel_size * patch_size
    th = int(math.floor(factor * height / side)) * side
    tw = int(math.floor(factor * width / side)) * side
    if th == 0 and tw == 0:
        raise ValueError(f"cannot resize {height}x{width} to a non-empty grid")
    max_side = (max_patches // pooling_kernel_size**2) * side
    if th == 0:
        th = side
        tw = min(int(math.floor(width / height)) * side, max_side)
    elif tw == 0:
        tw = side
        th = min(int(math.floor(height / width)) * side, max_side)
    if th * tw > target_px:
        raise ValueError(f"resize {height}x{width} -> {th}x{tw} exceeds {max_patches} patches")
    return th, tw


def patchify(image: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    """``[3, H, W]`` -> ``[(H/p)*(W/p), p*p*3]``; patches row-major, pixels ``(py, px, c)``."""
    c, h, w = image.shape
    ph, pw = h // patch_size, w // patch_size
    x = image.reshape(c, ph, patch_size, pw, patch_size).permute(1, 3, 2, 4, 0)
    return x.reshape(ph * pw, patch_size * patch_size * c)


def patch_position_ids(patch_height: int, patch_width: int) -> torch.Tensor:
    """Row-major ``(x, y)`` coordinates ``[ph*pw, 2]``: ``(0,0), (1,0), ..., (pw-1, ph-1)``."""
    ys, xs = torch.meshgrid(torch.arange(patch_height), torch.arange(patch_width), indexing="ij")
    return torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)


def pad_to_budget(
    patches: torch.Tensor, positions: torch.Tensor, max_patches: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-pad patches and ``-1``-pad positions along dim 0 to ``max_patches``."""
    n = patches.shape[0]
    if n > max_patches:
        raise ValueError(f"{n} patches exceed the budget of {max_patches}")
    pad = max_patches - n
    patches = F.pad(patches, (0, 0, 0, pad), value=0.0)
    positions = F.pad(positions, (0, 0, 0, pad), value=-1)
    return patches, positions


def preprocess_resized_image(
    image: torch.Tensor,
    patch_size: int = 16,
    max_soft_tokens: int = 280,
    pooling_kernel_size: int = 3,
    rescale_factor: float = 1.0 / 255.0,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Already-resized ``[3, H, W]`` RGB (uint8 or 0..255 float) -> model inputs.

    Returns ``(pixel_values [1, P, patch_dim] fp32, position_ids [1, P, 2] int64,
    num_soft_tokens)``. Rescale only (no mean/std normalization: the tower does
    ``2*(x-0.5)`` itself).
    """
    _, h, w = image.shape
    side = patch_size * pooling_kernel_size
    if h % side or w % side:
        raise ValueError(f"resized image {h}x{w} must have sides divisible by {side}")
    x = image.to(torch.float32) * rescale_factor
    patches = patchify(x, patch_size)
    positions = patch_position_ids(h // patch_size, w // patch_size)
    num_soft_tokens = patches.shape[0] // pooling_kernel_size**2
    patches, positions = pad_to_budget(patches, positions, max_soft_tokens * pooling_kernel_size**2)
    return patches.unsqueeze(0), positions.unsqueeze(0), num_soft_tokens


def build_image_input_ids(
    num_soft_tokens: int,
    *,
    bos_token_id: int | None = EG2_BOS_TOKEN_ID,
    eos_token_id: int | None = EG2_EOS_TOKEN_ID,
    boi_token_id: int = EG2_BOI_TOKEN_ID,
    eoi_token_id: int = EG2_EOI_TOKEN_ID,
    image_token_id: int = EG2_IMAGE_TOKEN_ID,
) -> list[int]:
    """Ids for an image-only input, as the HF processor emits them:
    ``<bos> <boi> <image> * num_soft_tokens <eoi> <eos>``."""
    ids = [boi_token_id] + [image_token_id] * num_soft_tokens + [eoi_token_id]
    if bos_token_id is not None:
        ids = [bos_token_id] + ids
    if eos_token_id is not None:
        ids = ids + [eos_token_id]
    return ids


def scatter_soft_tokens(
    inputs_embeds: torch.Tensor, placeholder_mask: torch.Tensor, soft_tokens: torch.Tensor
) -> torch.Tensor:
    """Replace placeholder rows with soft tokens, in order (``masked_scatter`` semantics).

    Gather-based so it exports at a static length: the ``j``-th placeholder in row ``b``
    takes ``soft_tokens[b, j]``. ``soft_tokens`` may carry extra trailing rows (e.g. the
    zero padding slots of a fixed budget); they are never read.

    Args:
        inputs_embeds: ``[B, S, D]`` text embeddings.
        placeholder_mask: ``[B, S]`` bool.
        soft_tokens: ``[B, T, D]`` with ``T >=`` placeholders per row.
    """
    idx = torch.cumsum(placeholder_mask.to(torch.int64), dim=1) - 1
    idx = idx.clamp(min=0, max=soft_tokens.shape[1] - 1)
    gathered = torch.gather(
        soft_tokens, 1, idx.unsqueeze(-1).expand(-1, -1, soft_tokens.shape[-1])
    ).to(inputs_embeds.dtype)
    return torch.where(placeholder_mask.unsqueeze(-1), gathered, inputs_embeds)


def build_image_inputs_embeds(
    embed_tokens: Callable[[torch.Tensor], torch.Tensor],
    soft_tokens: torch.Tensor,
    num_soft_tokens: int | Sequence[int],
    *,
    image_token_id: int = EG2_IMAGE_TOKEN_ID,
    pad_token_id: int = EG2_PAD_TOKEN_ID,
    **id_kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full text-encoder input for one image per row.

    ``embed_tokens`` maps ids -> *scaled* text embeddings (EG2 multiplies the table by
    ``sqrt(hidden)``; HF ``language_model.embed_tokens`` already does). Placeholders are
    looked up as ``<pad>`` (like HF) and then overwritten by the soft tokens.

    Returns ``(inputs_embeds [B, S, D], attention_mask [B, S] int64, input_ids [B, S])``,
    right-padded to the longest row.
    """
    counts = [num_soft_tokens] if isinstance(num_soft_tokens, int) else list(num_soft_tokens)
    if len(counts) != soft_tokens.shape[0]:
        raise ValueError("one num_soft_tokens per soft_tokens row")
    rows = [build_image_input_ids(n, image_token_id=image_token_id, **id_kwargs) for n in counts]
    s = max(len(r) for r in rows)
    input_ids = torch.full((len(rows), s), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), s), dtype=torch.long)
    for i, r in enumerate(rows):
        input_ids[i, : len(r)] = torch.tensor(r)
        attention_mask[i, : len(r)] = 1
    placeholder = input_ids == image_token_id
    lookup = torch.where(placeholder, torch.full_like(input_ids, pad_token_id), input_ids)
    embeds = embed_tokens(lookup.to(soft_tokens.device))
    embeds = scatter_soft_tokens(embeds, placeholder.to(embeds.device), soft_tokens)
    return embeds, attention_mask, input_ids
