# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""CLI entry point for ``coreai.vlm.export``.

Exports a vision-language model to Core AI format as a multi-asset bundle
(``<name>/``):

  - ``<name>.aimodel``   text decoder (asset role ``main``, inputs_embeds, stateful KV)
  - ``embed.aimodel``    token-embedding lookup (asset role ``embedding``)
  - ``vision.aimodel``   vision encoder (asset role ``vision``, 448x448 static shapes)
  - ``tokenizer/``       embedded HF tokenizer
  - ``metadata.json``    bundle manifest (``kind=vlm``)

Usage:
    uv run coreai.vlm.export qwen3-vl [--max-context-length 4096] [--num-layers N]
    uv run coreai.vlm.export muse-glimmer-vl [--max-context-length 4096]
    uv run coreai.vlm.export qwen3.5-0.8b --vision-only
    uv run coreai.vlm.export --list-models
"""

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

from coreai_models._constants import DEFAULT_INCLUDE_DEBUG_INFO
from coreai_models.export.macos import export_to_coreai
from coreai_models.export.metadata import build_aimodel_metadata
from coreai_models.models.macos.muse_glimmer import MuseGlimmerForCausalLMEmbeddings
from coreai_models.models.macos.muse_glimmer_vision import MuseGlimmerVisionModel
from coreai_models.models.macos.qwen3_vl import Qwen3VLForCausalLMEmbeddings

# Core AI state names for the persistent KV cache.
KV_STATE_NAMES = ("k_cache", "v_cache")


@dataclass(frozen=True)
class VLMSpec:
    """Per-model export recipe, keyed by registry short-name.

    Carries the bits that vary between VL checkpoints: the HF id, the output
    bundle name, the reauthored text-decoder class, and the vision geometry
    (resolution, patch/merge sizes, the image placeholder token, and CLIP
    normalization stats) that drive both the vision-encoder export and the
    ``vision`` block of ``metadata.json``.
    """

    short_name: str
    hf_model_id: str
    output_name: str
    #: Reauthored decoder the text bundle is built from, or ``None`` for a
    #: checkpoint whose decoder has no recipe here — then only the vision
    #: encoder can be exported.
    text_decoder_class: type | None
    image_token_id: int
    image_size: int
    patch_size: int
    spatial_merge_size: int
    temporal_patch_size: int
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]
    rescale_factor: float
    image_strategy: str = "stretch"
    include_image_info: bool = False

    @property
    def num_visual_tokens(self) -> int:
        """Visual tokens after spatial merge, e.g. (448/16/2)**2 = 196."""
        return (self.image_size // self.patch_size // self.spatial_merge_size) ** 2


SUPPORTED_MODELS: dict[str, VLMSpec] = {
    "qwen3-vl": VLMSpec(
        short_name="qwen3-vl",
        hf_model_id="Qwen/Qwen3-VL-2B-Instruct",
        output_name="qwen3_vl_2b",
        text_decoder_class=Qwen3VLForCausalLMEmbeddings,
        image_token_id=151655,  # <|image_pad|>
        image_size=448,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,  # Qwen frames-per-image (single image -> duplicated)
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        rescale_factor=1.0,
        image_strategy="stretch",
        include_image_info=True,
    ),
    "muse-glimmer-vl": VLMSpec(
        short_name="muse-glimmer-vl",
        hf_model_id="meta-models/Muse-Glimmer-30B",
        output_name="muse_glimmer_30b_vlm",
        text_decoder_class=MuseGlimmerForCausalLMEmbeddings,
        image_token_id=200092,
        image_size=448,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        image_mean=(0.48145466, 0.4578275, 0.40821073),  # CLIP normalization
        image_std=(0.26862954, 0.26130258, 0.27577711),
        rescale_factor=1.0,
    ),
    "qwen3.5-0.8b": VLMSpec(
        short_name="qwen3.5-0.8b",
        hf_model_id="Qwen/Qwen3.5-0.8B",
        output_name="qwen3_5_0p8b",
        text_decoder_class=None,
        image_token_id=248056,  # <|image_pad|>
        image_size=448,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        rescale_factor=1.0,
        image_strategy="stretch",
        include_image_info=True,
    ),
}


# ---------------------------------------------------------------------------
# Text decoder: direct safetensors loader
# (avoids the hf_memory_efficient layer-regex issue)
# ---------------------------------------------------------------------------


def _get_safetensors_files(model_dir: str) -> list[str]:
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            idx = json.load(f)
        shards = sorted(set(idx["weight_map"].values()))
        return [os.path.join(model_dir, s) for s in shards]
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        return [single]
    raise FileNotFoundError(f"No safetensors in {model_dir}")


def load_model_from_safetensors(
    model_class: type,
    hf_config,
    model_dir: str,
    max_ctx: int,
    num_layers: int | None,
    dtype: torch.dtype = torch.float16,
) -> torch.nn.Module:
    """Load a VL text decoder directly from safetensors, bypassing from_hf_memory_efficient.

    Handles two checkpoint layouts:
      - Qwen3-VL: ``model.language_model.*`` (text), ``model.visual.*`` (vision, skipped)
      - Muse Glimmer: ``model.language_model.*`` (text), ``model.vision_*`` (vision, skipped),
        plus a top-level ``lm_head.weight`` outside the language_model prefix.
    """
    # Set config
    text_cfg = model_class._get_reauthored_config(hf_config, max_ctx, num_layers)

    # Create model on meta device
    model = model_class(text_cfg, model_device="meta")
    model.to(dtype=dtype)

    # Build state dict from safetensors
    prefix = "model.language_model."
    # Prefixes to skip (vision components from various checkpoint formats)
    vision_prefixes = (
        "model.visual.",
        "model.vision_tower.",
        "model.vision_adapter.",
        "model.vision_projection.",
    )
    layer_pattern = re.compile(r"layers\.(\d+)\.")
    st_files = _get_safetensors_files(model_dir)

    state_dict: dict[str, torch.Tensor] = {}
    for path in st_files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118 — safe_open has no __iter__/__contains__
                if any(key.startswith(vp) for vp in vision_prefixes):
                    continue
                if key.startswith(prefix):
                    # Strip "model.language_model." → add "model."
                    stripped = key[len(prefix) :]
                    model_key = "model." + stripped
                    # Skip layers beyond num_layers
                    m = layer_pattern.match(stripped)
                    if m and num_layers is not None and int(m.group(1)) >= num_layers:
                        continue
                    tensor = f.get_tensor(key)
                    if tensor.dtype not in (torch.float16, torch.int8) and "zero_point" not in key:
                        tensor = tensor.to(dtype)
                    state_dict[model_key] = tensor
                elif key == "lm_head.weight":
                    # Muse Glimmer stores lm_head outside the language_model prefix
                    tensor = f.get_tensor(key)
                    if tensor.dtype not in (torch.float16, torch.int8):
                        tensor = tensor.to(dtype)
                    state_dict[key] = tensor

    # Fuse weights via _mutate_state_dict (handles keys in "model.layers.N.*" form)
    model._mutate_state_dict(state_dict)

    # Load (strict=False to allow tie_word_embeddings / missing embed_tokens)
    model.load_state_dict(state_dict, assign=True, strict=False)

    # Muse Glimmer: qk_norm has no checkpoint weights — initialize to ones
    # (identity RMSNorm), matching from_hf_memory_efficient in muse_glimmer.py.
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None and hasattr(attn, "qk_norm"):
                w = attn.qk_norm.weight
                if w.is_meta:
                    head_dim = getattr(model.config, "head_dim", None)
                    if head_dim is not None:
                        attn.qk_norm.weight = nn.Parameter(torch.ones(head_dim, dtype=dtype))

    # Verify no meta params remain
    meta = [n for n, p in model.named_parameters() if p.is_meta]
    if meta:
        raise RuntimeError(f"Parameters not loaded: {meta}")

    return model


# ---------------------------------------------------------------------------
# embed.aimodel: token-embedding lookup component
# ---------------------------------------------------------------------------


class EmbedTokens(torch.nn.Module):
    """Token-embedding lookup, exported as the bundle's `embedding` component.

    Mirrors the float path of ``primitives.ios.embedding.GatherEmbeddings``
    (``table[input_ids]``), which lowers cleanly with Int32 indices — unlike
    ``nn.Embedding``, whose gather requires Int64 indices the runtime won't feed.

    Input:  input_ids   int32 [1, seq_len]
    Output: embeddings   f16  [1, seq_len, hidden_size]
    """

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[input_ids]


class EmbedTokensNormed(torch.nn.Module):
    """Token-embedding lookup with weight-less RMSNorm, for Muse Glimmer.

    Muse Glimmer applies a weight-less RMSNorm to embeddings before they enter
    the transformer. For VLM, this must happen in the embed asset so that text
    and vision embeddings are in the same normalized space after scatter-merge.

    Input:  input_ids   int32 [1, seq_len]
    Output: embeddings   f16  [1, seq_len, hidden_size] (normalized)
    """

    def __init__(self, weight: torch.Tensor, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.eps = eps

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.weight[input_ids]
        return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)


def _load_embed_weight(model_dir: str) -> torch.Tensor:
    """Read the f16 embed_tokens weight table [vocab, hidden] from safetensors."""
    embed_key = "model.language_model.embed_tokens.weight"
    for path in _get_safetensors_files(model_dir):
        with safe_open(path, framework="pt", device="cpu") as f:
            if embed_key in f.keys():  # noqa: SIM118 — safe_open has no __contains__
                return f.get_tensor(embed_key).to(torch.float16)
    raise RuntimeError(f"embed_tokens not found in safetensors (looked for '{embed_key}')")


async def export_embed_model(
    spec: VLMSpec,
    bundle_path: Path,
    model_dir: str,
    max_ctx: int,
    overwrite: bool,
    include_debug_info: bool = DEFAULT_INCLUDE_DEBUG_INFO,
) -> str:
    """Export the token-embedding lookup as embed.aimodel (asset role `embedding`)."""
    weight = _load_embed_weight(model_dir)
    vocab_size, hidden_size = weight.shape
    if spec.short_name == "muse-glimmer-vl":
        module = EmbedTokensNormed(weight, eps=1e-5).eval()
    else:
        module = EmbedTokens(weight).eval()

    seq_len = 64
    input_ids = torch.zeros(1, seq_len, dtype=torch.int32)
    program = export_to_coreai(
        module,
        {"input_ids": input_ids},
        dynamic_shapes={"input_ids": {1: torch.export.Dim("embed_seq", max=max_ctx - 1)}},
        input_names=("input_ids",),
        output_names=("embeddings",),
        state_names=None,
        include_debug_info=include_debug_info,
    )

    embed_path = bundle_path / "embed.aimodel"
    if embed_path.exists():
        if not overwrite:
            raise FileExistsError(f"{embed_path} exists. Use --overwrite.")
        shutil.rmtree(embed_path)
    meta = build_aimodel_metadata(spec.hf_model_id)
    await asyncio.to_thread(program.save_asset, embed_path, meta)
    logging.info(f"Saved embed.aimodel: {vocab_size} × {hidden_size} × f16")
    return "embed.aimodel"


# ---------------------------------------------------------------------------
# Text decoder bundle (text decoder + embed + tokenizer + metadata.json)
# ---------------------------------------------------------------------------


def _vision_metadata(spec: VLMSpec) -> dict:
    """The top-level ``vision`` block of metadata.json.

    Consumed by Swift ``VisionConfig``, hence the snake_case keys.
    """
    return {
        "image_size": spec.image_size,
        "patch_size": spec.patch_size,
        "image_token_count": spec.num_visual_tokens,
        "image_token_id": spec.image_token_id,
        "image_mean": list(spec.image_mean),
        "image_std": list(spec.image_std),
        "rescale_factor": spec.rescale_factor,
        "image_strategy": spec.image_strategy,
        "include_image_info": spec.include_image_info,
    }


async def export_text_bundle(
    spec: VLMSpec,
    *,
    max_ctx: int,
    num_layers: int | None,
    output_dir: Path,
    overwrite: bool,
    compression: str = "none",
    include_debug_info: bool = DEFAULT_INCLUDE_DEBUG_INFO,
) -> Path:
    """Download weights and write the text portion of the VLM bundle.

    Produces ``<name>.aimodel`` (decoder), ``embed.aimodel``, ``tokenizer/``, and
    a ``metadata.json`` whose ``assets`` cover ``main``/``embedding``. The
    ``vision`` asset is added later by :func:`export_vision_encoder`.

    Raises ``ValueError`` for a spec without a text decoder, before anything is
    downloaded or loaded.
    """
    decoder_class = spec.text_decoder_class
    if decoder_class is None:
        raise ValueError(
            f"'{spec.short_name}' has no text decoder recipe; it supports vision-encoder "
            f"export only. Run with --vision-only."
        )

    output_name = spec.output_name

    # ---- 1. Download weights + load config ----
    logging.info(f"Downloading {spec.hf_model_id}...")
    model_dir = snapshot_download(
        spec.hf_model_id,
        allow_patterns=[
            "*.safetensors",
            "*.safetensors.index.json",
            "config.json",
            "tokenizer*",
            "vocab.json",
            "merges.txt",
            "*.model",
        ],
    )
    raw_cfg = AutoConfig.from_pretrained(model_dir)
    text_cfg = raw_cfg.text_config
    hidden_size = text_cfg.hidden_size
    vocab_size = text_cfg.vocab_size
    logging.info(f"Text config: hidden={hidden_size}, vocab={vocab_size}, ctx={max_ctx}")

    # ---- 2. Load model directly from safetensors ----
    logging.info("Loading model from safetensors (direct, skips vision encoder)...")
    model = load_model_from_safetensors(
        model_class=decoder_class,
        hf_config=raw_cfg,
        model_dir=model_dir,
        max_ctx=max_ctx,
        num_layers=num_layers,
        dtype=torch.float16,
    )
    model = model.eval()
    logging.info("Model loaded.")

    # ---- 3. Apply quantization (text decoder only) ----
    if compression != "none":
        from coreai_models.export.compression import quantize_for_export
        from coreai_models.export.presets import get_preset

        preset = get_preset(compression)
        quant_cfg = preset.get("torch_quantization_config")
        if quant_cfg is None:
            raise ValueError(
                f"Preset '{compression}' has no torch_quantization_config. "
                "VLM text decoder quantization requires a macOS quantization preset."
            )
        quant_cfg = dict(quant_cfg)
        logging.info(f"Applying {compression} quantization to text decoder...")
        model = quantize_for_export(model, text_cfg, torch.float16, quant_cfg)
        logging.info("Quantization complete.")

    # ---- 4. Build reference inputs (stateful KV: caches are in-place states) ----
    QUERY_LEN = 64
    OFFSET = 64
    inputs_embeds = torch.randn(1, QUERY_LEN, hidden_size, dtype=torch.float16)
    position_ids = torch.arange(QUERY_LEN + OFFSET, dtype=torch.int32).unsqueeze(0)

    n_layers = num_layers or text_cfg.num_hidden_layers
    n_kv_heads = text_cfg.num_key_value_heads
    head_dim = getattr(text_cfg, "head_dim", text_cfg.hidden_size // text_cfg.num_attention_heads)
    k_cache = torch.zeros(n_layers, 1, n_kv_heads, max_ctx, head_dim, dtype=torch.float16)
    v_cache = torch.zeros(n_layers, 1, n_kv_heads, max_ctx, head_dim, dtype=torch.float16)

    reference_inputs = {
        "inputs_embeds": inputs_embeds,
        "position_ids": position_ids,
        "k_cache": k_cache,
        "v_cache": v_cache,
    }
    dynamic_shapes = {
        "inputs_embeds": {1: torch.export.Dim("query_len", max=max_ctx - 2)},
        "position_ids": {1: torch.export.Dim("seq_pos", min=QUERY_LEN, max=max_ctx - 1)},
        "k_cache": None,  # fixed size
        "v_cache": None,
    }

    # ---- 5. Export (stateful KV: k_cache/v_cache surfaced as in-place states) ----
    logging.info("Exporting text decoder to CoreAI format (stateful KV)...")
    program = export_to_coreai(
        model,
        reference_inputs,
        dynamic_shapes=dynamic_shapes,
        input_names=("inputs_embeds", "position_ids"),
        output_names=("logits",),
        state_names=KV_STATE_NAMES,
        include_debug_info=include_debug_info,
    )

    # ---- 6. Save bundle ----
    bundle_path = output_dir / output_name
    bundle_path.mkdir(parents=True, exist_ok=True)
    aimodel_path = bundle_path / f"{output_name}.aimodel"

    if aimodel_path.exists() and not overwrite:
        raise FileExistsError(f"{aimodel_path} exists. Use --overwrite.")
    elif aimodel_path.exists():
        shutil.rmtree(aimodel_path)

    logging.info(f"Saving model to {aimodel_path}...")
    meta = build_aimodel_metadata(spec.hf_model_id)
    await asyncio.to_thread(program.save_asset, aimodel_path, meta)
    del model

    # ---- 6. Embed model ----
    logging.info("Exporting embed.aimodel...")
    embed_rel = await export_embed_model(
        spec, bundle_path, model_dir, max_ctx, overwrite, include_debug_info=include_debug_info
    )

    # ---- 7. Tokenizer ----
    logging.info("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.save_pretrained(str(bundle_path / "tokenizer"))

    # ---- 8. metadata.json ----
    # Asset roles match Swift ModelBundle.ComponentKey: `main` (decoder),
    # `embedding` (embed.aimodel), `vision` (added by export_vision_encoder).
    metadata = {
        "metadata_version": "0.2",
        "kind": "vlm",
        "name": output_name,
        "assets": {
            "main": f"{output_name}.aimodel",
            "embedding": embed_rel,
        },
        "language": {
            "tokenizer": spec.hf_model_id,
            "vocab_size": vocab_size,
            "max_context_length": max_ctx,
            "embedded_tokenizer": True,
            "function_map": {"main": ["main"]},
        },
        "vision": _vision_metadata(spec),
        "source": {
            "hf_model_id": spec.hf_model_id,
            "model_definition": "torch",
        },
    }
    with open(bundle_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logging.info(f"Text bundle complete: {bundle_path}")
    return bundle_path


# ---------------------------------------------------------------------------
# Vision encoder (448x448 fixed input, fully-static shapes)
# ---------------------------------------------------------------------------


def _f16_attention_forward(
    self, hidden_states, cu_seqlens=None, position_embeddings=None, **kwargs
):
    """Dtype-preserving eager attention for Qwen3VLVisionAttention.

    HF's eager path upcasts to fp32 inline twice per layer (RoPE's
    q/k.float() and softmax(dtype=fp32)); the ANE runs f16 only, so each
    upcast cuts the compiled graph — 2 cuts x 24 layers left 49 ANE regions
    whose boundary traffic made the hybrid slower than pure GPU. Keeping the
    whole layer in the input dtype lets the graph compile to a single region.

    cos/sin arrive pre-tiled to [seq, heads, head_dim] (exact match, zero
    broadcast); the 1/sqrt(head_dim) scale is folded into q right after RoPE
    so the [1, H, S, S] score tensor stays bounded for the f16 softmax.
    """
    seq_length = hidden_states.shape[0]
    qkv = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3)
    q, k, v = qkv.unbind(0)  # [S, H, D] each

    cos, sin = position_embeddings  # pre-tiled [S, H, D]
    half = q.shape[-1] // 2

    def rotate_half(x):
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    q = (q * cos + rotate_half(q) * sin) * self.scaling
    k = k * cos + rotate_half(k) * sin

    q = q.transpose(0, 1).unsqueeze(0)  # [1, H, S, D]
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)

    scores = torch.matmul(q, k.transpose(2, 3))  # [1, H, S, S]
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)  # [1, H, S, D]
    out = out.transpose(1, 2).reshape(seq_length, -1)  # [S, H*D]
    return self.proj(out)


class StaticVisionEncoder(nn.Module):
    """Vision encoder with pre-computed static position embeddings for a fixed grid.

    Avoids all data-dependent operations (linspace, repeat_interleave, etc.)
    by baking in the constant values at init time for the spec's resolution.

    Accepts raw CHW pixel values (the layout the Swift runner's ImagePreprocessor
    produces) and reproduces the Qwen image-processor patchify internally, so the
    runner needs no Qwen-specific preprocessing beyond resize + normalize.

    Single-image (num_frames=1):
      Input:  pixel_values  float32 [1, 3, image_size, image_size]
      Output: image_features float32 [num_visual_tokens, text_hidden]

    Multi-frame (num_frames>1, must be divisible by temporal_patch_size):
      Input:  pixel_values  float32 [1, 3*num_frames, image_size, image_size]
      Output: image_features float32 [grid_t * num_visual_tokens, text_hidden]
    """

    def __init__(
        self,
        visual_model,
        *,
        image_size: int,
        patch_size: int,
        spatial_merge_size: int,
        temporal_patch_size: int,
        num_frames: int = 1,
        patchified_input: bool = False,
        linear_patch_embed: bool = False,
        f16_attention: bool = False,
    ) -> None:
        super().__init__()
        self.patch_embed = visual_model.patch_embed
        self.blocks = visual_model.blocks
        self.merger = visual_model.merger
        self.patchified_input = patchified_input

        if f16_attention:
            # Class-level patch (same idiom as _patch_fast_pos_embed_interpolate):
            # torch.export is only guaranteed to trace the class forward, so an
            # instance-level override could silently fall back to HF's fp32 path.
            # Resolved from the instance so every qwen-vl-family tower works.
            type(visual_model.blocks[0].attn).forward = _f16_attention_forward

        self.image_size = image_size
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.channels = 3
        self.num_frames = num_frames

        if num_frames == 1:
            self.grid_t = 1
        elif num_frames % temporal_patch_size != 0:
            raise ValueError(
                f"num_frames ({num_frames}) must be divisible by "
                f"temporal_patch_size ({temporal_patch_size})"
            )
        else:
            self.grid_t = num_frames // temporal_patch_size
        self.grid_h = image_size // patch_size
        self.grid_w = image_size // patch_size
        self.num_patches = self.grid_t * self.grid_h * self.grid_w
        self.patch_dim = temporal_patch_size * self.channels * patch_size * patch_size

        grid_thw = torch.tensor([[self.grid_t, self.grid_h, self.grid_w]], dtype=torch.int32)

        with torch.no_grad():
            pos_embeds = visual_model.fast_pos_embed_interpolate(grid_thw)
            self.register_buffer("pos_embeds", pos_embeds)

            rotary_pos_emb = visual_model.rot_pos_emb(grid_thw)
            seq_len = rotary_pos_emb.shape[0]
            rotary_flat = rotary_pos_emb.reshape(seq_len, -1)
            emb = torch.cat([rotary_flat, rotary_flat], dim=-1)
            rot_cos, rot_sin = emb.cos(), emb.sin()
            if f16_attention:
                # Pre-tile to [seq, heads, head_dim] so the f16 attention applies
                # RoPE with exact-shape elementwise ops (no runtime broadcast).
                num_heads = visual_model.blocks[0].attn.num_heads
                rot_cos = rot_cos.reshape(seq_len, 1, -1).expand(-1, num_heads, -1).contiguous()
                rot_sin = rot_sin.reshape(seq_len, 1, -1).expand(-1, num_heads, -1).contiguous()
            self.register_buffer("rot_cos", rot_cos)
            self.register_buffer("rot_sin", rot_sin)

            total_patches = self.grid_t * self.grid_h * self.grid_w
            cu = torch.tensor([0, total_patches], dtype=torch.int32)
            self.register_buffer("cu_seqlens", cu)

        if linear_patch_embed:
            # Conv3d with kernel_size == stride sees exactly one window per output
            # position, so it is an exact Linear over the flattened [c, t, p, p]
            # patch vector — the form the ANE compiler can map.
            proj = visual_model.patch_embed.proj
            linear = nn.Linear(self.patch_dim, proj.out_channels, bias=proj.bias is not None)
            with torch.no_grad():
                linear.weight.copy_(proj.weight.reshape(proj.out_channels, -1))
                if proj.bias is not None:
                    linear.bias.copy_(proj.bias)
            self.patch_embed = linear

    def _patchify(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Turn pixels into Qwen's pre-patchified [num_patches, patch_dim].

        Single-image: [1, 3, H, W] → duplicate across temporal dim.
        Multi-frame:  [1, 3*N, H, W] → reshape real frames.
        """
        c, patch, merge = self.channels, self.patch_size, self.spatial_merge_size
        hw = self.image_size

        if self.num_frames == 1:
            x = pixel_values.reshape(c, hw, hw)
            x = x.unsqueeze(0).repeat(self.temporal_patch_size, 1, 1, 1)
        else:
            # [1, 3*N, H, W] → [N, 3, H, W]
            x = pixel_values.reshape(self.num_frames, c, hw, hw)

        # [N, C, H, W] → split H,W into (grid, merge, patch), T into (grid_t, temporal)
        x = x.reshape(
            self.grid_t,
            self.temporal_patch_size,
            c,
            self.grid_h // merge,
            merge,
            patch,
            self.grid_w // merge,
            merge,
            patch,
        )
        x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        return x.reshape(self.num_patches, self.patch_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [1, 3, H, W] (NCHW) → patchify → [num_patches, patch_dim];
        # patchified_input: [1, num_patches, patch_dim] fed directly (the rank-9
        # patchify reshape exceeds the ANE compiler's rank-5 limit, so ANE-friendly
        # exports move it host-side).
        if self.patchified_input:
            patches = pixel_values.reshape(self.num_patches, self.patch_dim)
        else:
            patches = self._patchify(pixel_values)
        hidden_states = self.patch_embed(patches)  # [num_patches, vision_hidden]
        hidden_states = hidden_states + self.pos_embeds

        position_embeddings = (self.rot_cos, self.rot_sin)

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=self.cu_seqlens,
                position_embeddings=position_embeddings,
            )

        # merger pixel_shuffle → [num_visual_tokens, text_hidden]
        return self.merger(hidden_states)


class BatchedF16VisionEncoder(nn.Module):
    """Conform the encoder output to the runner contract shared with embed/main.

    StaticVisionEncoder emits [num_visual_tokens, text_hidden]; PR #65 expects
    f16/bf16 [1, image_token_count, hidden] (a leading batch dim, like embed.aimodel).
    With `input_cast` the pixels are cast at the graph entry so the encoder can run
    in that dtype throughout (fp16 math is required for ANE mapping); without it the
    vision math stays f32 and only the final result is cast.
    """

    def __init__(self, encoder: nn.Module, input_cast: torch.dtype | None = None) -> None:
        super().__init__()
        self.encoder = encoder
        self.input_cast = input_cast

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.input_cast is not None:
            pixel_values = pixel_values.to(self.input_cast)
        out = self.encoder(pixel_values)
        if isinstance(out, tuple):
            out = out[0]
        return out.unsqueeze(0).to(torch.float16)


class CastF16VisionEncoder(nn.Module):
    """Cast-only wrapper for vision encoders that already output [1, N, hidden].

    MuseGlimmerVisionModel.forward returns [1, N, text_hidden] in f32 — the
    batch dimension is already present, so we only need the f16 cast.
    """

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.encoder(pixel_values)
        if isinstance(out, tuple):
            out = out[0]
        return out.to(torch.float16)


def _patch_fast_pos_embed_interpolate(vision_model_cls: type) -> None:
    """Monkeypatch to use Python ints — needed for the init-time pre-computation."""

    def patched(self, grid_thw):
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for _t, h, w in zip(grid_ts.tolist(), grid_hs.tolist(), grid_ws.tolist(), strict=False):
            h, w = int(h), int(w)
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)
            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor
            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side
            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=self.pos_embed.weight.device)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype, device=self.pos_embed.weight.device
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        hw_pairs = [
            (int(h), int(w)) for h, w in zip(grid_hs.tolist(), grid_ws.tolist(), strict=False)
        ]
        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in hw_pairs])

        merge_size = self.config.spatial_merge_size
        patch_pos_embeds_permute = []
        for pos_embed, t, (h, w) in zip(patch_pos_embeds, grid_ts.tolist(), hw_pairs, strict=False):
            t = int(t)
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        return torch.cat(patch_pos_embeds_permute)

    vision_model_cls.fast_pos_embed_interpolate = patched


async def export_vision_encoder(
    spec: VLMSpec,
    bundle_path: Path,
    overwrite: bool,
    num_frames: int = 1,
    include_debug_info: bool = DEFAULT_INCLUDE_DEBUG_INFO,
    vision_dtype: str = "f32",
    ane_friendly: bool = False,
    vision_compression: str = "none",
) -> str:
    """Export the vision encoder as vision.aimodel and patch metadata.json."""
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}. Export the text decoder first.")

    # ---- 1. Text hidden size (projection target) from the HF config ----
    text_hidden = AutoConfig.from_pretrained(spec.hf_model_id).text_config.hidden_size

    if spec.short_name == "muse-glimmer-vl":
        # Muse Glimmer: use our standalone MuseGlimmerVisionModel which
        # handles 2D patchify, 2D RoPE, spatial merge, adapter, and
        # projection internally.  Single-image only (num_frames=1).
        if num_frames != 1:
            raise NotImplementedError(
                "Muse Glimmer vision encoder currently supports single-image only "
                f"(num_frames=1), got {num_frames}"
            )
        if ane_friendly or vision_dtype != "f32":
            raise NotImplementedError(
                "The ANE-friendly form and the f16 vision graph are built around "
                "StaticVisionEncoder, which Muse Glimmer does not go through"
            )
        logging.info(f"Loading {spec.hf_model_id} vision encoder (MuseGlimmerVisionModel)...")
        vision_model = MuseGlimmerVisionModel.from_pretrained(spec.hf_model_id, dtype=torch.float32)
        # MuseGlimmerVisionModel.forward already returns [1, N, text_hidden]
        # so we only need the f16 cast wrapper (no unsqueeze needed).
        export_module = CastF16VisionEncoder(vision_model).eval()
        num_visual_tokens = spec.num_visual_tokens
        pixel_shape = (1, 3, spec.image_size, spec.image_size)
    else:
        # Qwen VL family: load the checkpoint's own vision tower and wrap it with
        # StaticVisionEncoder for Qwen's 3D patchify + rotary position embeddings.
        from transformers import AutoModelForImageTextToText

        # ---- 2. Load HF model (vision part only) ----
        logging.info(f"Loading {spec.hf_model_id} for vision encoder extraction...")
        hf_model = AutoModelForImageTextToText.from_pretrained(
            spec.hf_model_id, dtype=torch.float32
        )
        hf_model = hf_model.eval()
        _patch_fast_pos_embed_interpolate(type(hf_model.model.visual))

        wrapper = StaticVisionEncoder(
            hf_model.model.visual,
            image_size=spec.image_size,
            patch_size=spec.patch_size,
            spatial_merge_size=spec.spatial_merge_size,
            temporal_patch_size=spec.temporal_patch_size,
            num_frames=num_frames,
            patchified_input=ane_friendly,
            linear_patch_embed=ane_friendly,
            f16_attention=ane_friendly,
        ).eval()
        del hf_model

        grid_t = wrapper.grid_t
        num_visual_tokens = spec.num_visual_tokens * grid_t
        if ane_friendly:
            pixel_shape = (1, wrapper.num_patches, wrapper.patch_dim)
        elif num_frames == 1:
            pixel_shape = (1, 3, spec.image_size, spec.image_size)
        else:
            pixel_shape = (1, 3 * num_frames, spec.image_size, spec.image_size)

        # ---- 3. Validate output shape before export ----
        with torch.no_grad():
            test_out = wrapper(torch.randn(*pixel_shape, dtype=torch.float32))
            # merger returns (hidden_states, deepstack_features) in newer transformers
            if isinstance(test_out, tuple):
                test_out = test_out[0]
            logging.info(
                f"Vision encoder output {tuple(test_out.shape)}; "
                f"expected [{num_visual_tokens}, {text_hidden}]"
            )

        # ---- 4. Wrap merger to handle tuple output ----
        if isinstance(wrapper(torch.randn(*pixel_shape)), tuple):
            original_merger = wrapper.merger

            class MergerWrapper(nn.Module):
                def __init__(self, merger):
                    super().__init__()
                    self.merger = merger

                def forward(self, x):
                    out = self.merger(x)
                    return out[0] if isinstance(out, tuple) else out

            wrapper.merger = MergerWrapper(original_merger)

        # ---- 5. Export ----
        if vision_dtype == "f16":
            wrapper = wrapper.half()
            export_module = BatchedF16VisionEncoder(wrapper, input_cast=torch.float16).eval()
        else:
            export_module = BatchedF16VisionEncoder(wrapper).eval()

    # Final-shape sanity check (batched + f16) before export.
    with torch.no_grad():
        final_out = export_module(torch.randn(*pixel_shape, dtype=torch.float32))
        logging.info(
            f"Export module output: {tuple(final_out.shape)} {final_out.dtype} "
            f"(expected (1, {num_visual_tokens}, {text_hidden}) torch.float16)"
        )

    if vision_compression.endswith("-palettized"):
        from coreai_models.export.compression import palettize_pytorch_model

        n_bits = int(vision_compression.split("bit")[0])
        logging.info(f"Palettizing the vision encoder ({n_bits}-bit, group 32)...")
        export_module = palettize_pytorch_model(
            export_module,
            (torch.randn(*pixel_shape, dtype=torch.float32),),
            {
                "global_config": {
                    "op_state_spec": {
                        "weight": {
                            "n_bits": n_bits,
                            "granularity": {
                                "type": "per_grouped_channel",
                                "axis": 0,
                                "group_size": 32,
                            },
                        }
                    }
                }
            },
        )
        logging.info("Vision palettization complete.")
    elif vision_compression != "none":
        import copy

        from coreai_models.export.compression import quantize_pytorch_model
        from coreai_models.export.presets import MACOS_PRESETS

        logging.info(
            f"Applying {vision_compression} weight-only quantization to the vision encoder..."
        )
        quant_cfg = copy.deepcopy(MACOS_PRESETS["4bit"]["torch_quantization_config"])
        if vision_compression == "8bit":
            quant_cfg["global_config"]["op_state_spec"]["weight"]["dtype"] = "int8"
        export_module = quantize_pytorch_model(
            export_module,
            (torch.randn(*pixel_shape, dtype=torch.float32),),
            None,
            quant_cfg,
            cache_seq_len=0,
            state_indices=(),
        )
        logging.info("Vision quantization complete.")

    reference_inputs = {"pixel_values": torch.randn(*pixel_shape, dtype=torch.float32)}

    logging.info(
        f"Exporting vision encoder "
        f"(input: {list(pixel_shape)} → output: [1,{num_visual_tokens},{text_hidden}] f16)..."
    )
    program = export_to_coreai(
        export_module,
        reference_inputs,
        dynamic_shapes=None,
        input_names=("pixel_values",),
        output_names=("image_features",),
        include_debug_info=include_debug_info,
    )

    # ---- 6. Save vision.aimodel ----
    vision_path = bundle_path / "vision.aimodel"
    if vision_path.exists() and not overwrite:
        raise FileExistsError(f"{vision_path} exists. Use --overwrite.")
    elif vision_path.exists():
        shutil.rmtree(vision_path)

    logging.info(f"Saving to {vision_path}...")
    build_meta = build_aimodel_metadata(spec.hf_model_id)
    await asyncio.to_thread(program.save_asset, vision_path, build_meta)

    # ---- 7. Patch metadata.json ----
    with open(bundle_path / "metadata.json") as f:
        metadata = json.load(f)
    metadata["assets"]["vision"] = "vision.aimodel"
    if num_frames > 1:
        metadata["vision"]["image_token_count"] = num_visual_tokens
        metadata["vision"]["max_video_frames"] = num_frames
        metadata["vision"]["tokens_per_frame"] = spec.num_visual_tokens
        metadata["vision"]["temporal_patch_size"] = spec.temporal_patch_size
    with open(bundle_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logging.info("Updated metadata.json with vision asset")
    return "vision.aimodel"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _find_repo_root() -> Path | None:
    """Walk up to the workspace root (where pyproject.toml + python/ live)."""
    d = Path(__file__).resolve().parent
    while d != d.parent:
        if (d / "pyproject.toml").exists() and (d / "python").exists():
            return d
        d = d.parent
    return None


def _default_output_dir() -> Path:
    root = _find_repo_root()
    return (root / "exports") if root is not None else Path("exports")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coreai.vlm.export",
        description="Export a vision-language model to Core AI format "
        "(text decoder + token embedding + vision encoder bundle).",
    )
    parser.add_argument(
        "model",
        nargs="?",
        help="Registry short-name (e.g. qwen3-vl). Run --list-models to see options.",
    )
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=4096,
        help="KV cache context length (default: 4096)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Truncate the text decoder to N layers (useful for debugging)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for the bundle (default: <repo-root>/exports/)",
    )
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument(
        "--skip-vision",
        action="store_true",
        help="Export only the text decoder + embedding (skip the vision encoder)",
    )
    stages.add_argument(
        "--vision-only",
        action="store_true",
        help="Export only the vision encoder into <output-dir>/<bundle>/, creating "
        "that bundle and its manifest when they do not exist yet (the route for a "
        "checkpoint whose text decoder has no export recipe). A bundle created this "
        "way holds the vision asset alone: no text decoder, embedding or tokenizer, "
        "so the Swift runner cannot load it by itself",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=1,
        help="Number of video frames for the vision encoder (default: 1 = single image). "
        "Must be divisible by temporal_patch_size (2 for Qwen). "
        "Multi-frame exports bake temporal position embeddings for native video support.",
    )
    parser.add_argument(
        "--vision-dtype",
        choices=["f32", "f16"],
        default="f32",
        help="Vision-encoder math dtype: f16 keeps the whole graph in half "
        "precision (required for ANE mapping); f32 keeps math in float32 (default)",
    )
    parser.add_argument(
        "--vision-ane-friendly",
        action="store_true",
        help="Export the vision encoder in ANE-mappable form: pre-patchified "
        "[1, num_patches, patch_dim] input (host does patchify), the Conv3d "
        "patch embed linearized, and dtype-preserving attention (no inline "
        "fp32 islands, so the graph compiles to a single ANE region)",
    )
    parser.add_argument(
        "--vision-compression",
        choices=["none", "4bit", "8bit", "4bit-palettized", "8bit-palettized"],
        default="none",
        help="Vision-encoder weight compression: int4/int8 symmetric per-block "
        "weight-only quantization, or 4/8-bit k-means palettization group 32 "
        "(the palettized form is the one the ANE path supports; default: none)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List supported VLM short-names and exit",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--include-debug-info",
        action="store_true",
        help=(
            "Embed debug information in the exported .aimodel for debugging a conversion. "
            "Default: off, which embeds minimum debug information and makes the "
            "exported asset smaller."
        ),
    )
    parser.add_argument(
        "--compression",
        default="none",
        help="Compression preset for the text decoder (e.g. '4bit', 'none'). "
        "The vision encoder and embedding are always exported at full precision.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    return parser


def _prepare_vision_only_bundle(spec: VLMSpec, output_dir: Path) -> Path:
    """Bundle directory for a vision-only export, with a manifest to patch.

    :func:`export_vision_encoder` writes into an existing bundle and updates its
    ``metadata.json``; a checkpoint exported without its text decoder never gets
    one from :func:`export_text_bundle`, so seed it from the spec.
    """
    bundle_path = output_dir / spec.output_name
    bundle_path.mkdir(parents=True, exist_ok=True)
    metadata_path = bundle_path / "metadata.json"
    if not metadata_path.exists():
        metadata = {
            "metadata_version": "0.2",
            "kind": "vlm",
            "name": spec.output_name,
            "assets": {},
            "vision": _vision_metadata(spec),
            "source": {
                "hf_model_id": spec.hf_model_id,
                "model_definition": "torch",
            },
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    return bundle_path


async def _run(spec: VLMSpec, args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir()
    if args.vision_only:
        bundle_path = _prepare_vision_only_bundle(spec, output_dir)
    else:
        bundle_path = await export_text_bundle(
            spec,
            max_ctx=args.max_context_length,
            num_layers=args.num_layers,
            output_dir=output_dir,
            overwrite=args.overwrite,
            compression=args.compression,
            include_debug_info=args.include_debug_info,
        )
    if not args.skip_vision:
        logging.info("Exporting vision encoder...")
        await export_vision_encoder(
            spec,
            bundle_path,
            args.overwrite,
            args.num_frames,
            include_debug_info=args.include_debug_info,
            vision_dtype=args.vision_dtype,
            ane_friendly=args.vision_ane_friendly,
            vision_compression=args.vision_compression,
        )
    return bundle_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.list_models:
        print("VLM model types:")
        print()
        for name, spec in SUPPORTED_MODELS.items():
            print(f"  {name:20s} {spec.hf_model_id}")
        return

    if not args.model:
        parser.error("model is required (unless using --list-models)")

    spec = SUPPORTED_MODELS.get(args.model)
    if spec is None:
        raise SystemExit(
            f"Error: '{args.model}' is not a supported VLM short-name. "
            f"Available: {', '.join(SUPPORTED_MODELS)}. Run --list-models."
        )

    if spec.text_decoder_class is None and not args.vision_only:
        raise SystemExit(
            f"Error: '{args.model}' has no text decoder recipe; it supports vision-encoder "
            f"export only. Run with --vision-only."
        )

    bundle_path = asyncio.run(_run(spec, args))
    print(f"\nExport complete: {bundle_path.resolve()}")


if __name__ == "__main__":
    main()
