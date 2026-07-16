# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""CLI entry point for ``coreai.vlm.export``.

Exports a vision-language model to Core AI format as a multi-asset bundle
(``<name>.llmasset/``):

  - ``<name>.aimodel``   text decoder (asset role ``main``, inputs_embeds, stateful KV)
  - ``embed.aimodel``    token-embedding lookup (asset role ``embedding``)
  - ``vision.aimodel``   vision encoder (asset role ``vision``, 448x448 static shapes)
  - ``tokenizer/``       embedded HF tokenizer
  - ``metadata.json``    bundle manifest (``kind=vlm``)

Usage:
    uv run coreai.vlm.export qwen3-vl [--max-context-length 4096] [--num-layers N]
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

from coreai_models.export.macos import export_to_coreai
from coreai_models.export.metadata import build_aimodel_metadata
from coreai_models.models.macos.qwen3_vl import Qwen3VLForCausalLMEmbeddings

# Core AI state names for the persistent KV cache.
KV_STATE_NAMES = ("k_cache", "v_cache")


@dataclass(frozen=True)
class VLMSpec:
    """Per-model export recipe, keyed by registry short-name.

    Carries the bits that vary between VL checkpoints: the HF id, the output
    bundle name, and the vision geometry (resolution, patch/merge sizes, the
    image placeholder token, and CLIP normalization stats) that drive both the
    vision-encoder export and the ``vision`` block of ``metadata.json``.
    """

    short_name: str
    hf_model_id: str
    output_name: str
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

    The HF Qwen3-VL checkpoint structure:
        model.language_model.embed_tokens.weight
        model.language_model.layers.N.self_attn.q_proj.weight  (etc.)
        model.language_model.norm.weight
        model.visual.*  (skipped)
    """
    # Set config
    text_cfg = model_class._get_reauthored_config(hf_config, max_ctx, num_layers)

    # Create model on meta device
    model = model_class(text_cfg, model_device="meta")
    model.to(dtype=dtype)

    # Build state dict from safetensors
    prefix = "model.language_model."
    layer_pattern = re.compile(r"layers\.(\d+)\.")
    st_files = _get_safetensors_files(model_dir)

    state_dict: dict[str, torch.Tensor] = {}
    for path in st_files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118 — safe_open has no __iter__/__contains__
                if key.startswith("model.visual."):
                    continue
                if not key.startswith(prefix):
                    continue
                # Strip "model.language_model." → add "model."
                # "model.language_model.layers.0.self_attn.q_proj.weight" → "model.layers.0.*"
                stripped = key[len(prefix) :]  # e.g. "layers.0.self_attn.q_proj.weight"
                model_key = "model." + stripped  # e.g. "model.layers.0.self_attn.q_proj.weight"
                # Skip layers beyond num_layers
                m = layer_pattern.match(stripped)
                if m and num_layers is not None and int(m.group(1)) >= num_layers:
                    continue
                tensor = f.get_tensor(key)
                if tensor.dtype not in (torch.float16, torch.int8) and "zero_point" not in key:
                    tensor = tensor.to(dtype)
                state_dict[model_key] = tensor

    # Fuse weights via _mutate_state_dict (handles keys in "model.layers.N.*" form)
    model._mutate_state_dict(state_dict)

    # Load (strict=False to allow tie_word_embeddings / missing embed_tokens)
    model.load_state_dict(state_dict, assign=True, strict=False)

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


def _load_embed_weight(model_dir: str) -> torch.Tensor:
    """Read the f16 embed_tokens weight table [vocab, hidden] from safetensors."""
    embed_key = "model.language_model.embed_tokens.weight"
    for path in _get_safetensors_files(model_dir):
        with safe_open(path, framework="pt", device="cpu") as f:
            if embed_key in f.keys():  # noqa: SIM118 — safe_open has no __contains__
                return f.get_tensor(embed_key).to(torch.float16)
    raise RuntimeError(f"embed_tokens not found in safetensors (looked for '{embed_key}')")


async def export_embed_model(
    spec: VLMSpec, bundle_path: Path, model_dir: str, max_ctx: int, overwrite: bool
) -> str:
    """Export the token-embedding lookup as embed.aimodel (asset role `embedding`)."""
    weight = _load_embed_weight(model_dir)
    vocab_size, hidden_size = weight.shape
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
    )
    program.optimize()

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


async def export_text_bundle(
    spec: VLMSpec,
    *,
    max_ctx: int,
    num_layers: int | None,
    output_dir: Path,
    overwrite: bool,
) -> Path:
    """Download weights and write the text portion of the VLM bundle.

    Produces ``<name>.aimodel`` (decoder), ``embed.aimodel``, ``tokenizer/``, and
    a ``metadata.json`` whose ``assets`` cover ``main``/``embedding``. The
    ``vision`` asset is added later by :func:`export_vision_encoder`.
    """
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
        model_class=Qwen3VLForCausalLMEmbeddings,
        hf_config=raw_cfg,
        model_dir=model_dir,
        max_ctx=max_ctx,
        num_layers=num_layers,
        dtype=torch.float16,
    )
    model = model.eval()
    logging.info("Model loaded.")

    # ---- 3. Build reference inputs (stateful KV: caches are in-place states) ----
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

    # ---- 4. Export (stateful KV: k_cache/v_cache surfaced as in-place states) ----
    logging.info("Exporting text decoder to CoreAI format (stateful KV)...")
    program = export_to_coreai(
        model,
        reference_inputs,
        dynamic_shapes=dynamic_shapes,
        input_names=("inputs_embeds", "position_ids"),
        output_names=("logits",),
        state_names=KV_STATE_NAMES,
    )
    logging.info("Optimizing AIProgram...")
    program.optimize()

    # ---- 5. Save bundle ----
    bundle_path = output_dir / (output_name + ".llmasset")
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
    embed_rel = await export_embed_model(spec, bundle_path, model_dir, max_ctx, overwrite)

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
        # Top-level `vision` block consumed by Swift VisionConfig (snake_case keys).
        "vision": {
            "image_size": spec.image_size,
            "patch_size": spec.patch_size,
            "image_token_count": spec.num_visual_tokens,
            "image_token_id": spec.image_token_id,
            "image_mean": list(spec.image_mean),
            "image_std": list(spec.image_std),
            "rescale_factor": spec.rescale_factor,
            "image_strategy": spec.image_strategy,
            "include_image_info": spec.include_image_info,
        },
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


class StaticVisionEncoder(nn.Module):
    """Vision encoder with pre-computed static position embeddings for a fixed grid.

    Avoids all data-dependent operations (linspace, repeat_interleave, etc.)
    by baking in the constant values at init time for the spec's resolution.

    Accepts raw CHW pixel values (the layout the Swift runner's ImagePreprocessor
    produces) and reproduces the Qwen image-processor patchify internally, so the
    runner needs no Qwen-specific preprocessing beyond resize + normalize.

    Input:  pixel_values  float32 [1, 3, image_size, image_size]  (CLIP-normalized, NCHW)
    Output: image_features float32 [num_visual_tokens, text_hidden]
    """

    def __init__(
        self,
        visual_model,
        *,
        image_size: int,
        patch_size: int,
        spatial_merge_size: int,
        temporal_patch_size: int,
    ) -> None:
        super().__init__()
        self.patch_embed = visual_model.patch_embed
        self.blocks = visual_model.blocks
        self.merger = visual_model.merger

        self.image_size = image_size
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.channels = 3
        # Fixed grid for image_size x image_size.
        self.grid_t = 1
        self.grid_h = image_size // patch_size
        self.grid_w = image_size // patch_size
        self.num_patches = self.grid_h * self.grid_w
        self.patch_dim = temporal_patch_size * self.channels * patch_size * patch_size

        grid_thw = torch.tensor([[self.grid_t, self.grid_h, self.grid_w]], dtype=torch.int32)

        with torch.no_grad():
            # Position embeddings [num_patches, vision_hidden]
            pos_embeds = visual_model.fast_pos_embed_interpolate(grid_thw)
            self.register_buffer("pos_embeds", pos_embeds)

            # Rotary position embeddings
            rotary_pos_emb = visual_model.rot_pos_emb(grid_thw)  # [num_patches, rot_dim/2]
            seq_len = rotary_pos_emb.shape[0]
            rotary_flat = rotary_pos_emb.reshape(seq_len, -1)
            emb = torch.cat([rotary_flat, rotary_flat], dim=-1)  # [num_patches, rot_dim]
            self.register_buffer("rot_cos", emb.cos())
            self.register_buffer("rot_sin", emb.sin())

            # cu_seqlens for variable-length attention: [0, num_patches]
            # For single image batch: [0, GRID_T * GRID_H * GRID_W]
            total_patches = self.grid_t * self.grid_h * self.grid_w
            cu = torch.tensor([0, total_patches], dtype=torch.int32)
            self.register_buffer("cu_seqlens", cu)

    def _patchify(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Turn NCHW pixels into Qwen's pre-patchified [num_patches, patch_dim].

        Reproduces the exact reshape/permute of Qwen2/3-VL's image processor
        (transpose order ``(0,3,6,4,7,2,1,5,8)``) so the resulting patch order
        matches both the precomputed ``pos_embeds`` and the merger's 2×2
        spatial-merge grouping. The single image is duplicated across the
        temporal dimension, matching the processor's last-frame repeat.
        """
        c, patch, merge = self.channels, self.patch_size, self.spatial_merge_size
        hw = self.image_size
        # [1, 3, H, W] → [3, H, W] → [temporal, 3, H, W]
        x = pixel_values.reshape(c, hw, hw)
        x = x.unsqueeze(0).repeat(self.temporal_patch_size, 1, 1, 1)
        # split H,W into (grid, merge, patch) and T into (grid_t, temporal)
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
        # pixel_values: [1, 3, H, W] (NCHW) → patchify → [num_patches, patch_dim]
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

    StaticVisionEncoder emits f32 [num_visual_tokens, text_hidden]; PR #65 expects
    f16/bf16 [1, image_token_count, hidden] (a leading batch dim, like embed.aimodel).
    The vision math stays in f32; only the final result is batched and cast to f16.
    """

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.encoder(pixel_values)
        if isinstance(out, tuple):
            out = out[0]
        return out.unsqueeze(0).to(torch.float16)


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


async def export_vision_encoder(spec: VLMSpec, bundle_path: Path, overwrite: bool) -> str:
    """Export the vision encoder as vision.aimodel and patch metadata.json."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLForConditionalGeneration as HFModel,
    )
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLVisionModel,
    )

    _patch_fast_pos_embed_interpolate(Qwen3VLVisionModel)

    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}. Export the text decoder first.")

    # ---- 1. Text hidden size (projection target) from the HF config ----
    text_hidden = AutoConfig.from_pretrained(spec.hf_model_id).text_config.hidden_size

    # ---- 2. Load HF model (vision part only) ----
    logging.info(f"Loading {spec.hf_model_id} for vision encoder extraction...")
    hf_model = HFModel.from_pretrained(spec.hf_model_id, dtype=torch.float32)
    hf_model = hf_model.eval()

    wrapper = StaticVisionEncoder(
        hf_model.model.visual,
        image_size=spec.image_size,
        patch_size=spec.patch_size,
        spatial_merge_size=spec.spatial_merge_size,
        temporal_patch_size=spec.temporal_patch_size,
    ).eval()
    del hf_model

    num_visual_tokens = spec.num_visual_tokens
    pixel_shape = (1, 3, spec.image_size, spec.image_size)

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
    export_module = BatchedF16VisionEncoder(wrapper).eval()

    # Final-shape sanity check (batched + f16) before export.
    with torch.no_grad():
        final_out = export_module(torch.randn(*pixel_shape, dtype=torch.float32))
        logging.info(
            f"Export module output: {tuple(final_out.shape)} {final_out.dtype} "
            f"(expected (1, {num_visual_tokens}, {text_hidden}) torch.float16)"
        )

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
    )
    logging.info("Optimizing AIProgram...")
    program.optimize()

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
    parser.add_argument(
        "--skip-vision",
        action="store_true",
        help="Export only the text decoder + embedding (skip the vision encoder)",
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
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    return parser


async def _run(spec: VLMSpec, args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir()
    bundle_path = await export_text_bundle(
        spec,
        max_ctx=args.max_context_length,
        num_layers=args.num_layers,
        output_dir=output_dir,
        overwrite=args.overwrite,
    )
    if not args.skip_vision:
        logging.info("Exporting vision encoder...")
        await export_vision_encoder(spec, bundle_path, args.overwrite)
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

    bundle_path = asyncio.run(_run(spec, args))
    print(f"\nExport complete: {bundle_path.resolve()}")


if __name__ == "__main__":
    main()
