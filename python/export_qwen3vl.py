#!/usr/bin/env python3
# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Export Qwen3-VL text decoder (inputs_embeds variant, stateful KV cache).

Creates exports/qwen3_vl_2b.llmasset/ with:
  - qwen3_vl_2b.aimodel                (text decoder, asset role `main`)
  - embed.aimodel                      (token-embedding lookup, asset role `embedding`)
  - tokenizer/                         (embedded HF tokenizer)
  - metadata.json                      (bundle manifest, kind=vlm)

Run export_vision_encoder_224.py afterwards to add the `vision` component.

Usage:
    cd <repo-root>
    uv run python python/export_qwen3vl.py [--max-ctx 4096] [--num-layers N]
"""

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

from coreai_models.export.macos import export_to_coreai
from coreai_models.export.metadata import build_aimodel_metadata
from coreai_models.models.gpu.qwen3_vl import Qwen3VLForCausalLMEmbeddings

HF_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
OUTPUT_NAME = "qwen3_vl_2b"
IMAGE_TOKEN_ID = 151655  # <|image_pad|>
NUM_VISUAL_TOKENS = 196  # 448×448 / patch_size(16) / spatial_merge_size(2) squared
IMAGE_SIZE = 448  # vision encoder input resolution (see export_vision_encoder_224.py)
PATCH_SIZE = 16

IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]
RESCALE_FACTOR = 1.0

# Core AI state names for the persistent KV cache
KV_STATE_NAMES = ("k_cache", "v_cache")


# ---------------------------------------------------------------------------
# Direct safetensors loader (avoids the hf_memory_efficient layer-regex issue)
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
    bundle_path: Path, model_dir: str, max_ctx: int, overwrite: bool
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
    meta = build_aimodel_metadata(HF_MODEL_ID)
    await asyncio.to_thread(program.save_asset, embed_path, meta)
    logging.info(f"Saved embed.aimodel: {vocab_size} × {hidden_size} × f16")
    return "embed.aimodel"


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------


async def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s"
    )

    max_ctx = args.max_ctx
    output_name = OUTPUT_NAME

    # ---- 1. Download weights + load config ----
    logging.info(f"Downloading {HF_MODEL_ID}...")
    model_dir = snapshot_download(
        HF_MODEL_ID,
        allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"],
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
        num_layers=args.num_layers,
        dtype=torch.float16,
    )
    model = model.eval()
    logging.info("Model loaded.")

    # ---- 3. Build reference inputs (stateful KV: caches are in-place states) ----
    QUERY_LEN = 64
    OFFSET = 64
    inputs_embeds = torch.randn(1, QUERY_LEN, hidden_size, dtype=torch.float16)
    position_ids = torch.arange(QUERY_LEN + OFFSET, dtype=torch.int32).unsqueeze(0)

    n_layers = args.num_layers or text_cfg.num_hidden_layers
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
    bundle_path = Path("exports") / (output_name + ".llmasset")
    bundle_path.mkdir(parents=True, exist_ok=True)
    aimodel_path = bundle_path / f"{output_name}.aimodel"

    if aimodel_path.exists() and not args.overwrite:
        raise FileExistsError(f"{aimodel_path} exists. Use --overwrite.")
    elif aimodel_path.exists():
        import shutil

        shutil.rmtree(aimodel_path)

    logging.info(f"Saving model to {aimodel_path}...")
    meta = build_aimodel_metadata(HF_MODEL_ID)
    await asyncio.to_thread(program.save_asset, aimodel_path, meta)
    del model

    # ---- 6. Embed model ----
    logging.info("Exporting embed.aimodel...")
    embed_rel = await export_embed_model(bundle_path, model_dir, max_ctx, args.overwrite)

    # ---- 7. Tokenizer ----
    logging.info("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.save_pretrained(str(bundle_path / "tokenizer"))

    # ---- 8. metadata.json ----
    # Asset roles match Swift ModelBundle.ComponentKey: `main` (decoder),
    # `embedding` (embed.aimodel), `vision` (added by export_vision_encoder_224.py).
    metadata = {
        "metadata_version": "0.2",
        "kind": "vlm",
        "name": output_name,
        "assets": {
            "main": f"{output_name}.aimodel",
            "embedding": embed_rel,
        },
        "language": {
            "tokenizer": HF_MODEL_ID,
            "vocab_size": vocab_size,
            "max_context_length": max_ctx,
            "embedded_tokenizer": True,
            "function_map": {"main": ["main"]},
        },
        # Top-level `vision` block consumed by Swift VisionConfig (snake_case keys).
        "vision": {
            "image_size": IMAGE_SIZE,
            "patch_size": PATCH_SIZE,
            "image_token_count": NUM_VISUAL_TOKENS,
            "image_token_id": IMAGE_TOKEN_ID,
            "image_mean": IMAGE_MEAN,
            "image_std": IMAGE_STD,
            "rescale_factor": RESCALE_FACTOR,
        },
        "source": {
            "hf_model_id": HF_MODEL_ID,
            "model_definition": "torch",
        },
    }
    with open(bundle_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logging.info(f"Bundle complete: {bundle_path}")
    print(f"\nExport done: {bundle_path.resolve()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-ctx", type=int, default=4096, help="KV cache context length")
    p.add_argument("--num-layers", type=int, default=None, help="Truncate to N layers (debug)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
