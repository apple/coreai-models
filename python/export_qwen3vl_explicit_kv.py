#!/usr/bin/env python3
# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Export Qwen3-VL text decoder (inputs_embeds variant, explicit scatter KV cache).

Creates exports/qwen3_vl_2b_explicit_kv.llmasset/ with:
  - qwen3_vl_2b_explicit_kv.aimodel  (text decoder)
  - embed_tokens.bin                   (flat float16 embedding table)
  - tokenizer/                         (embedded HF tokenizer)
  - metadata.json                      (bundle manifest, kind=vlm)

Usage:
    cd <repo-root>
    uv run python python/export_qwen3vl_explicit_kv.py [--max-ctx 4096] [--num-layers N]
"""

import argparse
import asyncio
import json
import logging
import os
import re
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

from coreai_models.export.macos import export_to_coreai
from coreai_models.export.metadata import build_aimodel_metadata
from coreai_models.models.gpu.qwen3_vl import Qwen3VLForCausalLMEmbeddings
from coreai_models.primitives.macos.cache_scatter import KVCache as KVCacheScatter

HF_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
OUTPUT_NAME = "qwen3_vl_2b_explicit_kv"
IMAGE_TOKEN_ID = 151655  # <|image_pad|>
NUM_VISUAL_TOKENS = 196  # 448×448 / patch_size(16) / spatial_merge_size(2) squared


# ---------------------------------------------------------------------------
# Model subclass: use explicit scatter KV cache
# ---------------------------------------------------------------------------

class Qwen3VLEmbeddingsExplicitKV(Qwen3VLForCausalLMEmbeddings):
    """inputs_embeds decoder returning (logits, k_cache_updated, v_cache_updated).

    Uses slice_scatter KV cache so each call takes and returns the full cache
    explicitly — avoids GPU state mutations that cause Metal OOM on some hardware.
    """

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.IntTensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache = KVCacheScatter(k_cache, v_cache)
        out = self.model(inputs_embeds, position_ids, cache)
        logits = self.lm_head(out)
        if logits.dtype == torch.bfloat16:
            logits = logits.to(torch.float16)
        return logits, cache._k_cache, cache._v_cache


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
            for key in f.keys():
                if key.startswith("model.visual."):
                    continue
                if not key.startswith(prefix):
                    continue
                # Strip "model.language_model." → add "model."
                # "model.language_model.layers.0.self_attn.q_proj.weight" → "model.layers.0.*"
                stripped = key[len(prefix):]  # e.g. "layers.0.self_attn.q_proj.weight"
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
# embed_tokens extraction
# ---------------------------------------------------------------------------

def export_embed_tokens(bundle_path: Path, model_dir: str, hidden_size: int, vocab_size: int) -> str:
    """Save embed_tokens weights as flat float16 binary."""
    st_files = _get_safetensors_files(model_dir)
    embed_key = "model.language_model.embed_tokens.weight"
    weights = None
    for path in st_files:
        with safe_open(path, framework="pt", device="cpu") as f:
            if embed_key in f.keys():
                weights = f.get_tensor(embed_key).to(torch.float16)
                break
    if weights is None:
        raise RuntimeError(f"embed_tokens not found in safetensors (looked for '{embed_key}')")
    embed_path = bundle_path / "embed_tokens.bin"
    with open(embed_path, "wb") as f:
        f.write(weights.numpy().tobytes())
    logging.info(f"Saved embed_tokens.bin: {vocab_size} × {hidden_size} × f16")
    return "embed_tokens.bin"


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s: %(message)s")

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
        model_class=Qwen3VLEmbeddingsExplicitKV,
        hf_config=raw_cfg,
        model_dir=model_dir,
        max_ctx=max_ctx,
        num_layers=args.num_layers,
        dtype=torch.float16,
    )
    model = model.eval()
    logging.info("Model loaded.")

    # ---- 3. Build reference inputs (explicit KV I/O — avoids Metal state OOM) ----
    QUERY_LEN = 64
    OFFSET = 64
    inputs_embeds = torch.randn(1, QUERY_LEN, hidden_size, dtype=torch.float16)
    position_ids = (
        torch.arange(QUERY_LEN + OFFSET, dtype=torch.int32).unsqueeze(0)
    )

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
        "position_ids": {
            1: torch.export.Dim("seq_pos", min=QUERY_LEN, max=max_ctx - 1)
        },
        "k_cache": None,  # fixed size
        "v_cache": None,
    }

    # ---- 4. Export (explicit KV: returns updated caches as outputs) ----
    logging.info("Exporting text decoder to CoreAI format (explicit KV)...")
    program = export_to_coreai(
        model,
        reference_inputs,
        dynamic_shapes=dynamic_shapes,
        input_names=("inputs_embeds", "position_ids", "k_cache", "v_cache"),
        output_names=("logits", "k_cache", "v_cache"),
        state_names=None,
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

    # ---- 6. Embed tokens ----
    logging.info("Extracting embed_tokens.bin...")
    embed_rel = export_embed_tokens(bundle_path, model_dir, hidden_size, vocab_size)

    # ---- 7. Tokenizer ----
    logging.info("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.save_pretrained(str(bundle_path / "tokenizer"))

    # ---- 8. metadata.json ----
    metadata = {
        "metadata_version": "0.2",
        "kind": "vlm",
        "name": output_name,
        "assets": {
            "main": f"{output_name}.aimodel",
        },
        "language": {
            "tokenizer": HF_MODEL_ID,
            "vocab_size": vocab_size,
            "max_context_length": max_ctx,
            "embedded_tokenizer": True,
            "function_map": {"main": ["main"]},
        },
        "vision": {
            "image_token_id": IMAGE_TOKEN_ID,
            "num_visual_tokens": NUM_VISUAL_TOKENS,
            "hidden_size": hidden_size,
            "embed_tokens_path": embed_rel,
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
